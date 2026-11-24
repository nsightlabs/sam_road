import numpy as np
import pandas as pd
import os
import torch
import torch.multiprocessing as mp
import cv2
from utils import load_config, create_output_dir_and_save_config
from dataset import read_rgb_img, get_patch_info_one_img
from model import SAMRoad
from tqdm import tqdm
import graph_extraction
import graph_utils
import triage
import pickle
import scipy
import rtree
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import time
from uuid import uuid4
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument(
    "--checkpoint", default=None, help="checkpoint of the model to test."
)
parser.add_argument(
    "--config", default=None, help="model config."
)
parser.add_argument(
    "--output_dir", default=None, help="Name of the output dir, if not specified will use timestamp"
)
parser.add_argument(
    "--submit", action="store_true", help="Whether to generate submission file"
)
parser.add_argument(
    "--save_output", action="store_true", help="Whether to save output images and graphs"
)
parser.add_argument("--device", default="cuda", help="device to use for inference (cuda or cpu)")
parser.add_argument(
    "--num_gpus", type=int, default=0, help="number of GPUs to use, 0 = all visible GPUs"
)
parser.add_argument(
    "--img_batch", type=int, default=1,
    help="images per forward pass. >1 enables cross-image batching (only for images that "
         "produce exactly one patch covering the whole image; others fall back to the "
         "per-image path automatically)."
)
parser.add_argument(
    "--io_threads", type=int, default=4, help="threads used to read images in each worker"
)
args = parser.parse_args()


def crop_img_patch(img, x0, y0, x1, y1):
    return img[y0:y1, x0:x1, :]


def get_batch_img_patches(img, batch_patch_info):
    patches = []
    for _, (x0, y0), (x1, y1) in batch_patch_info:
        patch = crop_img_patch(img, x0, y0, x1, y1)
        patches.append(torch.tensor(patch, dtype=torch.float32))
    batch = torch.stack(patches, 0).contiguous()
    return batch


# --------------------------------------------------------------------------- #
# Per-image inference (original path, works for any number of patches)
# --------------------------------------------------------------------------- #
def infer_one_img(net, img, config):
    # NOTE: uses the global args.device, which each worker sets for its own GPU.
    image_size = img.shape[0]

    batch_size = config.INFER_BATCH_SIZE
    # list of (i, (x_begin, y_begin), (x_end, y_end))
    all_patch_info = get_patch_info_one_img(
        0, image_size, config.SAMPLE_MARGIN, config.PATCH_SIZE, config.INFER_PATCHES_PER_EDGE)
    patch_num = len(all_patch_info)
    batch_num = (
        patch_num // batch_size
        if patch_num % batch_size == 0
        else patch_num // batch_size + 1
    )

    # [IMG_H, IMG_W]
    fused_keypoint_mask = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device, non_blocking=False)
    fused_road_mask = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device, non_blocking=False)
    pixel_counter = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device, non_blocking=False)

    # stores img embeddings for toponet
    # list of [B, D, h, w], len=batch_num
    img_features = list()

    for batch_index in range(batch_num):
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset : offset + batch_size]
        # tensor [B, H, W, C]
        batch_img_patches = get_batch_img_patches(img, batch_patch_info)

        with torch.no_grad():
            batch_img_patches = batch_img_patches.to(args.device, non_blocking=False)
            # [B, H, W, 2]
            mask_scores, patch_img_features = net.infer_masks_and_img_features(batch_img_patches)
            img_features.append(patch_img_features)
        # Aggregate masks
        for patch_index, patch_info in enumerate(batch_patch_info):
            _, (x0, y0), (x1, y1) = patch_info
            keypoint_patch, road_patch = mask_scores[patch_index, :, :, 0], mask_scores[patch_index, :, :, 1]
            fused_keypoint_mask[y0:y1, x0:x1] += keypoint_patch
            fused_road_mask[y0:y1, x0:x1] += road_patch
            pixel_counter[y0:y1, x0:x1] += torch.ones(road_patch.shape[0:2], dtype=torch.float32, device=args.device)

    fused_keypoint_mask /= pixel_counter
    fused_road_mask /= pixel_counter
    # range 0-1 -> 0-255
    fused_keypoint_mask = (fused_keypoint_mask * 255).to(torch.uint8).cpu().numpy()
    fused_road_mask = (fused_road_mask * 255).to(torch.uint8).cpu().numpy()

    ## Extract sample points from masks
    graph_points = graph_extraction.extract_graph_points(fused_keypoint_mask, fused_road_mask, config)
    if graph_points.shape[0] == 0:
        return graph_points, np.zeros((0, 2), dtype=np.int32), fused_keypoint_mask, fused_road_mask

    # for box query
    graph_rtree = rtree.index.Index()
    for i, v in enumerate(graph_points):
        x, y = v
        # hack to insert single points
        graph_rtree.insert(i, (x, y, x, y))

    ## Pass 2: infer toponet to predict topology of points from stored img features
    edge_scores = defaultdict(float)
    edge_counts = defaultdict(float)
    for batch_index in range(batch_num):
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset : offset + batch_size]

        topo_data = {
            'points': [],
            'pairs': [],
            'valid': [],
        }
        idx_maps = []

        # prepares pairs queries
        for patch_info in batch_patch_info:
            _, (x0, y0), (x1, y1) = patch_info
            patch_point_indices = list(graph_rtree.intersection((x0, y0, x1, y1)))
            idx_patch2all = {patch_idx : all_idx for patch_idx, all_idx in enumerate(patch_point_indices)}
            patch_point_num = len(patch_point_indices)
            # normalize into patch
            patch_points = graph_points[patch_point_indices, :] - np.array([[x0, y0]], dtype=graph_points.dtype)
            # for knn and circle query
            patch_kdtree = scipy.spatial.KDTree(patch_points)

            # k+1 because the nearest one is always self
            # idx is to the patch subgraph
            knn_d, knn_idx = patch_kdtree.query(patch_points, k=config.MAX_NEIGHBOR_QUERIES + 1, distance_upper_bound=config.NEIGHBOR_RADIUS)
            # [patch_point_num, n_nbr]
            knn_idx = knn_idx[:, 1:]  # removes self
            # [patch_point_num, n_nbr] idx is to the patch subgraph
            src_idx = np.tile(
                np.arange(patch_point_num)[:, np.newaxis],
                (1, config.MAX_NEIGHBOR_QUERIES)
            )
            valid = knn_idx < patch_point_num
            tgt_idx = np.where(valid, knn_idx, src_idx)
            # [patch_point_num, n_nbr, 2]
            pairs = np.stack([src_idx, tgt_idx], axis=-1)

            topo_data['points'].append(patch_points)
            topo_data['pairs'].append(pairs)
            topo_data['valid'].append(valid)
            idx_maps.append(idx_patch2all)

        # collate
        collated = {}
        for key, x_list in topo_data.items():
            length = max([x.shape[0] for x in x_list])
            collated[key] = np.stack([
                np.pad(x, [(0, length - x.shape[0])] + [(0, 0)] * (len(x.shape) - 1))
                for x in x_list
            ], axis=0)

        # skips this batch if there's no points
        if collated['points'].shape[1] == 0:
            continue

        # infer toponet
        # [B, D, h, w]
        batch_features = img_features[batch_index]
        # [B, N_sample, N_pair, 2]
        batch_points = torch.tensor(collated['points'], device=args.device)
        batch_pairs = torch.tensor(collated['pairs'], device=args.device)
        batch_valid = torch.tensor(collated['valid'], device=args.device)

        with torch.no_grad():
            # [B, N_samples, N_pairs, 1]
            topo_scores = net.infer_toponet(batch_features, batch_points, batch_pairs, batch_valid)

        # all-invalid (padded, no neighbors) queries returns nan scores
        # [B, N_samples, N_pairs]
        topo_scores = torch.where(torch.isnan(topo_scores), -100.0, topo_scores).squeeze(-1).cpu().numpy()

        # aggregate edge scores
        batch_size, n_samples, n_pairs = topo_scores.shape
        for bi in range(batch_size):
            for si in range(n_samples):
                for pi in range(n_pairs):
                    if not collated['valid'][bi, si, pi]:
                        continue
                    # idx to the full graph
                    src_idx_patch, tgt_idx_patch = collated['pairs'][bi, si, pi, :]
                    src_idx_all, tgt_idx_all = idx_maps[bi][src_idx_patch], idx_maps[bi][tgt_idx_patch]
                    edge_score = topo_scores[bi, si, pi]
                    assert 0.0 <= edge_score <= 1.0
                    edge_scores[(src_idx_all, tgt_idx_all)] += edge_score
                    edge_counts[(src_idx_all, tgt_idx_all)] += 1.0

    # avg edge scores and filter
    pred_edges = []
    for edge, score_sum in edge_scores.items():
        score = score_sum / edge_counts[edge]
        if score > config.TOPO_THRESHOLD:
            pred_edges.append(edge)
    pred_edges = np.array(pred_edges).reshape(-1, 2)
    pred_nodes = graph_points[:, ::-1]  # to rc

    return pred_nodes, pred_edges, fused_keypoint_mask, fused_road_mask


# --------------------------------------------------------------------------- #
# Cross-image batched inference (single patch per image)
# --------------------------------------------------------------------------- #
_single_patch_cache = {}


def is_single_patch(size, config):
    """True if an image of this size yields exactly one patch that covers all of it."""
    if size not in _single_patch_cache:
        info = get_patch_info_one_img(
            0, size, config.SAMPLE_MARGIN, config.PATCH_SIZE, config.INFER_PATCHES_PER_EDGE)
        ok = False
        if len(info) == 1:
            _, (x0, y0), (x1, y1) = info[0]
            ok = (x0, y0, x1, y1) == (0, 0, size, size)
        _single_patch_cache[size] = ok
    return _single_patch_cache[size]


def can_batch(imgs, config):
    if len(imgs) < 2:
        return False
    shapes = {im.shape for im in imgs}
    if len(shapes) != 1:
        return False
    h, w = imgs[0].shape[:2]
    return h == w and is_single_patch(h, config)


def infer_images_batched(net, imgs, config):
    """
    imgs: list of [H, W, C] arrays, same size, one patch each (see can_batch).
    Returns a list of (pred_nodes (r, c), pred_edges, keypoint_mask, road_mask).
    With one patch per image there is nothing to average, so edge scores are
    thresholded directly.
    """
    # ---- pass 1: one forward pass for K images ----
    batch = torch.stack([torch.tensor(im, dtype=torch.float32) for im in imgs]).to(args.device)
    with torch.no_grad():
        mask_scores, feats = net.infer_masks_and_img_features(batch)   # [K,H,W,2], [K,D,h,w]
    masks = (mask_scores * 255).to(torch.uint8).cpu().numpy()

    # ---- per-image CPU work: graph points + neighbour pairs ----
    K = len(imgs)
    results = [None] * K
    pts_l, pairs_l, valid_l, owners = [], [], [], []
    for i in range(K):
        kp, road = masks[i, :, :, 0], masks[i, :, :, 1]
        pts = graph_extraction.extract_graph_points(kp, road, config)
        if pts.shape[0] == 0:
            results[i] = (pts, np.zeros((0, 2), dtype=np.int32), kp, road)
            continue
        n = pts.shape[0]
        tree = scipy.spatial.KDTree(pts)
        _, knn = tree.query(pts, k=config.MAX_NEIGHBOR_QUERIES + 1,
                            distance_upper_bound=config.NEIGHBOR_RADIUS)
        knn = knn[:, 1:]  # removes self
        src = np.tile(np.arange(n)[:, None], (1, config.MAX_NEIGHBOR_QUERIES))
        valid = knn < n
        tgt = np.where(valid, knn, src)
        pts_l.append(pts)
        pairs_l.append(np.stack([src, tgt], -1))
        valid_l.append(valid)
        owners.append(i)
        results[i] = (pts[:, ::-1], None, kp, road)  # nodes in (r, c); edges filled below

    # ---- pass 2: one toponet call for all images that have points ----
    if owners:
        L = max(p.shape[0] for p in pts_l)

        def pad(a):
            return np.pad(a, [(0, L - a.shape[0])] + [(0, 0)] * (a.ndim - 1))

        bp = torch.tensor(np.stack([pad(a) for a in pts_l]), device=args.device)
        bpair = torch.tensor(np.stack([pad(a) for a in pairs_l]), device=args.device)
        bvalid = torch.tensor(np.stack([pad(a) for a in valid_l]), device=args.device)
        with torch.no_grad():
            scores = net.infer_toponet(feats[owners], bp, bpair, bvalid)
        # all-invalid (padded, no neighbors) queries return nan scores
        scores = torch.where(torch.isnan(scores), -100.0, scores).squeeze(-1).cpu().numpy()

        for j, i in enumerate(owners):
            n = pts_l[j].shape[0]
            keep = valid_l[j] & (scores[j, :n] > config.TOPO_THRESHOLD)
            edges = pairs_l[j][keep]  # [E, 2]
            nodes, _, kp, road = results[i]
            results[i] = (nodes, edges.reshape(-1, 2), kp, road)
    return results


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def handle_result(img_id, img, pred_nodes, pred_edges, itsc_mask, road_mask,
                  config, output_dir, rows):
    img_size = img.shape[0]

    if args.save_output:
        # fused masks
        mask_save_dir = os.path.join(output_dir, 'mask')
        os.makedirs(mask_save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(mask_save_dir, f'{img_id}_road.png'), road_mask)
        cv2.imwrite(os.path.join(mask_save_dir, f'{img_id}_itsc.png'), itsc_mask)

        # visualizes merged large map
        viz_save_dir = os.path.join(output_dir, 'viz')
        os.makedirs(viz_save_dir, exist_ok=True)
        viz_img = triage.visualize_image_and_graph(np.copy(img), pred_nodes / img_size, pred_edges, img_size)
        cv2.imwrite(os.path.join(viz_save_dir, f'{img_id}.png'), viz_img)

    # WKT must be built BEFORE the sat2graph row flip below
    if args.submit:
        graph_wkt = triage.graph_to_wkt(pred_nodes / img_size, pred_edges, img_size)
        rows.append({'ID': img_id, 'geometry': graph_wkt})

    if args.save_output:
        if config.DATASET in ['spacenet', 'hotosm']:
            pred_nodes = np.stack([400 - pred_nodes[:, 0], pred_nodes[:, 1]], axis=1)
        large_map_sat2graph_format = graph_utils.convert_to_sat2graph_format(pred_nodes, pred_edges)
        graph_save_dir = os.path.join(output_dir, 'graph')
        os.makedirs(graph_save_dir, exist_ok=True)
        graph_save_path = os.path.join(graph_save_dir, f'{img_id}.p')
        with open(graph_save_path, 'wb') as file:
            pickle.dump(large_map_sat2graph_format, file)


# --------------------------------------------------------------------------- #
# Worker: one process per GPU
# --------------------------------------------------------------------------- #
def worker(rank, world_size, output_dir, img_ids):
    """Runs inference on img_ids[rank::world_size] using one device."""
    # infer_one_img / infer_images_batched read this global, so each process
    # points it at its own device.
    if args.device != "cpu":
        args.device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    config = load_config(args.config)

    # Builds eval model
    net = SAMRoad(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if rank == 0:
        print(f'##### Loading Trained CKPT {args.checkpoint} #####')
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()
    net.to(args.device)

    if config.DATASET == 'cityscale':
        rgb_pattern = './cityscale/20cities/region_{}_sat.png'
    elif config.DATASET == 'spacenet':
        rgb_pattern = './spacenet/RGB_1.0_meter/{}__rgb.png'
    elif config.DATASET == 'hotosm':
        rgb_pattern = './hotosm/RGB_1.0_meter/{}__rgb.png'

    my_ids = img_ids[rank::world_size]
    K = max(1, args.img_batch)
    rows = []
    total_inference_seconds = 0.0

    pbar = tqdm(total=len(my_ids), desc=f"Inference [{args.device}]", position=rank)
    with ThreadPoolExecutor(max(1, args.io_threads)) as pool:
        for s in range(0, len(my_ids), K):
            ids = my_ids[s:s + K]
            # [H, W, C] RGB
            imgs = list(pool.map(lambda i: read_rgb_img(rgb_pattern.format(i)), ids))

            start_seconds = time.time()
            if can_batch(imgs, config):
                outs = infer_images_batched(net, imgs, config)
            else:
                # coords in (r, c)
                outs = [infer_one_img(net, im, config) for im in imgs]
            if args.device != "cpu":
                torch.cuda.synchronize()
            total_inference_seconds += time.time() - start_seconds

            for img_id, img, (pred_nodes, pred_edges, itsc_mask, road_mask) in zip(ids, imgs, outs):
                handle_result(img_id, img, pred_nodes, pred_edges, itsc_mask, road_mask,
                              config, output_dir, rows)
            pbar.update(len(ids))
    pbar.close()

    # per-worker results, merged by the main process
    pd.DataFrame(rows, columns=['ID', 'geometry']).to_csv(
        os.path.join(output_dir, f'part_{rank}.csv'), index=False)
    with open(os.path.join(output_dir, f'time_{rank}.txt'), 'w') as f:
        f.write(str(total_inference_seconds))


if __name__ == "__main__":
    config = load_config(args.config)

    output_dir_prefix = './save/predict_'
    if args.output_dir:
        output_dir = create_output_dir_and_save_config(output_dir_prefix, config, specified_dir=f'./save/{args.output_dir}')
    else:
        output_dir = create_output_dir_and_save_config(output_dir_prefix, config)

    # create dirs once here to avoid races between workers
    if args.save_output:
        for d in ("mask", "viz", "graph"):
            os.makedirs(os.path.join(output_dir, d), exist_ok=True)

    img_ids = list(config.PREDICT_IMG_INDICES)

    if args.device == "cpu":
        n_workers = 1
    else:
        avail = torch.cuda.device_count()
        n_workers = min(args.num_gpus, avail) if args.num_gpus > 0 else avail
        n_workers = max(1, n_workers)
    n_workers = min(n_workers, max(1, len(img_ids)))

    wall_start = time.time()
    if n_workers > 1:
        mp.spawn(worker, args=(n_workers, output_dir, img_ids), nprocs=n_workers, join=True)
    else:
        worker(0, 1, output_dir, img_ids)
    wall_seconds = time.time() - wall_start

    # log inference time
    summed = 0.0
    for r in range(n_workers):
        tpath = os.path.join(output_dir, f'time_{r}.txt')
        with open(tpath) as f:
            summed += float(f.read())
        os.remove(tpath)
    time_txt = (f'Inference completed for {args.config} in {wall_seconds:.1f} seconds wall-clock '
                f'({summed:.1f} seconds model+postprocess time summed over {n_workers} worker(s), '
                f'img_batch={args.img_batch}).')
    print(time_txt)
    with open(os.path.join(output_dir, 'inference_time.txt'), 'w') as f:
        f.write(time_txt)

    # merge submission file
    if args.submit:
        parts = []
        for r in range(n_workers):
            part_path = os.path.join(output_dir, f'part_{r}.csv')
            parts.append(pd.read_csv(part_path, dtype=str, keep_default_na=False))
            os.remove(part_path)
        submission_df = pd.concat(parts, ignore_index=True)
        # restore original image order
        order = {str(i): k for k, i in enumerate(img_ids)}
        submission_df = submission_df.sort_values('ID', key=lambda s: s.map(order)).reset_index(drop=True)
        fp = os.path.join(output_dir, str(uuid4()) + '.csv')
        submission_df.to_csv(fp, index=False)
        print(f"Saved submission file to: {fp}")
    else:
        for r in range(n_workers):
            part_path = os.path.join(output_dir, f'part_{r}.csv')
            if os.path.exists(part_path):
                os.remove(part_path)