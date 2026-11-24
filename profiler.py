import numpy as np
import pandas as pd
import os
import torch
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
from contextlib import contextmanager
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
parser.add_argument("--device", default="cuda", help="device to use for inference")
parser.add_argument(
    "--warmup", type=int, default=1,
    help="number of initial images excluded from the profile (CUDA / cudnn.benchmark warm-up)"
)
parser.add_argument(
    "--max_images", type=int, default=0, help="only process the first N images (0 = all), handy for quick profiling"
)
args = parser.parse_args()


# --------------------------------------------------------------------------- #
# Profiler
# --------------------------------------------------------------------------- #
class Profiler:
    """
    Accumulates wall time per named section. On CUDA it synchronizes before and
    after each section so asynchronous GPU work is attributed to the section
    that launched it, not to whichever later section happens to block on it.
    (Sync slightly slows the run; that's the price of accurate numbers.)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.t = defaultdict(float)
        self.n = defaultdict(int)
        self.stats = defaultdict(list)
        self.wall = 0.0
        self.images = 0

    @staticmethod
    def _sync():
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(args.device))

    @contextmanager
    def section(self, name):
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.t[name] += time.perf_counter() - t0
            self.n[name] += 1

    def add_image(self, seconds):
        self.wall += seconds
        self.images += 1

    def report(self):
        lines = []
        accounted = sum(self.t.values())
        unaccounted = max(0.0, self.wall - accounted)
        lines.append(f"Profiled {self.images} image(s), total wall time {self.wall:.2f}s "
                     f"({self.wall / max(1, self.images):.3f}s per image)")
        lines.append("")
        lines.append(f"{'section':<22}{'total(s)':>10}{'calls':>8}{'mean(ms)':>11}{'% wall':>9}")
        lines.append("-" * 60)
        rows = sorted(self.t.items(), key=lambda kv: -kv[1])
        for name, total in rows:
            calls = self.n[name]
            lines.append(f"{name:<22}{total:>10.2f}{calls:>8d}{1000 * total / calls:>11.1f}"
                         f"{100 * total / max(self.wall, 1e-9):>8.1f}%")
        lines.append(f"{'(unaccounted)':<22}{unaccounted:>10.2f}{'':>8}{'':>11}"
                     f"{100 * unaccounted / max(self.wall, 1e-9):>8.1f}%")
        if self.stats:
            lines.append("")
            lines.append(f"{'per-image stat':<22}{'mean':>10}{'min':>10}{'max':>10}")
            lines.append("-" * 52)
            for name, vals in self.stats.items():
                v = np.asarray(vals, dtype=float)
                lines.append(f"{name:<22}{v.mean():>10.1f}{v.min():>10.0f}{v.max():>10.0f}")
        return "\n".join(lines), rows, unaccounted


prof = Profiler()


def crop_img_patch(img, x0, y0, x1, y1):
    return img[y0:y1, x0:x1, :]


def get_batch_img_patches(img, batch_patch_info):
    patches = []
    for _, (x0, y0), (x1, y1) in batch_patch_info:
        patch = crop_img_patch(img, x0, y0, x1, y1)
        patches.append(torch.tensor(patch, dtype=torch.float32))
    batch = torch.stack(patches, 0).contiguous()
    return batch


def infer_one_img(net, img, config):
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
    prof.stats['patches/image'].append(patch_num)

    # ---------------- pass 1: masks ----------------
    with prof.section("mask_alloc"):
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

        with prof.section("patch_prep_cpu"):
            # tensor [B, H, W, C]
            batch_img_patches = get_batch_img_patches(img, batch_patch_info)

        with prof.section("pass1_model"):  # host->device copy + image encoder + mask decoder
            with torch.no_grad():
                batch_img_patches = batch_img_patches.to(args.device, non_blocking=False)
                # [B, H, W, 2]
                mask_scores, patch_img_features = net.infer_masks_and_img_features(batch_img_patches)
                img_features.append(patch_img_features)

        with prof.section("mask_fuse"):
            for patch_index, patch_info in enumerate(batch_patch_info):
                _, (x0, y0), (x1, y1) = patch_info
                keypoint_patch, road_patch = mask_scores[patch_index, :, :, 0], mask_scores[patch_index, :, :, 1]
                fused_keypoint_mask[y0:y1, x0:x1] += keypoint_patch
                fused_road_mask[y0:y1, x0:x1] += road_patch
                pixel_counter[y0:y1, x0:x1] += torch.ones(road_patch.shape[0:2], dtype=torch.float32, device=args.device)

    with prof.section("mask_to_cpu"):
        fused_keypoint_mask /= pixel_counter
        fused_road_mask /= pixel_counter
        # range 0-1 -> 0-255
        fused_keypoint_mask = (fused_keypoint_mask * 255).to(torch.uint8).cpu().numpy()
        fused_road_mask = (fused_road_mask * 255).to(torch.uint8).cpu().numpy()

    # ---------------- graph points ----------------
    with prof.section("extract_points"):
        graph_points = graph_extraction.extract_graph_points(fused_keypoint_mask, fused_road_mask, config)
    prof.stats['graph_points/image'].append(graph_points.shape[0])
    if graph_points.shape[0] == 0:
        prof.stats['edge_candidates/image'].append(0)
        prof.stats['pred_edges/image'].append(0)
        return graph_points, np.zeros((0, 2), dtype=np.int32), fused_keypoint_mask, fused_road_mask

    with prof.section("rtree_build"):
        # for box query
        graph_rtree = rtree.index.Index()
        for i, v in enumerate(graph_points):
            x, y = v
            # hack to insert single points
            graph_rtree.insert(i, (x, y, x, y))

    # ---------------- pass 2: toponet ----------------
    edge_scores = defaultdict(float)
    edge_counts = defaultdict(float)
    edge_candidates = 0
    for batch_index in range(batch_num):
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset : offset + batch_size]

        topo_data = {
            'points': [],
            'pairs': [],
            'valid': [],
        }
        idx_maps = []

        with prof.section("topo_prep_cpu"):  # rtree query + KDTree knn + padding/collate
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
        edge_candidates += int(collated['valid'].sum())

        # infer toponet
        # [B, D, h, w]
        batch_features = img_features[batch_index]
        with prof.section("topo_model"):  # host->device copies + toponet forward
            # [B, N_sample, N_pair, 2]
            batch_points = torch.tensor(collated['points'], device=args.device)
            batch_pairs = torch.tensor(collated['pairs'], device=args.device)
            batch_valid = torch.tensor(collated['valid'], device=args.device)

            with torch.no_grad():
                # [B, N_samples, N_pairs, 1]
                topo_scores = net.infer_toponet(batch_features, batch_points, batch_pairs, batch_valid)

        with prof.section("topo_to_cpu"):
            # all-invalid (padded, no neighbors) queries returns nan scores
            # [B, N_samples, N_pairs]
            topo_scores = torch.where(torch.isnan(topo_scores), -100.0, topo_scores).squeeze(-1).cpu().numpy()

        with prof.section("edge_agg_loop"):  # the triple Python loop
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

    with prof.section("edge_filter"):
        # avg edge scores and filter
        pred_edges = []
        for edge, score_sum in edge_scores.items():
            score = score_sum / edge_counts[edge]
            if score > config.TOPO_THRESHOLD:
                pred_edges.append(edge)
        pred_edges = np.array(pred_edges).reshape(-1, 2)
        pred_nodes = graph_points[:, ::-1]  # to rc

    prof.stats['edge_candidates/image'].append(edge_candidates)
    prof.stats['pred_edges/image'].append(pred_edges.shape[0])

    return pred_nodes, pred_edges, fused_keypoint_mask, fused_road_mask


if __name__ == "__main__":
    config = load_config(args.config)
    predict_img_indices = list(config.PREDICT_IMG_INDICES)
    if args.max_images > 0:
        predict_img_indices = predict_img_indices[:args.max_images]
    # Builds eval model
    device = torch.device("cuda") if args.device == "cuda" else torch.device(args.device)
    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    net = SAMRoad(config)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    print(f'##### Loading Trained CKPT {args.checkpoint} #####')
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()
    net.to(device)

    if config.DATASET == 'cityscale':
        rgb_pattern = './cityscale/20cities/region_{}_sat.png'
    elif config.DATASET == 'spacenet':
        rgb_pattern = './spacenet/RGB_1.0_meter/{}__rgb.png'
    elif config.DATASET == 'hotosm':
        rgb_pattern = './hotosm/RGB_1.0_meter/{}__rgb.png'

    output_dir_prefix = './save/predict_'
    if args.output_dir:
        output_dir = create_output_dir_and_save_config(output_dir_prefix, config, specified_dir=f'./save/{args.output_dir}')
    else:
        output_dir = create_output_dir_and_save_config(output_dir_prefix, config)

    total_inference_seconds = 0.0

    if args.submit: tmp = []
    for idx, img_id in enumerate(tqdm(predict_img_indices, total=len(predict_img_indices), desc="Inference")):
        t_img = time.perf_counter()

        # [H, W, C] RGB
        with prof.section("io_read"):
            img = read_rgb_img(rgb_pattern.format(img_id))

        start_seconds = time.time()
        # coords in (r, c)
        pred_nodes, pred_edges, itsc_mask, road_mask = infer_one_img(net, img, config)
        end_seconds = time.time()
        total_inference_seconds += (end_seconds - start_seconds)

        img_size = img.shape[0]

        if args.save_output:
            with prof.section("save_masks"):
                mask_save_dir = os.path.join(output_dir, 'mask')
                os.makedirs(mask_save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(mask_save_dir, f'{img_id}_road.png'), road_mask)
                cv2.imwrite(os.path.join(mask_save_dir, f'{img_id}_itsc.png'), itsc_mask)

            with prof.section("save_viz"):
                viz_save_dir = os.path.join(output_dir, 'viz')
                os.makedirs(viz_save_dir, exist_ok=True)
                viz_img = triage.visualize_image_and_graph(np.copy(img), pred_nodes / img_size, pred_edges, img_size)
                cv2.imwrite(os.path.join(viz_save_dir, f'{img_id}.png'), viz_img)

        # WKT must be built BEFORE the sat2graph row flip below
        if args.submit:
            with prof.section("wkt"):
                graph_wkt = triage.graph_to_wkt(pred_nodes / img_size, pred_edges, img_size)
                tmp.append({
                    'ID': img_id,
                    'geometry': graph_wkt
                })

        # Saves the large map
        if args.save_output:
            with prof.section("save_graph"):
                if config.DATASET in ['spacenet', 'hotosm']:
                    pred_nodes = np.stack([400 - pred_nodes[:, 0], pred_nodes[:, 1]], axis=1)
                large_map_sat2graph_format = graph_utils.convert_to_sat2graph_format(pred_nodes, pred_edges)
                graph_save_dir = os.path.join(output_dir, 'graph')
                os.makedirs(graph_save_dir, exist_ok=True)
                graph_save_path = os.path.join(graph_save_dir, f'{img_id}.p')
                with open(graph_save_path, 'wb') as file:
                    pickle.dump(large_map_sat2graph_format, file)

        prof.add_image(time.perf_counter() - t_img)

        # drop the warm-up images from the profile
        if args.warmup > 0 and idx + 1 == args.warmup:
            prof.reset()

    # log inference time
    time_txt = f'Inference completed for {args.config} in {total_inference_seconds} seconds.'
    print(time_txt)
    with open(os.path.join(output_dir, 'inference_time.txt'), 'w') as f:
        f.write(time_txt)

    # profile report
    if prof.images > 0:
        report_txt, rows, unaccounted = prof.report()
        print("\n" + report_txt)
        with open(os.path.join(output_dir, 'profile.txt'), 'w') as f:
            f.write(report_txt)
        pd.DataFrame(
            [{'section': k, 'total_s': v, 'calls': prof.n[k], 'mean_ms': 1000 * v / prof.n[k],
              'pct_wall': 100 * v / max(prof.wall, 1e-9)} for k, v in rows]
        ).to_csv(os.path.join(output_dir, 'profile.csv'), index=False)
    else:
        print("Profile is empty (fewer images than --warmup). Lower --warmup or add images.")

    # save submission file
    if args.submit:
        submission_df = pd.DataFrame(tmp)
        fp = os.path.join(output_dir, str(uuid4()) + '.csv')
        submission_df.to_csv(fp, index=False)
        print(f"Saved submission file to: {fp}")