import os
import shutil
import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from shapely import wkt as shapely_wkt
from collections import defaultdict
from shapely.geometry import MultiLineString
from shapely.ops import unary_union


def lines_to_graph(lines, size=256, flip_y=True, border=1.0, node_first=True):
    """
    lines -> SpaceNet-style adjacency dict {(r, c): [(r, c), ...]}

    flip_y=True : key = (size - y, x)   (SpaceNet convention, matches your loader)
    flip_y=False: key = (y, x)          (plain image row, col)
    """
    if node_first:  # insert vertices where lines cross or touch
        noded = unary_union(MultiLineString([list(map(tuple, l)) for l in lines]))
        lines = [list(g.coords) for g in getattr(noded, "geoms", [noded])]

    def to_key(p):
        x, y = int(round(p[0])), int(round(p[1]))
        return (size - y, x) if flip_y else (y, x)

    # fine-grained adjacency between consecutive vertices
    nbrs = defaultdict(set)
    for line in lines:
        pts = [to_key(p) for p in line]
        for a, b in zip(pts[:-1], pts[1:]):
            if a != b:
                nbrs[a].add(b)
                nbrs[b].add(a)

    def on_border(k):
        r, c = k
        x, y = (c, size - r) if flip_y else (c, r)
        return x <= border or y <= border or x >= size - border or y >= size - border

    # nodes = degree != 2, or on the image border
    nodes = {k for k, n in nbrs.items() if len(n) != 2 or on_border(k)}

    # closed loops with no node: promote one vertex per component
    seen = set()
    for start in list(nbrs):
        if start in seen:
            continue
        comp, stack = set(), [start]
        while stack:
            k = stack.pop()
            if k not in comp:
                comp.add(k)
                stack.extend(nbrs[k])
        seen |= comp
        if not (comp & nodes):
            nodes.add(min(comp))

    # walk from each node along degree-2 chains to the next node
    graph = defaultdict(list)
    for n in nodes:
        for first in nbrs[n]:
            prev, cur = n, first
            while cur not in nodes:
                nxt = [m for m in nbrs[cur] if m != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            if cur != n:
                if cur not in graph[n]:
                    graph[n].append(cur)
                if n not in graph[cur]:
                    graph[cur].append(n)
    return {k: sorted(v) for k, v in graph.items()}

if __name__ == "__main__":
    IMAGE_DIR = Path("/kaggle/input/datasets/labanochwo/hotosm-train/zindi/images")
    df = pd.read_csv("Train.csv")
    df.geometry = df.geometry.apply(shapely_wkt.loads)
    df['coords'] = df.geometry.apply(lambda geom: [list(part.coords) for part in geom.geoms])
    df['image_path'] = df.ID.apply(lambda ID: IMAGE_DIR / f"{ID}.png")
    
    os.makedirs("RGB_1.0_meter", exist_ok=True)
    for row in tqdm(df.itertuples(), total=len(df), desc="Processing rows"):
        tile_index = row.ID
        lines = row.coords
        shutil.copy(row.image_path, f"./RGB_1.0_meter/{tile_index}__rgb.png")
        graph = lines_to_graph(lines, size=256, flip_y=True, border=1.0, node_first=True)
        with open(f"./RGB_1.0_meter/{tile_index}__gt_graph.p", 'wb') as f:
            pickle.dump(graph, f)
        
    