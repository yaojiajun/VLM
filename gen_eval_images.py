"""
Generate eval image datasets for the 300 eval/test.json samples.

Two output directories:
  data_sft/cvrp/eval/images/       — continuous [0,1] style (matches data_sft/cvrp/images/)
  data_sft/cvrp/eval/images_grid/  — discrete grid style    (matches data_sft/cvrp/images_grid/)

Image naming: eval_{idx:03d}_n{num_nodes}.png  (0-based, matching eval/test.json order)
"""

import json, re, math, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

EVAL_JSON   = './data_sft/cvrp/eval/test.json'
OUT_IMAGES  = './data_sft/cvrp/eval/images'
OUT_GRID    = './data_sft/cvrp/eval/images_grid'
NUM_WORKERS = 8


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_nodes(input_str):
    """Return dict nid -> {x, y, demand}."""
    nodes = {}
    for m in re.finditer(r'Node (\d+), coordinates: \[(\d+), (\d+)\], demand: (\d+)', input_str):
        nid = int(m.group(1))
        nodes[nid] = {'x': int(m.group(2)), 'y': int(m.group(3)), 'demand': int(m.group(4))}
    return nodes


def resolve_conflicts(nodes_grid, grid_size):
    """Snap nodes to grid; nudge duplicates to nearest free neighbour."""
    placed = {}
    positions = {}
    for nid, (gx, gy) in nodes_grid:
        cell = (gx, gy)
        if cell not in placed:
            placed[cell] = nid
            positions[nid] = cell
        else:
            found = False
            for radius in range(1, grid_size):
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        if abs(dr) != radius and abs(dc) != radius:
                            continue
                        nc, nr = gx + dc, gy + dr
                        if 0 <= nc < grid_size and 0 <= nr < grid_size:
                            cand = (nc, nr)
                            if cand not in placed:
                                placed[cand] = nid
                                positions[nid] = cand
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
    return positions


# ─────────────────────────────────────────────────────────────────────────────
# Style 1 — continuous [0,1]  (matches data_sft/cvrp/images/)
# ─────────────────────────────────────────────────────────────────────────────

def draw_images_style(nodes, num_nodes, capacity, inst_idx, out_path):
    """Blue circles on [0,1] space, adjustText labels, green star depot."""
    try:
        from adjustText import adjust_text
        HAS_ADJUST = True
    except ImportError:
        HAS_ADJUST = False

    max_coord = 1000.0
    demands   = [nodes[i]['demand'] for i in nodes if i != 0]
    max_demand = max(demands) if demands else 1
    base_size  = 120

    fig, ax = plt.subplots(figsize=(7, 7))
    texts = []

    for nid, node in nodes.items():
        x = node['x'] / max_coord
        y = node['y'] / max_coord
        d = node['demand']
        if nid == 0:
            ax.scatter(x, y, marker='*', s=400, color='green', zorder=5)
        else:
            size = base_size * (d / max_demand)
            ax.scatter(x, y, s=size, color='steelblue', alpha=0.75, zorder=4)
            t = ax.text(x, y, f'C{nid}[{d}]', fontsize=6, ha='center', color='#222222')
            texts.append(t)

    if HAS_ADJUST and texts:
        adjust_text(
            texts, ax=ax,
            expand=(1.5, 1.8), force_text=(0.4, 0.6),
            arrowprops=dict(arrowstyle='-', color='steelblue', lw=1.5),
        )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f'Blue\u25cf=customer, \u2605=depot, size\u221ddepand\n'
        f'Instance {inst_idx + 1}  |  Nodes: {num_nodes}  |  Cap: {capacity}',
        fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Style 2 — discrete grid  (matches data_sft/cvrp/images_grid/)
# ─────────────────────────────────────────────────────────────────────────────

def draw_grid_style(nodes, num_nodes, capacity, inst_idx, out_path):
    """Blue circles on integer grid, green star depot, title with metadata."""
    nodes_info = [(nid, v['x'], v['y'], v['demand']) for nid, v in nodes.items()]
    grid_size  = max(20, math.ceil(math.sqrt(num_nodes) * 2.5))

    xs = [x for _, x, _, _ in nodes_info]
    ys = [y for _, _, y, _ in nodes_info]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def to_grid(x, y):
        gx = round((x - min_x) / max(max_x - min_x, 1) * (grid_size - 1))
        gy = round((y - min_y) / max(max_y - min_y, 1) * (grid_size - 1))
        return gx, gy

    raw_grid   = [(nid, to_grid(x, y)) for nid, x, y, _ in nodes_info]
    positions  = resolve_conflicts(raw_grid, grid_size)
    demand_map = {nid: d for nid, _, _, d in nodes_info}
    max_demand = max((d for d in demand_map.values() if d > 0), default=1)

    fig_in    = max(6, min(18, grid_size * 22 / 100))
    base_ms   = max(4, min(14, 180 / grid_size))
    font_size = max(3.5, min(8, 120 / grid_size))

    fig, ax = plt.subplots(figsize=(fig_in, fig_in), facecolor='white')

    for k in range(grid_size + 1):
        ax.axhline(k - 0.5, color='#cccccc', lw=0.4, zorder=0)
        ax.axvline(k - 0.5, color='#cccccc', lw=0.4, zorder=0)

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')
    ax.set_facecolor('#f8f8f8')

    for nid, (col, row) in positions.items():
        d = demand_map[nid]
        if nid == 0:
            ax.plot(col, row, '*', color='#2ca02c',
                    markersize=base_ms * 2.5,
                    markeredgecolor='darkgreen', markeredgewidth=0.6, zorder=4)
        else:
            size = base_ms * (0.5 + 1.2 * d / max_demand)
            ax.plot(col, row, 'o', color='#6baed6',
                    markersize=size,
                    markeredgecolor='#2171b5', markeredgewidth=0.5, zorder=3)
            ax.text(col, row + 0.35, f'C{nid}[{d}]',
                    ha='center', va='bottom',
                    fontsize=font_size, color='#222222', zorder=5)

    ax.set_title(
        f'Blue\u25cf=customer, \u2605=depot, size\u221ddepand\n'
        f'Instance {inst_idx + 1}  |  Nodes: {num_nodes}  |  Cap: {capacity}  |  Grid: {grid_size}\xd7{grid_size}',
        fontsize=max(7, font_size + 1), pad=6)
    ax.tick_params(labelsize=max(5, font_size - 1))
    ax.set_xlabel('Grid Column', fontsize=max(6, font_size))
    ax.set_ylabel('Grid Row',    fontsize=max(6, font_size))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Worker (called by multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def worker(args_tuple):
    idx, rec, out_images, out_grid = args_tuple
    num_nodes = int(rec['num_nodes'])
    capacity  = float(rec['vehicle_capacity'])
    fname     = f'eval_{idx:03d}_n{num_nodes}.png'

    nodes = parse_nodes(rec['input'])
    if not nodes:
        return

    p_img  = os.path.join(out_images, fname)
    p_grid = os.path.join(out_grid,   fname)

    if not os.path.exists(p_img):
        try:
            draw_images_style(nodes, num_nodes, capacity, idx, p_img)
        except Exception as e:
            print(f'  [images] eval[{idx}] error: {e}')

    if not os.path.exists(p_grid):
        try:
            draw_grid_style(nodes, num_nodes, capacity, idx, p_grid)
        except Exception as e:
            print(f'  [grid]   eval[{idx}] error: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUT_IMAGES, exist_ok=True)
    os.makedirs(OUT_GRID,   exist_ok=True)

    with open(EVAL_JSON) as f:
        eval_data = json.load(f)

    print(f'Generating images for {len(eval_data)} eval samples ...')
    print(f'  images style -> {OUT_IMAGES}')
    print(f'  grid  style  -> {OUT_GRID}')

    tasks = [(i, rec, OUT_IMAGES, OUT_GRID) for i, rec in enumerate(eval_data)]

    with Pool(processes=NUM_WORKERS) as pool:
        for _ in tqdm(pool.imap_unordered(worker, tasks, chunksize=4),
                      total=len(tasks)):
            pass

    # Verify
    n_img  = len([f for f in os.listdir(OUT_IMAGES) if f.endswith('.png')])
    n_grid = len([f for f in os.listdir(OUT_GRID)   if f.endswith('.png')])
    print(f'\nDone.  images/: {n_img} files,  images_grid/: {n_grid} files')
