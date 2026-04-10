"""
POMO-style 8x augmentation for eval/images_grid.
For each eval instance, apply 8 symmetry transforms (4 rotations x 2 flips)
and regenerate grid-style images into eval/images_grid8/.

Transform index:
  0: original         (x,  y )
  1: rot90            (y,  1-x)
  2: rot180           (1-x, 1-y)
  3: rot270           (1-y, x )
  4: flip_x           (1-x, y )
  5: rot90 + flip_x   (y,   x )
  6: rot180 + flip_x  (x,   1-y)
  7: rot270 + flip_x  (1-y, 1-x)

Output naming: eval_{idx:03d}_aug{aug:d}_n{num_nodes}.png
"""

import json, re, math, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

EVAL_JSON   = './data_sft/cvrp/eval/test.json'
OUT_GRID8   = './data_sft/cvrp/eval/images_grid8'
NUM_WORKERS = 8


def parse_nodes(input_str):
    nodes = {}
    for m in re.finditer(r'Node (\d+), coordinates: \[(\d+), (\d+)\], demand: (\d+)', input_str):
        nid = int(m.group(1))
        nodes[nid] = {'x': int(m.group(2)), 'y': int(m.group(3)), 'demand': int(m.group(4))}
    return nodes


def apply_augment(x, y, aug_idx):
    """Apply one of 8 POMO symmetry transforms. x,y in [0,1]."""
    if   aug_idx == 0: return  x,       y
    elif aug_idx == 1: return  y,   1 - x
    elif aug_idx == 2: return  1-x, 1 - y
    elif aug_idx == 3: return  1-y,     x
    elif aug_idx == 4: return  1-x,     y
    elif aug_idx == 5: return  y,       x
    elif aug_idx == 6: return  x,   1 - y
    elif aug_idx == 7: return  1-y, 1 - x


def resolve_conflicts(nodes_grid, grid_size):
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


def draw_grid_aug(nodes, num_nodes, capacity, inst_idx, aug_idx, out_path):
    """Draw grid-style image after applying POMO augmentation aug_idx."""
    max_coord = 1000.0
    grid_size = max(20, math.ceil(math.sqrt(num_nodes) * 2.5))

    # Apply augmentation to normalized [0,1] coords, then scale to grid
    aug_coords = {}
    for nid, v in nodes.items():
        nx = v['x'] / max_coord
        ny = v['y'] / max_coord
        ax, ay = apply_augment(nx, ny, aug_idx)
        aug_coords[nid] = {'gx': ax, 'gy': ay, 'demand': v['demand']}

    # Map [0,1] -> integer grid
    def to_grid(fx, fy):
        gx = round(fx * (grid_size - 1))
        gy = round(fy * (grid_size - 1))
        return gx, gy

    raw_grid   = [(nid, to_grid(v['gx'], v['gy'])) for nid, v in aug_coords.items()]
    positions  = resolve_conflicts(raw_grid, grid_size)
    demand_map = {nid: v['demand'] for nid, v in aug_coords.items()}
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

    aug_names = ['orig', 'rot90', 'rot180', 'rot270',
                 'flip', 'rot90f', 'rot180f', 'rot270f']
    ax.set_title(
        f'Blue\u25cf=customer, \u2605=depot, size\u221ddepand\n'
        f'Instance {inst_idx + 1}  |  Nodes: {num_nodes}  |  Cap: {capacity}'
        f'  |  Aug: {aug_names[aug_idx]}  |  Grid: {grid_size}\xd7{grid_size}',
        fontsize=max(7, font_size + 1), pad=6)
    ax.tick_params(labelsize=max(5, font_size - 1))
    ax.set_xlabel('Grid Column', fontsize=max(6, font_size))
    ax.set_ylabel('Grid Row',    fontsize=max(6, font_size))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def worker(args_tuple):
    idx, rec, out_dir = args_tuple
    num_nodes = int(rec['num_nodes'])
    capacity  = float(rec['vehicle_capacity'])
    nodes     = parse_nodes(rec['input'])
    if not nodes:
        return
    for aug_idx in range(8):
        fname    = f'eval_{idx:03d}_aug{aug_idx}_n{num_nodes}.png'
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path):
            continue
        try:
            draw_grid_aug(nodes, num_nodes, capacity, idx, aug_idx, out_path)
        except Exception as e:
            print(f'  [eval {idx} aug {aug_idx}] error: {e}')


if __name__ == '__main__':
    os.makedirs(OUT_GRID8, exist_ok=True)

    with open(EVAL_JSON) as f:
        eval_data = json.load(f)

    print(f'Generating 8x augmented images for {len(eval_data)} eval samples -> {OUT_GRID8}')
    print(f'Total output: {len(eval_data) * 8} images')

    tasks = [(i, rec, OUT_GRID8) for i, rec in enumerate(eval_data)]

    with Pool(processes=NUM_WORKERS) as pool:
        for _ in tqdm(pool.imap_unordered(worker, tasks, chunksize=4),
                      total=len(tasks)):
            pass

    n_out = len([f for f in os.listdir(OUT_GRID8) if f.endswith('.png')])
    print(f'\nDone. {n_out} images in {OUT_GRID8}')
