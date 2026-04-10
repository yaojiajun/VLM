"""
Generate eval/images_grid_uniform/ — uniform-size node images for the eval set.

Source: data_sft/cvrp/eval/test.json
Output: data_sft/cvrp/eval/images_grid_uniform/  eval_{j:05d}_n{num_nodes}.png
"""

import json, re, math, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm


def parse_nodes(input_str):
    nodes = {}
    for m in re.finditer(r'Node (\d+), coordinates: \[(\d+), (\d+)\], demand: (\d+)', input_str):
        nid = int(m.group(1))
        nodes[nid] = {'x': int(m.group(2)), 'y': int(m.group(3)), 'demand': int(m.group(4))}
    return nodes


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


def draw_uniform_grid(nodes, num_nodes, capacity, inst_idx, out_path):
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
            size = base_ms * 1.0
            ax.plot(col, row, 'o', color='#6baed6',
                    markersize=size,
                    markeredgecolor='#2171b5', markeredgewidth=0.5, zorder=3)
            ax.text(col, row + 0.35, f'C{nid}[{d}]',
                    ha='center', va='bottom',
                    fontsize=font_size, color='#222222', zorder=5)

    ax.set_title(
        f'Blue\u25cf=customer, \u2605=depot, uniform size\n'
        f'Instance {inst_idx + 1}  |  Nodes: {num_nodes}  |  Cap: {capacity}  |  Grid: {grid_size}\xd7{grid_size}',
        fontsize=max(7, font_size + 1), pad=6)
    ax.tick_params(labelsize=max(5, font_size - 1))
    ax.set_xlabel('Grid Column', fontsize=max(6, font_size))
    ax.set_ylabel('Grid Row',    fontsize=max(6, font_size))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def convert_single(args_tuple):
    idx, sample, output_dir = args_tuple
    num_nodes = int(sample.get('num_nodes', 0))
    out_path  = os.path.join(output_dir, f"eval_{idx:05d}_n{num_nodes}.png")
    if os.path.exists(out_path):
        return
    try:
        nodes    = parse_nodes(sample['input'])
        capacity = float(sample.get('vehicle_capacity', 0))
        draw_uniform_grid(nodes, num_nodes, capacity, idx, out_path)
    except Exception as e:
        print(f"  Error on sample {idx}: {e}")


if __name__ == '__main__':
    eval_json = './data_sft/cvrp/eval/test.json'
    output_dir = './data_sft/cvrp/eval/images_grid_uniform'
    os.makedirs(output_dir, exist_ok=True)

    with open(eval_json) as f:
        data = json.load(f)
    print(f"Generating {len(data)} uniform-grid eval images -> {output_dir}")

    tasks = [(i, sample, output_dir) for i, sample in enumerate(data)]
    with Pool(processes=8) as pool:
        for _ in tqdm(pool.imap_unordered(convert_single, tasks, chunksize=10),
                      total=len(tasks)):
            pass
    print("Done.")
