"""
Convert CVRP JSON data to continuous-coordinate images (images/ style).
- Same JSON source as images_grid/ (train_cvrp-001.json)
- Same title metadata: Instance | Nodes | Cap
- Continuous [0,1] coordinates, NO background grid lines
- Output naming: cvrp_NNNNN.png (0-based)
"""
import json
import re
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool


def parse_nodes(input_str):
    nodes = {}
    for m in re.finditer(r'Node (\d+), coordinates: \[(\d+), (\d+)\], demand: (\d+)', input_str):
        nid = int(m.group(1))
        nodes[nid] = {'x': int(m.group(2)), 'y': int(m.group(3)), 'demand': int(m.group(4))}
    return nodes


def draw_images_style(nodes, num_nodes, capacity, inst_idx, out_path):
    """Continuous [0,1] coords, no grid lines, title with metadata — matches eval/images/ style."""
    try:
        from adjustText import adjust_text
        HAS_ADJUST = True
    except ImportError:
        HAS_ADJUST = False

    max_coord  = 1000.0
    demands    = [nodes[i]['demand'] for i in nodes if i != 0]
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
    ax.grid(False)
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def convert_single(args_tuple):
    i, sample, output_dir = args_tuple
    out_path = os.path.join(output_dir, f"cvrp_{i:05d}.png")
    if os.path.exists(out_path):
        return
    try:
        nodes     = parse_nodes(sample['input'])
        num_nodes = int(sample.get('num_nodes', len(nodes) - 1))
        capacity  = float(sample.get('vehicle_capacity', 0))
        draw_images_style(nodes, num_nodes, capacity, i, out_path)
    except Exception as e:
        print(f"  Error on sample {i}: {e}")


def convert_json_to_images(json_path, output_dir, num_samples, num_workers):
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, 'r') as f:
        data = json.load(f)
    data = data[:num_samples]
    print(f"Converting {len(data)} samples with {num_workers} workers...")
    tasks = [(i, sample, output_dir) for i, sample in enumerate(data)]
    with Pool(processes=num_workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(convert_single, tasks, chunksize=10)):
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(tasks)} done...")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str,
                        default='./data_sft/cvrp/train_cvrp-001.json')
    parser.add_argument('--output_dir', type=str,
                        default='./data_sft/cvrp/images')
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--num_workers', type=int, default=12)
    args = parser.parse_args()
    convert_json_to_images(args.json_path, args.output_dir, args.num_samples, args.num_workers)
