"""
draw_solutions.py

Read eval_predictions.json, find feasible solutions, and draw each route
on top of the corresponding eval/images_grid/ image.

Feasible solutions are saved to:
    data_sft/cvrp/eval/solutions/   (named eval_{idx:03d}_n{num_nodes}_sol.png)
"""

import json, re, math, os, ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

PREDICTIONS_JSON = './eval_predictions.json'
OUT_DIR          = './data_sft/cvrp/eval/solutions'
MAX_COORD        = 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_nodes(input_str):
    nodes = {}
    for m in re.finditer(r'Node (\d+), coordinates: \[(\d+), (\d+)\], demand: (\d+)', input_str):
        nid = int(m.group(1))
        nodes[nid] = {'x': int(m.group(2)), 'y': int(m.group(3)), 'demand': int(m.group(4))}
    return nodes


def parse_routes(prediction_text):
    """Extract routes list from model output. Returns list of lists or None."""
    m = re.search(r'Routes:\s*\[\s*(.*)\]', prediction_text, re.DOTALL)
    if not m:
        return None
    try:
        routes = ast.literal_eval(f'[{m.group(1).strip()}]')
        if all(isinstance(r, list) for r in routes):
            return routes
    except Exception:
        pass
    return None


def parse_objective(prediction_text):
    m = re.search(r'Objective:\s*([\d.]+)', prediction_text)
    return float(m.group(1)) if m else None


def check_feasibility(routes, nodes, capacity):
    """Return True if routes are feasible (all customers visited once, capacity ok)."""
    if routes is None:
        return False
    demands = {nid: v['demand'] for nid, v in nodes.items()}
    n_customers = len(nodes) - 1  # exclude depot 0
    visited = set()
    for route in routes:
        if not route or route[0] != 0 or route[-1] != 0:
            return False
        total_demand = sum(demands.get(n, 0) for n in route if n != 0)
        if total_demand > capacity:
            return False
        visited.update(n for n in route if n != 0)
    return visited == set(range(1, n_customers + 1))


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_solution_on_grid(nodes, num_nodes, capacity, inst_idx, routes, obj_pred, obj_label, out_path):
    """
    Re-draw the grid-style instance image AND overlay the predicted routes.
    Each route gets a distinct colour. Arrows show direction of travel.
    """
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

    # Build grid positions (no conflict resolution needed for drawing—just use direct mapping)
    pos = {}
    for nid, x, y, _ in nodes_info:
        pos[nid] = to_grid(x, y)

    demand_map = {nid: d for nid, _, _, d in nodes_info}
    max_demand = max((d for d in demand_map.values() if d > 0), default=1)

    fig_in    = max(8, min(20, grid_size * 22 / 100))
    base_ms   = max(4, min(14, 180 / grid_size))
    font_size = max(3.5, min(8, 120 / grid_size))

    fig, ax = plt.subplots(figsize=(fig_in, fig_in), facecolor='white')

    # Grid lines
    for k in range(grid_size + 1):
        ax.axhline(k - 0.5, color='#cccccc', lw=0.4, zorder=0)
        ax.axvline(k - 0.5, color='#cccccc', lw=0.4, zorder=0)

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')
    ax.set_facecolor('#f8f8f8')

    # ── Draw routes (behind nodes) ────────────────────────────────────────────
    colors = cm.tab20.colors  # up to 20 distinct colours
    for r_idx, route in enumerate(routes):
        col = colors[r_idx % len(colors)]
        for step in range(len(route) - 1):
            n_from = route[step]
            n_to   = route[step + 1]
            x0, y0 = pos[n_from]
            x1, y1 = pos[n_to]
            ax.annotate(
                '', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle='->', color=col, lw=1.2,
                    connectionstyle='arc3,rad=0.05'
                ),
                zorder=2,
            )

    # ── Draw nodes (on top of routes) ────────────────────────────────────────
    for nid, (col, row) in pos.items():
        d = demand_map[nid]
        if nid == 0:
            ax.plot(col, row, '*', color='#2ca02c',
                    markersize=base_ms * 2.5,
                    markeredgecolor='darkgreen', markeredgewidth=0.6, zorder=5)
        else:
            size = base_ms * (0.5 + 1.2 * d / max_demand)
            ax.plot(col, row, 'o', color='#6baed6',
                    markersize=size,
                    markeredgecolor='#2171b5', markeredgewidth=0.5, zorder=4)
            ax.text(col, row + 0.35, f'C{nid}[{d}]',
                    ha='center', va='bottom',
                    fontsize=font_size, color='#222222', zorder=6)

    # ── Title ─────────────────────────────────────────────────────────────────
    obj_str = f'{obj_pred:.2f}' if obj_pred is not None else 'N/A'
    ref_str = f'{obj_label:.2f}' if obj_label is not None else 'N/A'
    gap_str = ''
    if obj_pred is not None and obj_label is not None and obj_label > 0:
        gap = (obj_pred - obj_label) / obj_label * 100
        gap_str = f'  gap={gap:+.1f}%'

    ax.set_title(
        f'Blue\u25cf=customer, \u2605=depot  |  {len(routes)} routes  |  '
        f'Cap: {capacity}\n'
        f'Instance {inst_idx + 1}  |  Nodes: {num_nodes}  |  '
        f'Pred obj: {obj_str}  Ref: {ref_str}{gap_str}',
        fontsize=max(7, font_size + 1), pad=6)
    ax.tick_params(labelsize=max(5, font_size - 1))
    ax.set_xlabel('Grid Column', fontsize=max(6, font_size))
    ax.set_ylabel('Grid Row',    fontsize=max(6, font_size))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(PREDICTIONS_JSON) as f:
        records = json.load(f)

    # Load eval JSON to get node input strings
    eval_json_path = './data_sft/cvrp/eval/test.json'
    with open(eval_json_path) as f:
        eval_data = json.load(f)

    feasible_count = 0
    infeasible_count = 0

    for rec in records:
        idx       = rec['idx']
        num_nodes = int(rec['num_nodes'])
        capacity  = float(rec['vehicle_capacity'])
        prediction = rec['prediction']
        label      = rec['label']

        # Parse nodes from eval JSON (same index)
        nodes = parse_nodes(eval_data[idx]['input'])

        # Parse routes
        routes = parse_routes(prediction)

        # Feasibility check
        if not check_feasibility(routes, nodes, capacity):
            infeasible_count += 1
            continue

        feasible_count += 1

        # Parse objectives
        obj_pred  = parse_objective(prediction)
        obj_label = parse_objective(label)

        # Output filename
        fname    = f'eval_{idx:03d}_n{num_nodes}_sol.png'
        out_path = os.path.join(OUT_DIR, fname)

        draw_solution_on_grid(
            nodes, num_nodes, capacity, idx,
            routes, obj_pred, obj_label, out_path
        )
        print(f'  [{idx:3d}] saved -> {fname}  '
              f'(routes={len(routes)}, pred={obj_pred}, ref={obj_label})')

    print(f'\nDone.  feasible={feasible_count}, infeasible={infeasible_count}')
    print(f'Saved to {OUT_DIR}/')
