import json, re, ast, sys
import numpy as np
from utils import load_pkl_dataset, compute_euclidean_distance_matrix, calculate_total_distance, extract_predicted_solution

pred_file = sys.argv[1] if len(sys.argv) > 1 else './eval_predictions.json'
pkl_file  = sys.argv[2] if len(sys.argv) > 2 else './data_sft/cvrp/instances.pkl'

with open(pred_file) as f:
    records = json.load(f)
instances = load_pkl_dataset(pkl_file)

print(f"{'idx':>4}  {'pred_cost':>10}  {'opt_cost':>10}  {'gap%':>8}  {'feasible':>8}")
print("-" * 52)

gaps = []
for r in records:
    i = r['idx']
    pred_text  = extract_predicted_solution("### Response:\n" + r['prediction'])
    label_text = extract_predicted_solution(r['label'])

    locs     = instances[i][0]
    demands  = instances[i][1]
    capacity = instances[i][2]
    dist_mat = compute_euclidean_distance_matrix(locs)

    # parse label cost
    lm = re.search(r"Objective:\s*([\d.]+)", label_text)
    opt_cost = float(lm.group(1)) if lm else None

    # parse predicted routes
    pm = re.search(r"Routes:\s*\[\s*(.*)\]", pred_text, re.DOTALL)
    if not pm or opt_cost is None:
        print(f"{i:>4}  {'N/A':>10}  {opt_cost if opt_cost else 'N/A':>10}  {'infeas':>8}  {'No':>8}")
        continue
    try:
        routes = ast.literal_eval(f'[{pm.group(1).strip()}]')
    except:
        print(f"{i:>4}  {'parse_err':>10}  {opt_cost:>10.4f}  {'infeas':>8}  {'No':>8}")
        continue

    # feasibility check
    feasible = True
    for route in routes:
        if not route or route[0] != 0 or route[-1] != 0:
            feasible = False; break
        if sum(demands[n] for n in route if n != 0) > capacity:
            feasible = False; break
    visited = set()
    for route in routes:
        visited.update(route[1:-1])
    if visited != set(range(1, len(demands))):
        feasible = False

    if not feasible:
        print(f"{i:>4}  {'infeas':>10}  {opt_cost:>10.4f}  {'infeas':>8}  {'No':>8}")
        continue

    pred_cost = sum(calculate_total_distance(r, dist_mat) for r in routes)
    gap = (pred_cost - opt_cost) / opt_cost * 100
    gaps.append(gap)
    print(f"{i:>4}  {pred_cost:>10.4f}  {opt_cost:>10.4f}  {gap:>7.2f}%  {'Yes':>8}")

print("-" * 52)
if gaps:
    print(f"Feasible: {len(gaps)}/{len(records)}  |  Mean gap: {np.mean(gaps):.2f}%  |  Std: {np.std(gaps):.2f}%")
