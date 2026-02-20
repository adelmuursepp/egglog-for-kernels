"""
Run the full attention rewrite pipeline:
  1. attention.py:  build egraph, apply rewrites, serialize to egraph.json
  2. extract.py:    read egraph.json, run ILP extraction with traffic costs
  3. visualize.py:  render egraph diagrams (plain + cost-annotated)

Usage:
  python run.py

Or run each step separately:
  python attention.py    # outputs egraph.json
  python extract.py      # reads egraph.json, runs ILP

Requirements:
  pip install egglog pulp graphviz
"""

import json
import os
from attention import build_and_serialize
from extract import compute_traffic_costs, ilp_extract_all_optimal, format_extraction
from visualize import visualize

OUTPUT_DIR = "output"
EGRAPH_JSON = os.path.join(OUTPUT_DIR, "egraph.json")

# Tile specs: name -> (rows, cols, dtype_bytes, loop_iters)
# Q is loaded once (loop_iters=1), K and V are streamed (loop_iters=trip count).
TILES = {
    "Q": (128, 64, 2, 1),
    "K": (64, 128, 2, 8),
    "V": (128, 64, 2, 8),
}
ACCUM_DTYPE_BYTES = 4  # fp32 accumulator output from WGMMA

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Building egraph and serializing to JSON")
build_and_serialize(EGRAPH_JSON, tiles=TILES, accum_dtype_bytes=ACCUM_DTYPE_BYTES)
print()

# Load the serialized egraph for ILP extraction and visualization
with open(EGRAPH_JSON) as f:
    graph_json = json.load(f)

print("Running ILP extraction (all optimal solutions)")
traffic_costs = compute_traffic_costs(graph_json)
total_cost, all_solutions = ilp_extract_all_optimal(graph_json, traffic_costs)
root_ec = graph_json["root_eclasses"][0]
nodes = graph_json["nodes"]

print(f"Optimal cost: {int(total_cost)} bytes")
print()

def live_eclasses(selected, root):
    """Return the subset of selected that is reachable from root."""
    visited = set()
    stack = [root]
    while stack:
        ec = stack.pop()
        if ec in visited or ec not in selected:
            continue
        visited.add(ec)
        for child_nid in nodes[selected[ec]]["children"]:
            stack.append(nodes[child_nid]["eclass"])
    return {ec: selected[ec] for ec in visited}

# Deduplicate: ILP may find solutions that differ only in dead (unreachable)
# e-classes. Filter each solution to live e-classes only, then deduplicate
# by root-reachable expression so coloring reflects only meaningful selections.
seen_exprs = {}
for selected in all_solutions:
    live = live_eclasses(selected, root_ec)
    expr = format_extraction(graph_json, live, root_ec)
    if expr not in seen_exprs:
        seen_exprs[expr] = live
unique_solutions = list(seen_exprs.values())

print(f"{len(unique_solutions)} unique optimal structure(s) found"
      f" ({len(all_solutions)} total ILP solutions before deduplication):")
for i, selected in enumerate(unique_solutions):
    expr = format_extraction(graph_json, selected, root_ec)
    print(f"  Solution {i + 1}: {expr}")
print()

print("Transaction dict (eclass -> bytes moved) for solution 1:")
for ec, nid in sorted(unique_solutions[0].items()):
    op = nodes[nid]["op"]
    c = traffic_costs.get(nid, 0)
    if c > 0:
        print(f"  {ec:30s}  {op:20s}  {c} bytes")
print()

# Build node_costs: {node_id: bytes} for every node selected in any solution.
# Includes 0-cost nodes so the cost row shows up on all selected nodes.
node_costs = {}
for sol in unique_solutions:
    for nid in sol.values():
        node_costs[nid] = traffic_costs.get(nid, 0)

print("Generating egraph visualizations")
visualize(graph_json, output_path=os.path.join(OUTPUT_DIR, "egraph"),
          node_costs=node_costs, all_selected=unique_solutions)
