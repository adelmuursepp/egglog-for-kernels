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

# Deduplicate: ILP may find solutions that differ only in dead (unreachable)
# e-classes. Keep only structurally distinct root-reachable expressions.
seen_exprs = {}
for selected in all_solutions:
    expr = format_extraction(graph_json, selected, root_ec)
    if expr not in seen_exprs:
        seen_exprs[expr] = selected
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

# Build eclass_costs from solution 1 for annotation (costs are same across solutions)
eclass_costs = {}
for ec, nid in unique_solutions[0].items():
    c = traffic_costs.get(nid, 0)
    if c > 0:
        eclass_costs[ec] = c

print("Generating egraph visualizations")
visualize(graph_json, output_path=os.path.join(OUTPUT_DIR, "egraph"),
          eclass_costs=eclass_costs, all_selected=unique_solutions)
