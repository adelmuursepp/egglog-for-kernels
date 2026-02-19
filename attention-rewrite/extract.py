"""
ILP-based optimal DAG extraction from a serialized egraph JSON.

Reads an egraph JSON file, computes traffic costs from the analysis
nodes embedded in the graph, and solves for the minimum-cost DAG
using integer linear programming (PuLP/CBC).

Input:  egraph.json (from attention.py)
Output: prints the optimal extraction and its cost
"""

import json
import sys
import pulp


def ilp_extract(graph_json, node_costs=None):
    """Optimal DAG extraction via ILP.

    Binary variable x_n for each e-node: 1 if selected.
    Binary variable y_c for each e-class: 1 if reachable.

    Minimize: sum of cost(n) * x_n
    Subject to:
      1. Root e-classes must be active: y_root = 1
      2. Exactly one e-node per active e-class: sum(x_n) = y_c
      3. If e-node selected, children e-classes active: y_child >= x_n
    """
    nodes = graph_json["nodes"]
    root_eclasses = graph_json["root_eclasses"]

    eclass_to_nodes = {}
    for nid, ndata in nodes.items():
        ec = ndata["eclass"]
        eclass_to_nodes.setdefault(ec, []).append(nid)

    def children_eclasses(nid):
        return [nodes[child]["eclass"] for child in nodes[nid]["children"]]

    def cost(nid):
        if node_costs and nid in node_costs:
            return node_costs[nid]
        return nodes[nid]["cost"]

    prob = pulp.LpProblem("egraph_extraction", pulp.LpMinimize)

    x = {nid: pulp.LpVariable(f"x_{nid}", cat="Binary") for nid in nodes}
    y = {ec: pulp.LpVariable(f"y_{ec}", cat="Binary") for ec in eclass_to_nodes}

    prob += pulp.lpSum(cost(nid) * x[nid] for nid in nodes)

    for rec in root_eclasses:
        if rec in y:
            prob += y[rec] == 1

    for ec, nids in eclass_to_nodes.items():
        prob += pulp.lpSum(x[nid] for nid in nids) == y[ec]

    for nid in nodes:
        for child_ec in children_eclasses(nid):
            if child_ec in y:
                prob += y[child_ec] >= x[nid]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    total_cost = pulp.value(prob.objective)
    selected = {}
    for ec, nids in eclass_to_nodes.items():
        for nid in nids:
            if pulp.value(x[nid]) == 1:
                selected[ec] = nid
                break

    return total_cost, selected


def format_extraction(graph_json, selected, root_eclass):
    """Recursively format the selected extraction as a readable string."""
    nodes = graph_json["nodes"]
    if root_eclass not in selected:
        return f"<{root_eclass}>"
    nid = selected[root_eclass]
    ndata = nodes[nid]
    op = ndata["op"]
    children = ndata["children"]
    if not children:
        return op
    child_strs = []
    for child_nid in children:
        child_ec = nodes[child_nid]["eclass"]
        child_strs.append(format_extraction(graph_json, selected, child_ec))
    return f"{op}({', '.join(child_strs)})"


def compute_traffic_costs(graph_json):
    """Compute per-node transaction cost from the serialized egraph JSON.

    Each e-node gets a transaction size (bytes of data moved):
      LDS, LDR, STS, STG: tile_bytes * loop_iters
      WGMMA: implicit smem->reg load for any shared operands * loop_iters
      Elementwise, input: 0

    After ILP extraction selects one e-node per active e-class, the
    transaction dict is {eclass: bytes}. Each e-class counted once,
    so no double-counting. Total cost = sum of the dict values.
    """
    nodes = graph_json["nodes"]

    # Collect primitive integer values per eclass
    eclass_values = {}
    for nid, ndata in nodes.items():
        try:
            eclass_values[ndata["eclass"]] = int(ndata["op"])
        except ValueError:
            pass

    # Collect analysis properties per tile eclass from analysis nodes
    tile_rows = {}
    tile_cols = {}
    tile_dtype = {}
    tile_loop = {}
    tile_mem = {}

    for nid, ndata in nodes.items():
        op = ndata["op"]
        children = ndata["children"]
        ec = ndata["eclass"]

        if len(children) != 1:
            continue
        tile_ec = nodes[children[0]]["eclass"]

        if op == "\u00b7.rows" and ec in eclass_values:
            tile_rows[tile_ec] = eclass_values[ec]
        elif op == "\u00b7.cols" and ec in eclass_values:
            tile_cols[tile_ec] = eclass_values[ec]
        elif op == "\u00b7.dtype_bytes" and ec in eclass_values:
            tile_dtype[tile_ec] = eclass_values[ec]
        elif op == "\u00b7.loop_iters" and ec in eclass_values:
            tile_loop[tile_ec] = eclass_values[ec]
        elif op == "\u00b7.mem_region":
            tile_mem[tile_ec] = ec

    # Map MemRegion eclasses to names
    mem_names = {}
    for nid, ndata in nodes.items():
        if ndata["op"] in ("MemRegion.SHARED", "MemRegion.GLOBAL", "MemRegion.REGISTERS"):
            mem_names[ndata["eclass"]] = ndata["op"]

    def tile_bytes(tile_ec):
        r = tile_rows.get(tile_ec, 0)
        c = tile_cols.get(tile_ec, 0)
        d = tile_dtype.get(tile_ec, 0)
        return r * c * d

    def loop_iters(tile_ec):
        return tile_loop.get(tile_ec, 1)

    def mem_region(tile_ec):
        mr_ec = tile_mem.get(tile_ec)
        return mem_names.get(mr_ec, "") if mr_ec else ""

    costs = {}
    for nid, ndata in nodes.items():
        op = ndata["op"]
        ec = ndata["eclass"]
        children = ndata["children"]

        if op in ("Tile.LDS", "Tile.LDR", "Tile.STS", "Tile.STG"):
            costs[nid] = tile_bytes(ec) * loop_iters(ec)
        elif op == "Tile.WGMMA" and len(children) == 2:
            # Implicit smem->reg load for shared operands
            a_ec = nodes[children[0]]["eclass"]
            b_ec = nodes[children[1]]["eclass"]
            li = loop_iters(ec)
            c = 0
            if mem_region(a_ec) == "MemRegion.SHARED":
                c += tile_bytes(a_ec)
            if mem_region(b_ec) == "MemRegion.SHARED":
                c += tile_bytes(b_ec)
            costs[nid] = c * li
        else:
            costs[nid] = 0

    return costs


def _ilp_solve(graph_json, node_costs, target_cost=None, excluded=None):
    """Build and solve the egraph ILP from scratch.

    target_cost: if given, adds an equality constraint fixing the objective
                 to that value (used for enumerating all optimal solutions).
    excluded:    list of previously found selected dicts; a no-good cut is
                 added for each so the solver must find a different solution.

    Returns (total_cost, selected_dict) or (None, None) if infeasible.
    """
    nodes = graph_json["nodes"]
    root_eclasses = graph_json["root_eclasses"]

    eclass_to_nodes = {}
    for nid, ndata in nodes.items():
        ec = ndata["eclass"]
        eclass_to_nodes.setdefault(ec, []).append(nid)

    def children_eclasses(nid):
        return [nodes[child]["eclass"] for child in nodes[nid]["children"]]

    def cost(nid):
        if node_costs and nid in node_costs:
            return node_costs[nid]
        return nodes[nid]["cost"]

    prob = pulp.LpProblem("egraph_extraction", pulp.LpMinimize)
    x = {nid: pulp.LpVariable(f"x_{nid}", cat="Binary") for nid in nodes}
    y = {ec: pulp.LpVariable(f"y_{ec}", cat="Binary") for ec in eclass_to_nodes}

    obj = pulp.lpSum(cost(nid) * x[nid] for nid in nodes)
    prob += obj

    if target_cost is not None:
        prob += obj == target_cost

    for rec in root_eclasses:
        if rec in y:
            prob += y[rec] == 1

    for ec, nids in eclass_to_nodes.items():
        prob += pulp.lpSum(x[nid] for nid in nids) == y[ec]

    for nid in nodes:
        for child_ec in children_eclasses(nid):
            if child_ec in y:
                prob += y[child_ec] >= x[nid]

    # No-good cuts: each prior solution must differ in at least one selected node
    for prev in (excluded or []):
        prev_nids = list(prev.values())
        prob += pulp.lpSum(x[nid] for nid in prev_nids) <= len(prev_nids) - 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return None, None

    total_cost = pulp.value(prob.objective)
    selected = {}
    for ec, nids in eclass_to_nodes.items():
        for nid in nids:
            if pulp.value(x[nid]) is not None and pulp.value(x[nid]) > 0.5:
                selected[ec] = nid
                break

    return total_cost, selected


def ilp_extract_all_optimal(graph_json, node_costs=None, max_solutions=10):
    """Find all optimal DAG extractions via ILP using no-good cuts.

    After the first solution is found at cost C*, additional solves fix
    the objective to C* and add a no-good cut for each prior solution,
    forcing the solver to find a structurally different extraction with
    the same total cost.

    Returns (optimal_cost, [selected_dict_1, selected_dict_2, ...]).
    The list has length 1 if the optimal solution is unique.
    """
    optimal_cost, first_selected = ilp_extract(graph_json, node_costs)
    solutions = [first_selected]

    for _ in range(max_solutions - 1):
        _, next_selected = _ilp_solve(
            graph_json, node_costs,
            target_cost=int(optimal_cost),
            excluded=solutions,
        )
        if next_selected is None:
            break
        solutions.append(next_selected)

    return optimal_cost, solutions


def main(json_path):
    with open(json_path) as f:
        graph_json = json.load(f)

    traffic_costs = compute_traffic_costs(graph_json)

    print("ILP extraction (traffic cost, optimal DAG):")
    total_cost, selected = ilp_extract(graph_json, traffic_costs)
    root_ec = graph_json["root_eclasses"][0]
    expr = format_extraction(graph_json, selected, root_ec)
    print(f"  {expr}")
    print(f"  cost: {int(total_cost)} bytes")

    # Print the transaction dict: {eclass: bytes moved}
    nodes = graph_json["nodes"]
    print()
    print("Transaction dict (eclass -> bytes moved):")
    for ec, nid in sorted(selected.items()):
        op = nodes[nid]["op"]
        c = traffic_costs.get(nid, 0)
        if c > 0:
            print(f"  {ec:30s}  {op:20s}  {c} bytes")

# For running as standalone file
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "output/egraph.json"
    main(path)