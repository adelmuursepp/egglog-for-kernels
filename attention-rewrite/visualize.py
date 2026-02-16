"""
Visualize a serialized egraph as SVG and PDF.

Filters to show only Tile operation nodes (LDS, LDR, WGMMA, etc),
omitting analysis property nodes (.rows, .cols, .dtype_bytes, etc).

Generates two diagrams:
  egraph.svg/pdf       - clean dataflow diagram
  egraph-costs.svg/pdf - annotated with per-e-class transaction costs
"""

import graphviz


def _build_dot(graph_json, eclass_costs=None, selected=None):
    """Build a filtered graphviz Digraph showing only Tile operation nodes.

    If eclass_costs is provided (dict of eclass -> bytes), annotates each
    e-class with its transaction cost.
    If selected is provided (dict of eclass -> node_id from ILP), highlights
    the chosen nodes and dims the alternatives.
    """
    nodes = graph_json["nodes"]
    root_eclasses = set(graph_json["root_eclasses"])

    tile_ops = {"Tile.input", "Tile.LDS", "Tile.LDR", "Tile.WGMMA",
                "Tile.Elementwise", "Tile.STS", "Tile.STG"}
    tile_nodes = {nid: n for nid, n in nodes.items() if n["op"] in tile_ops}

    eclass_to_nodes = {}
    for nid, n in tile_nodes.items():
        eclass_to_nodes.setdefault(n["eclass"], []).append(nid)

    selected_nids = set(selected.values()) if selected else set()

    dot = graphviz.Digraph(format="svg")
    dot.attr(rankdir="TB", fontname="Helvetica", fontsize="11",
             nodesep="0.4", ranksep="0.6")
    dot.attr("node", shape="none", fontname="Helvetica", fontsize="10")

    # Add legend when showing costs
    if eclass_costs:
        total_bytes = sum(eclass_costs.values())
        total_kb = total_bytes / 1024
        with dot.subgraph(name="cluster_legend") as legend:
            legend.attr(style="rounded", color="#999999",
                        label="", fontsize="9")
            legend_text = (
                f'Costs shown are for the ILP-selected optimal DAG path.\\n'
                f'Total traffic: {total_kb:,.0f} KB ({total_bytes:,} bytes)\\n'
            )
            if selected:
                legend_text += (
                    f'Green = selected node  |  Gray = alternative (not chosen)'
                )
            legend.node("legend", shape="note", style="filled",
                        fillcolor="#ffffee", fontsize="9",
                        fontname="Helvetica", label=legend_text)

    for ec, nids in eclass_to_nodes.items():
        is_root = ec in root_eclasses
        ec_selected = selected and ec in selected
        if is_root:
            color = "#ffcccc"
        elif ec_selected:
            color = "#e8f5e9"
        else:
            color = "#f5f5f5" if selected else "#e8e8e8"

        ec_label = ec.split("-")[-1]
        if eclass_costs and ec in eclass_costs and eclass_costs[ec] > 0:
            cost_kb = eclass_costs[ec] / 1024
            ec_label += f"  ({cost_kb:,.0f} KB)"

        with dot.subgraph(name=f"cluster_{ec}") as sub:
            sub.attr(style="dashed,rounded,filled", fillcolor=color,
                     label=ec_label, fontsize="9", fontcolor="#666666")
            for nid in nids:
                op = nodes[nid]["op"]
                children = nodes[nid]["children"]
                if op == "Tile.input":
                    child_ops = [nodes[c]["op"] for c in children]
                    label = f'{op}({", ".join(child_ops)})'
                else:
                    label = op

                if selected and nid in selected_nids:
                    bg = "#c8e6c9"
                    border_color = "#2e7d32"
                    font_color = "black"
                elif selected:
                    bg = "#e0e0e0"
                    border_color = "#bdbdbd"
                    font_color = "#888888"
                else:
                    bg = "white"
                    border_color = "black"
                    font_color = "black"

                sub.node(nid, label=(
                    f'<<TABLE BGCOLOR="{bg}" BORDER="1" CELLBORDER="0" '
                    f'CELLPADDING="4" COLOR="{border_color}">'
                    f'<TR><TD><FONT COLOR="{font_color}">{label}</FONT></TD></TR>'
                    f'</TABLE>>'
                ))

    for nid, n in tile_nodes.items():
        for child_nid in n["children"]:
            child_ec = nodes[child_nid]["eclass"]
            if child_ec in eclass_to_nodes:
                target = eclass_to_nodes[child_ec][0]
                dot.edge(nid, target, lhead=f"cluster_{child_ec}")

    return dot


def visualize(graph_json, output_path="egraph", eclass_costs=None, selected=None):
    """Save egraph visualizations as SVG.

    graph_json: serialized egraph dict (from serialize_egraph or loaded from JSON)
    output_path: base path for output files (without extension)
    eclass_costs: optional dict of {eclass_id: bytes_moved} from ILP extraction
    selected: optional dict of {eclass_id: node_id} from ILP extraction
    """
    dot = _build_dot(graph_json)
    dot.render(output_path, format="svg", cleanup=True)
    print(f"Wrote {output_path}.svg")

    if eclass_costs:
        dot_costs = _build_dot(graph_json, eclass_costs=eclass_costs,
                               selected=selected)
        cost_path = f"{output_path}-costs"
        dot_costs.render(cost_path, format="svg", cleanup=True)
        print(f"Wrote {cost_path}.svg")
