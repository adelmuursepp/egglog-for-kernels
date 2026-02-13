"""
Serializing an e-graph to JSON.

Builds on the example from 02-extraction.py. Shows how to serialize
the e-graph structure and combine it with extraction results.
"""

from __future__ import annotations
from egglog import *
from egglog.egraph import to_runtime_expr
import json

class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...

    @classmethod
    def var(cls, name: StringLike) -> Num: ...

    @method(cost=2)
    def __add__(self, other: Num) -> Num: ...

    @method(cost=10)
    def __mul__(self, other: Num) -> Num: ...

egraph = EGraph()
expr = egraph.let("expr", Num.var("x") * Num(2) + Num(1))
sub_expr = Num.var("x") * Num(2)

a, = vars_("a", Num)
egraph.register(
    rewrite(a * Num(2)).to(a + a),
)
egraph.run(1)


def serialize_egraph(egraph, root_exprs=None):
    """Serialize the e-graph, optionally marking expressions as roots.

    root_exprs controls which e-classes are marked in root_eclasses:
      - None or []: bare graph structure, no roots marked
      - [expr]:     single root
      - [expr, x]:  multiple roots (useful for codegen of several outputs)
    """
    roots = []
    if root_exprs:
        for e in root_exprs:
            rt = to_runtime_expr(e)
            egg_expr = egraph._state.typed_expr_to_egg(rt.__egg_typed_expr__)
            roots.append(egg_expr)

    serialized = egraph._egraph.serialize(
        roots,
        max_functions=None,
        max_calls_per_function=None,
        include_temporary_functions=False,
    )
    serialized.split_classes(egraph._egraph, set())
    serialized.map_ops(egraph._state.op_mapping())
    return json.loads(serialized.to_json())


# 1) Bare graph structure, no roots marked
print("1) Bare serialization")
bare = serialize_egraph(egraph)
print(json.dumps(bare, indent=2))
print()

# 2) With a root e-class marked
# Passing root_exprs tells the serializer which e-classes matter.
# The JSON's root_eclasses field will list their IDs, and class_data
# shows metadata like the "let" binding name.
print("2) Single root")
single = serialize_egraph(egraph, root_exprs=[expr])
print(f"root_eclasses: {single['root_eclasses']}")
root_id = single["root_eclasses"][0]
print(f"class_data: {single['class_data'][root_id]}")
print()

# 3) Graph + extraction results per root
# extract() returns the lowest-cost expression for an e-class.
# extract_multiple() returns alternative equivalent expressions.
# To mark multiple roots, pass more expressions:
#   serialize_egraph(egraph, root_exprs=[expr, sub_expr])
print("3) Serialization with extraction")
roots = [expr, sub_expr]
graph_json = serialize_egraph(egraph, root_exprs=roots)

extractions = {}
for i, root in enumerate(roots):
    eclass_id = graph_json["root_eclasses"][i]
    best, cost = egraph.extract(root, include_cost=True)
    variants = egraph.extract_multiple(root, 10)
    extractions[eclass_id] = {
        "best": str(best),
        "cost": cost,
        "variants": [str(v) for v in variants],
    }

output = {"egraph": graph_json, "extractions": extractions}
print(json.dumps(output, indent=2))
