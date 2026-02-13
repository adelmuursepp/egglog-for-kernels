from __future__ import annotations
from egglog import *
# from egglog.bindings import *

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
print("before:", egraph.extract(expr, include_cost=True))

a, = vars_("a", Num)
egraph.register(
        rewrite(a * Num(2)).to(a + a),
)
egraph.run(1)

print("after:", egraph.extract(expr, include_cost=True))
print(egraph._serialize().to_json())

with open("after.svg", "w") as f:
    f.write(egraph._graphviz().pipe(format="svg", encoding="utf-8"))
