from __future__ import annotations
from egglog import *

egraph = EGraph()

class Num(Expr):
    def __init__(self, value: i64Like) -> None: pass

    @classmethod
    def var(cls, name: StringLike) -> Num: pass


    def __add__(self, other: Num) -> Num: pass
    def __mul__(self, other: Num) -> Num: pass
    
a = egraph.let("a", Num(2) + Num(3))

print("before:", egraph.extract(a))



p, q = vars_("p q", i64)
x, y = vars_("x y", Num)

egraph.check_fail(eq(Num(1) + Num(3)).to(Num(3)+Num(1)))

egraph.register(
        rewrite(Num(p) + Num(q)).to(Num(p + q)),
        rewrite(x+y).to(y+x)
)


egraph.run(2)

print("after:", egraph.extract(a))

with open("after.svg", "w") as f:
    f.write(egraph._graphviz().pipe(format="svg", encoding="utf-8"))

egraph.check(eq(a).to(Num(5)))

