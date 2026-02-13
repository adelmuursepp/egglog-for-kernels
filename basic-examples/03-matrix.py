
from __future__ import annotations
from egglog import *
from collections.abc import Iterable

class Matrix(Expr):
  def __init__(self, rows: i64Like, cols: i64Like) -> None: ...
  def __matmul__(self, other: Matrix) -> Matrix: ...

  @property
  def row(self) -> i64: ...

  @property
  def col(self) -> i64: ...

egraph = EGraph()
Mexpr = egraph.let("Mexpr", (Matrix(64, 8) @ Matrix(8, 256)) @ Matrix(256, 2))
print("initial:", egraph.extract(Mexpr))


@egraph.register
def _(x: Matrix, y: Matrix, z: Matrix, r: i64, c: i64, m: i64) -> Iterable[RewriteOrRule]:
  # Rule 1: A matrix literal has its row and col from its constructor
  yield rule(x == Matrix(r, c)).then(
      set_(x.row).to(r),
      set_(x.col).to(c),
  )

  # Rule 2: A matrix product inherits dimensions from its operands
  yield rule(
      x == (y @ z),
      r == y.row,
      y.col == z.row,
      c == z.col,
  ).then(
      set_(x.row).to(r),
      set_(x.col).to(c),
  )

  # Rule 3: The cost of a matrix multiply is rows * shared * cols
  yield rule(
      y @ z,
      r == y.row,
      m == y.col,
      c == z.col,
  ).then(set_cost(y @ z, r * m * c))

  # Rule 4: Matrix multiplication is associative (both directions)
  yield birewrite(x @ (y @ z)).to((x @ y) @ z)

egraph.run(5)
print("optimized:", egraph.extract(Mexpr))
print("with cost:", egraph.extract(Mexpr, include_cost=True))
