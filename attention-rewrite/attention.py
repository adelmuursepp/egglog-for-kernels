"""
Simplified GPU attention kernel dataflow in egglog.

Encodes two equivalent dataflow graphs for attention:
  Naive: Q,K,V tiles loaded to SMEM, QK matmul, elementwise softmax/scale,
    AV matmul, store back. (see the first kernel in kernels-diagram)

  Rearranged: elementwise moved before QK matmul and applied to Q once
    (outside mainloop), saving repeated work. (see the second kernel in kernels-diagram)

Uses set_cost rules to assign traffic costs (bytes moved) to each op.
The WGMMA cost depends on whether each operand is in SHARED 
(implicit smem->reg load, costs bytes) or REGISTERS (no additional traffic).

A loop_iters analysis tracks how many times each op executes. 
Ops that consume streamed inputs (K, V) are in the mainloop and their costs
are multiplied by the trip count. 
This is what makes the rearranged path cheaper: elementwise runs once instead of every iteration.

Outputs: egraph.json
"""

from __future__ import annotations
from collections.abc import Iterable
from egglog import *
from egglog.egraph import to_runtime_expr
import json


class MemRegion(Expr):
    @classmethod
    def GLOBAL(cls) -> MemRegion: ...
    @classmethod
    def SHARED(cls) -> MemRegion: ...
    @classmethod
    def REGISTERS(cls) -> MemRegion: ...


class Tile(Expr):
    # These are all ways to create tile objects

    # Create a named tile with dimensions and data type
    # It starts in global memory
    @classmethod
    def input(cls, name: StringLike, rows: i64Like, cols: i64Like,
              dtype_bytes: i64Like) -> Tile: ...

    # Load a src tile from global to shared
    @classmethod
    def LDS(cls, src: Tile) -> Tile: ...

    # Load a src tile from shared to registers
    @classmethod
    def LDR(cls, src: Tile) -> Tile: ...

    # Matmul a and b tiles, result in registers
    @classmethod
    def WGMMA(cls, a: Tile, b: Tile) -> Tile: ...

    # Do an arbitrary elementwise op for src tile in registers, result in registers
    # This has no data movement
    @classmethod
    def Elementwise(cls, src: Tile) -> Tile: ...

    # Store from registers to shared, data movement = elements of tile x dtype
    @classmethod
    def STS(cls, src: Tile) -> Tile: ...

    # Store from shared to global, data movement = elements of tile x dtype
    @classmethod
    def STG(cls, src: Tile) -> Tile: ...


    # The following ones are the analysis properties
    # Metadata that gets propagated through rules
    @property
    def rows(self) -> i64: ...

    @property
    def cols(self) -> i64: ...

    @property
    def dtype_bytes(self) -> i64: ...

    # Metadata for where the tile currently lives
    # Assigned and tracked through rules
    @property
    def mem_region(self) -> MemRegion: ...

    # How many times this op executes. 1 = outside mainloop.
    # For example load tile of Q just once
    # Higher = inside mainloop (multiplied by trip count).
    # For example loading K and V tiles 8 times in the loop
    @property
    def loop_iters(self) -> i64: ...



def build_egraph(tiles, accum_dtype_bytes=4):
    """Build the attention egraph, run rewrites, return (egraph, result, A).

    tiles: dictionary of tile name -> (rows, cols, dtype_bytes, loop_iters)
    This is how the tiles will later be built:
    q_r, q_c, q_d, q_li = tiles["Q"]
    k_r, k_c, k_d, k_li = tiles["K"]
    v_r, v_c, v_d, v_li = tiles["V"]

    Q_input = Tile.input("Q", q_r, q_c, q_d)
    K_input = Tile.input("K", k_r, k_c, k_d)
    V_input = Tile.input("V", v_r, v_c, v_d)

    accum_dtype_bytes: dtype size for WGMMA output (4 = fp32)
    """
    egraph = EGraph()

    # Capture accum_dtype_bytes for use inside rules
    _accum_db = accum_dtype_bytes

    @egraph.register
    def _(t: Tile, a: Tile, b: Tile, s: String,
          r: i64, c: i64, d: i64,
          ar: i64, ac: i64, ad: i64,
          br: i64, bc: i64, bd: i64,
          al: i64, bl: i64,
          mr: MemRegion) -> Iterable[RewriteOrRule]:

        # If you see a input tile set its 
        yield rule(t == Tile.input(s, r, c, d)).then(
            set_(t.rows).to(r),
            set_(t.cols).to(c),
            set_(t.dtype_bytes).to(d),
            set_(t.mem_region).to(MemRegion.GLOBAL()),
        )

        # LDS: global -> shared, transaction = tile bytes * loop_iters
        yield rule(
            t == Tile.LDS(a),
            r == a.rows, c == a.cols, d == a.dtype_bytes, al == a.loop_iters,
        ).then(
            set_(t.rows).to(r),
            set_(t.cols).to(c),
            set_(t.dtype_bytes).to(d),
            set_(t.mem_region).to(MemRegion.SHARED()),
            set_(t.loop_iters).to(al),
            set_cost(Tile.LDS(a), r * c * d * al),
        )

        # LDR: shared -> registers, transaction = tile bytes * loop_iters
        yield rule(
            t == Tile.LDR(a),
            r == a.rows, c == a.cols, d == a.dtype_bytes, al == a.loop_iters,
        ).then(
            set_(t.rows).to(r),
            set_(t.cols).to(c),
            set_(t.dtype_bytes).to(d),
            set_(t.mem_region).to(MemRegion.REGISTERS()),
            set_(t.loop_iters).to(al),
            set_cost(Tile.LDR(a), r * c * d * al),
        )

        # WGMMA: transaction = implicit smem->reg load for shared operands.
        # Operands already in registers cost 0 (no additional traffic).
        # Output is rows(a) x cols(b), accum dtype, in registers.

        # Both operands in shared: implicit load of both
        yield rule(
            t == Tile.WGMMA(a, b), r == a.rows, c == b.cols,
            ar == a.rows, ac == a.cols, ad == a.dtype_bytes, al == a.loop_iters,
            br == b.rows, bc == b.cols, bd == b.dtype_bytes, bl == b.loop_iters,
            eq(a.mem_region).to(MemRegion.SHARED()),
            eq(b.mem_region).to(MemRegion.SHARED()),
        ).then(
            set_(t.rows).to(r),
            set_(t.cols).to(c),
            set_(t.dtype_bytes).to(i64(_accum_db)),
            set_(t.mem_region).to(MemRegion.REGISTERS()),
            set_(t.loop_iters).to(al.max(bl)),
            set_cost(Tile.WGMMA(a, b), (ar * ac * ad + br * bc * bd) * al.max(bl)),
        )

        # a in registers, b in shared: implicit load of b only
        yield rule(
            t == Tile.WGMMA(a, b), r == a.rows, c == b.cols,
            al == a.loop_iters,
            br == b.rows, bc == b.cols, bd == b.dtype_bytes, bl == b.loop_iters,
            eq(a.mem_region).to(MemRegion.REGISTERS()),
            eq(b.mem_region).to(MemRegion.SHARED()),
        ).then(
            set_(t.rows).to(r),
            set_(t.cols).to(c),
            set_(t.dtype_bytes).to(i64(_accum_db)),
            set_(t.mem_region).to(MemRegion.REGISTERS()),
            set_(t.loop_iters).to(al.max(bl)),
            set_cost(Tile.WGMMA(a, b), br * bc * bd * al.max(bl)),
        )

        # Both in registers: no data movement
        yield rule(
            t == Tile.WGMMA(a, b), r == a.rows, c == b.cols,
            al == a.loop_iters, bl == b.loop_iters,
            eq(a.mem_region).to(MemRegion.REGISTERS()),
            eq(b.mem_region).to(MemRegion.REGISTERS()),
        ).then(
            set_(t.rows).to(r),
            set_(t.cols).to(c),
            set_(t.dtype_bytes).to(i64(_accum_db)),
            set_(t.mem_region).to(MemRegion.REGISTERS()),
            set_(t.loop_iters).to(al.max(bl)),
            set_cost(Tile.WGMMA(a, b), i64(0)),
        )

        # Elementwise: pure compute, 0 transaction cost
        yield rule(
            t == Tile.Elementwise(a),
            r == a.rows, c == a.cols, d == a.dtype_bytes,
            mr == a.mem_region, al == a.loop_iters,
        ).then(
            set_(t.rows).to(r),
            set_(t.cols).to(c),
            set_(t.dtype_bytes).to(d),
            set_(t.mem_region).to(mr),
            set_(t.loop_iters).to(al),
            set_cost(Tile.Elementwise(a), i64(0)),
        )

        # STS: registers -> shared, transaction = tile bytes * loop_iters
        yield rule(
            t == Tile.STS(a),
            r == a.rows, c == a.cols, d == a.dtype_bytes, al == a.loop_iters,
        ).then(
            set_(t.rows).to(r),
            set_(t.cols).to(c),
            set_(t.dtype_bytes).to(d),
            set_(t.mem_region).to(MemRegion.SHARED()),
            set_(t.loop_iters).to(al),
            set_cost(Tile.STS(a), r * c * d * al),
        )

        # STG: shared -> global, transaction = tile bytes * loop_iters
        yield rule(
            t == Tile.STG(a),
            r == a.rows, c == a.cols, d == a.dtype_bytes, al == a.loop_iters,
        ).then(
            set_(t.rows).to(r),
            set_(t.cols).to(c),
            set_(t.dtype_bytes).to(d),
            set_(t.mem_region).to(MemRegion.GLOBAL()),
            set_(t.loop_iters).to(al),
            set_cost(Tile.STG(a), r * c * d * al),
        )

        # Rewrite 1: move elementwise before matmul
        # Elementwise(WGMMA(q, k)) => WGMMA(Elementwise(LDR(q)), k)
        # When a scale sits on top of a matmul, move it before (valid because scale
        # is linear) and introduce an explicit LDR. Combined effect: Q is scaled
        # once outside the loop and stays in registers for all iterations.
        # Conditioned on a.mem_region == SHARED for the same reason as Rewrite 2:
        # prevents firing when a is already in registers (e.g. LDR(Q)), which
        # would create a LDR(LDR(Q)) term.
        yield rule(
            t == Tile.Elementwise(Tile.WGMMA(a, b)),
            eq(a.mem_region).to(MemRegion.SHARED()),
        ).then(
            union(t).with_(Tile.WGMMA(Tile.Elementwise(Tile.LDR(a)), b))
        )

        # Rewrite 2: explicitly load left WGMMA operand to registers when in SMEM.
        # WGMMA(a_smem, b) => WGMMA(LDR(a_smem), b)
        # More primitive than Rewrite 1: fires on any WGMMA where a is in SMEM,
        # independent of whether an elementwise op is present above it.
        # The ILP chooses: pay for LDR once (bytes(a) * a.loop_iters) vs. let
        # WGMMA reload a implicitly each iteration (bytes(a) * loop_iters_of_loop).
        # Conditioned on a.mem_region == SHARED to prevent chaining: once a is
        # in REGISTERS the rule no longer matches and saturation is reached.
        yield rule(
            t == Tile.WGMMA(a, b),
            eq(a.mem_region).to(MemRegion.SHARED()),
        ).then(
            union(t).with_(Tile.WGMMA(Tile.LDR(a), b))
        )

    # Build tile inputs from the tiles dict
    q_r, q_c, q_d, q_li = tiles["Q"]
    k_r, k_c, k_d, k_li = tiles["K"]
    v_r, v_c, v_d, v_li = tiles["V"]

    Q_input = Tile.input("Q", q_r, q_c, q_d)
    K_input = Tile.input("K", k_r, k_c, k_d)
    V_input = Tile.input("V", v_r, v_c, v_d)

    egraph.register(
        set_(Q_input.loop_iters).to(i64(q_li)),
        set_(K_input.loop_iters).to(i64(k_li)),
        set_(V_input.loop_iters).to(i64(v_li)),
    )

    Q_smem = Tile.LDS(Q_input)
    K_smem = Tile.LDS(K_input)
    V_smem = Tile.LDS(V_input)

    QK = Tile.WGMMA(Q_smem, K_smem)
    A = Tile.Elementwise(QK)
    output = Tile.WGMMA(A, V_smem)

    result = egraph.let("attention_output", Tile.STG(Tile.STS(output)))

    egraph.run(10)
    return egraph, result, A


def serialize_egraph(egraph, root_exprs=None):
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


def build_and_serialize(output_path="output/egraph.json", tiles=None, accum_dtype_bytes=4):
    """Build egraph, print tree extraction, serialize to JSON."""
    if tiles is None:
        tiles = {"Q": (128, 64, 2, 1), "K": (64, 128, 2, 8), "V": (128, 64, 2, 8)}
    egraph, result, A = build_egraph(tiles=tiles, accum_dtype_bytes=accum_dtype_bytes)

    print("Tree extraction (set_cost = traffic bytes * loop_iters):")
    tree_best, tree_cost = egraph.extract(result, include_cost=True)
    print(f"  {tree_best}")
    print(f"  cost: {tree_cost}")
    print()

    print("Variants of the attention weights (A) e-class:")
    A_variants = egraph.extract_multiple(A, 10)
    for i, v in enumerate(A_variants):
        print(f"  {i+1}. {v}")
    print()

    graph_json = serialize_egraph(egraph, root_exprs=[result])
    with open(output_path, "w") as f:
        json.dump(graph_json, f, indent=2)
    print(f"Wrote {output_path}")

    return output_path

# For running as standalone file
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "output/egraph.json"
    build_and_serialize(path)
