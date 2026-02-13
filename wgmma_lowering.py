"""wgmma operand sourcing via equality saturation.

H100 wgmma.mma_async has two modes (PTX ISA 8.0):
  1. A from registers, B from SMEM
  2. A from SMEM, B from SMEM

Given gemm(A, B) where both operands start in SMEM, explore whether
loading A to registers first is worth the register cost.
"""
from __future__ import annotations
from egglog import *


egraph = EGraph()


# ── Sorts ───────────────────────────────────────────────────────

class Tile(Expr):
    """A matrix tile that lives somewhere in the memory hierarchy."""
    @classmethod
    def smem(cls, name: StringLike) -> Tile: ...


class MemRegion(Expr):
    @classmethod
    def register_file(cls) -> MemRegion: ...
    @classmethod
    def shared(cls) -> MemRegion: ...


class Mode(Expr):
    """Wgmma variant tag — stays in its own e-class so metrics don't merge."""
    @classmethod
    def reg_smem(cls) -> Mode: ...
    @classmethod
    def smem_smem(cls) -> Mode: ...


# ── Operations ──────────────────────────────────────────────────

@function(cost=20)  # H100: ldmatrix ~20 cycles. maybe better if num of movements
def load_to_regs(t: Tile) -> Tile: ...

@function(cost=10)  # H100: st.shared ~10 cycles
def store_to_smem(t: Tile) -> Tile: ...

# The two wgmma modes — same result, different resource tradeoffs
@function(cost=8)   # A from regs, B from SMEM — lower latency per instruction
def wgmma_reg_smem(a: Tile, b: Tile) -> Tile: ...

@function(cost=10)  # A from SMEM, B from SMEM — saves registers
def wgmma_smem_smem(a: Tile, b: Tile) -> Tile: ...

# High-level gemm — will be lowered to one of the wgmma variants
@function(cost=15)
def gemm(a: Tile, b: Tile) -> Tile: ...


# ── Analysis functions (on Mode, not Tile — so they don't merge) ──

@function
def mem_loc(t: Tile) -> MemRegion: ...

@function
def regs_used(m: Mode) -> i64: ...

@function
def smem_reads(m: Mode) -> i64: ...

@function
def occupancy_pct(m: Mode) -> i64: ...


# ── Inputs ──────────────────────────────────────────────────────
# Both tiles start in shared memory (loaded by TMA before the wgmma loop)

A = Tile.smem("A")
B = Tile.smem("B")

result = egraph.let("result", gemm(A, B))


# ── Rewrite rules ──────────────────────────────────────────────

a, b = vars_("a b", Tile)

egraph.register(
    # Lowering: gemm can become either wgmma variant
    rewrite(gemm(a, b)).to(wgmma_reg_smem(load_to_regs(a), b)),  # load A to regs first
    rewrite(gemm(a, b)).to(wgmma_smem_smem(a, b)),                # use A directly from SMEM

    # Store-load elimination: round-trip through SMEM is a no-op
    rewrite(load_to_regs(store_to_smem(a))).to(a),
)


# ── Memory location propagation ────────────────────────────────

egraph.register(
    # Inputs live in SMEM
    union(mem_loc(A)).with_(MemRegion.shared()),
    union(mem_loc(B)).with_(MemRegion.shared()),

    # load_to_regs moves data to register file
    rule(load_to_regs(a)).then(
        union(mem_loc(load_to_regs(a))).with_(MemRegion.register_file()),
    ),
    # store_to_smem moves data to shared memory
    rule(store_to_smem(a)).then(
        union(mem_loc(store_to_smem(a))).with_(MemRegion.shared()),
    ),
    # wgmma accumulator always lands in registers (H100: accumulator in RF)
    rule(wgmma_reg_smem(a, b)).then(
        union(mem_loc(wgmma_reg_smem(a, b))).with_(MemRegion.register_file()),
    ),
    rule(wgmma_smem_smem(a, b)).then(
        union(mem_loc(wgmma_smem_smem(a, b))).with_(MemRegion.register_file()),
    ),
)


# ── Metric rules (attached to Mode, not Tile) ────────────────
# Assumption: 64x64 tile with FP32 accumulator, 1 warpgroup (128 threads).
# Accumulator regs/thread = (M * N * sizeof(accum)) / (128 threads * 4 bytes/reg)
#   e.g. 64x64 FP32: (64*64*4)/(128*4) = 32 regs
#        64x128 FP32: (64*128*4)/(128*4) = 64 regs
#        64x256 FP32: (64*256*4)/(128*4) = 128 regs
# To generalize: parameterize M, N, accum_bytes and compute instead of hardcoding.

egraph.register(
    # wgmma(reg, smem): A in regs costs ~8 extra regs for the descriptor
    set_(regs_used(Mode.reg_smem())).to(i64(40)),       # 32 accum (64x64 FP32) + 8 A descriptor
    set_(smem_reads(Mode.reg_smem())).to(i64(1)),        # only B read from SMEM
    set_(occupancy_pct(Mode.reg_smem())).to(i64(75)),    # reduced occupancy from higher reg usage

    # wgmma(smem, smem): no regs for A, both operands compete for SMEM bandwidth
    set_(regs_used(Mode.smem_smem())).to(i64(32)),      # 32 accum only (64x64 FP32)
    set_(smem_reads(Mode.smem_smem())).to(i64(2)),       # both A and B read from SMEM
    set_(occupancy_pct(Mode.smem_smem())).to(i64(100)),  # full occupancy, low reg pressure
)


# ── Run ─────────────────────────────────────────────────────────

egraph.run(10)
