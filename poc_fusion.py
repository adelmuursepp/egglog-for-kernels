"""GPU Kernel Fusion via Equality Saturation (egglog).

Defines the e-graph, sorts, functions, rewrite rules,
Datalog rules, builds the computation graph, and runs saturation.

See test_poc_fusion.py for assertions and result output.
"""
from __future__ import annotations
from egglog import *


egraph = EGraph()


# ── Sorts: computation graph ─────────────────────────────────────

class Op(Expr):
    @classmethod
    def input(cls, name: StringLike) -> Op: ...

    @method(cost=10)  # single-kernel elementwise op
    def square(self) -> Op: ...

    @method(cost=10)  # single-kernel reduction
    def sum_reduce(self) -> Op: ...

    @method(cost=10)  # single-kernel elementwise op
    def divide(self, divisor: Op) -> Op: ...


@function(cost=10)  # standard matrix multiply kernel
def gemm(a: Op, b: Op) -> Op: ...


@function(cost=12)  # slightly costlier single kernel, but eliminates a launch + mem round-trip
def fused_gemm_div(a: Op, b: Op, divisor: Op) -> Op: ...


# ── Lower-level ops: memory movement ────────────────────────────
# Explicit data movement between memory hierarchy levels

@function(cost=20)  # H100 lookup: ldmatrix / lds latency ~20 cycles
def load_to_regs(a: Op) -> Op: ...

@function(cost=10)  # H100 lookup: st.shared latency ~10 cycles
def store_to_smem(a: Op) -> Op: ...

@function(cost=150)  # H100 lookup: cp.async.bulk (TMA) latency ~150 cycles, but async
def tma_load(a: Op) -> Op: ...


# ── Lower-level ops: wgmma instruction variants ─────────────────
# H100 wgmma.mma_async has two operand sourcing modes (lookup: PTX ISA 8.0, wgmma instruction)

@function(cost=8)   # A from registers, B from SMEM — faster if A is already in regs
def wgmma_reg_smem(a: Op, b: Op) -> Op: ...

@function(cost=10)  # A from SMEM, B from SMEM — saves registers but slightly higher latency
def wgmma_smem_smem(a: Op, b: Op) -> Op: ...


# ── Lower-level analysis: register budget ───────────────────────
# H100 lookup: 65536 32-bit regs per SM, 255 max per thread, 2048 max threads per SM

@function(merge=lambda old, new: old.max(new))
def total_regs(op: Op) -> i64: ...

@function(merge=lambda old, new: old.min(new))
def reg_legal(op: Op) -> i64: ...  # 1 = within budget, 0 = exceeds H100 register limit


# ── Sorts: memory hierarchy ──────────────────────────────────────

class MemRegion(Expr):
    @classmethod
    def register_file(cls) -> MemRegion: ...
    @classmethod
    def shared(cls) -> MemRegion: ...
    @classmethod
    def global_(cls) -> MemRegion: ...


@function
def mem_location(op: Op) -> MemRegion: ...

@function
def transfer_cost(src: MemRegion, dst: MemRegion) -> i64: ...


# ── Kernel metrics ───────────────────────────────────────────────

@function(merge=lambda old, new: old.min(new))
def num_launches(op: Op) -> i64: ...

@function(merge=lambda old, new: old.max(new))
def regs_per_thread(op: Op) -> i64: ...

@function(merge=lambda old, new: old.max(new))
def occupancy_pct(op: Op) -> i64: ...

@function(merge=lambda old, new: old.max(new))
def mem_bw_pct(op: Op) -> i64: ...


# ── Fusion legality predicates ───────────────────────────────────

@function(merge=lambda old, new: old.min(new))
def fits_in_smem(op: Op) -> i64: ...

@function(merge=lambda old, new: old.min(new))
def needs_reload(op: Op) -> i64: ...


# ── Warp specialization ─────────────────────────────────────────

class WarpSpec(Expr):
    @classmethod
    def homogeneous(cls) -> WarpSpec: ...
    @classmethod
    def producer_consumer(cls) -> WarpSpec: ...
    @classmethod
    def pingpong(cls) -> WarpSpec: ...


@function(cost=1)  # cheap wrapper — just annotates which warp strategy to use
def with_warp_spec(op: Op, ws: WarpSpec) -> Op: ...


class Result(Expr):
    pass

@function
def result_of(op: Op) -> Result: ...

@function(merge=lambda old, new: old.max(new))
def composite_score(op: Op) -> i64: ...


# ── Computation graph ────────────────────────────────────────────
#    result = GEMM( X / sum(X^2),  Y )

X = Op.input("X")
Y = Op.input("Y")

original = egraph.let("original",
    gemm(X.divide(X.square().sum_reduce()), Y)
)


# ── Rewrite rules ───────────────────────────────────────────────

a, b, c = vars_("a b c", Op)

egraph.register(
    # Rearrangement:  GEMM(A/c, B) == GEMM(A,B)/c
    rewrite(gemm(a.divide(c), b)).to(gemm(a, b).divide(c)),
    # Fusion pattern: GEMM(A,B)/c  -> fused_gemm_div(A,B,c)
    rewrite(gemm(a, b).divide(c)).to(fused_gemm_div(a, b, c)),
)


# ── Lower-level rewrite rules ─────────────────────────────────────

egraph.register(
    # Store-load elimination: storing to SMEM then loading back is a no-op if already in regs
    rewrite(load_to_regs(store_to_smem(a))).to(a),

    # GEMM can be lowered to either wgmma variant — same result, different resource usage
    rewrite(gemm(a, b)).to(wgmma_reg_smem(load_to_regs(a), b)),
    rewrite(gemm(a, b)).to(wgmma_smem_smem(a, b)),

    # Both wgmma variants produce the same mathematical result
    birewrite(wgmma_reg_smem(a, b)).to(wgmma_smem_smem(a, b)),
)


# ── Lower-level memory placement rules ────────────────────────────

egraph.register(
    # load_to_regs: output is in registers
    rule(
        load_to_regs(a),
    ).then(
        union(mem_location(load_to_regs(a))).with_(MemRegion.register_file()),
    ),

    # store_to_smem: output is in shared memory
    rule(
        store_to_smem(a),
    ).then(
        union(mem_location(store_to_smem(a))).with_(MemRegion.shared()),
    ),

    # tma_load: output lands in shared memory (H100 lookup: TMA writes to SMEM)
    rule(
        tma_load(a),
    ).then(
        union(mem_location(tma_load(a))).with_(MemRegion.shared()),
    ),

    # wgmma output always accumulates in registers (H100 lookup: accumulator is in register file)
    rule(wgmma_reg_smem(a, b)).then(
        union(mem_location(wgmma_reg_smem(a, b))).with_(MemRegion.register_file()),
    ),
    rule(wgmma_smem_smem(a, b)).then(
        union(mem_location(wgmma_smem_smem(a, b))).with_(MemRegion.register_file()),
    ),
)


# ── Lower-level register budget rules ─────────────────────────────

r = var("r", i64)

egraph.register(
    # wgmma_reg_smem: A in regs adds ~8 regs for the 16x16 descriptor (H100 lookup: wgmma register mapping)
    rule(
        wgmma_reg_smem(a, b),
        eq(regs_per_thread(a)).to(r),
    ).then(
        set_(total_regs(wgmma_reg_smem(a, b))).to(r + i64(8)),
    ),

    # wgmma_smem_smem: no extra regs for A, just accumulator (~4 regs baseline)
    rule(
        wgmma_smem_smem(a, b),
    ).then(
        set_(total_regs(wgmma_smem_smem(a, b))).to(i64(4)),
    ),

    # H100 hard limit: 255 regs per thread max (H100 lookup: SM90 register file constraint)
    # Legal if within budget
    rule(
        eq(total_regs(a)).to(r),
        r <= i64(255),
    ).then(
        set_(reg_legal(a)).to(i64(1)),
    ),
)

# Metric rules for wgmma variants
egraph.register(
    rule(wgmma_reg_smem(a, b)).then(
        set_(regs_per_thread(wgmma_reg_smem(a, b))).to(i64(40)),   # base + A occupies regs
        set_(occupancy_pct(wgmma_reg_smem(a, b))).to(i64(75)),      # reduced by register pressure
        set_(mem_bw_pct(wgmma_reg_smem(a, b))).to(i64(70)),         # B still streamed from SMEM
        set_(num_launches(wgmma_reg_smem(a, b))).to(i64(1)),
    ),
    rule(wgmma_smem_smem(a, b)).then(
        set_(regs_per_thread(wgmma_smem_smem(a, b))).to(i64(32)),  # fewer regs, both operands in SMEM
        set_(occupancy_pct(wgmma_smem_smem(a, b))).to(i64(100)),    # full occupancy, low register pressure
        set_(mem_bw_pct(wgmma_smem_smem(a, b))).to(i64(60)),        # both operands compete for SMEM bandwidth
        set_(num_launches(wgmma_smem_smem(a, b))).to(i64(1)),
    ),
)


# ── Memory propagation rules ────────────────────────────────────

egraph.register(
    union(mem_location(X)).with_(MemRegion.global_()),
    union(mem_location(Y)).with_(MemRegion.global_()),

    set_(transfer_cost(MemRegion.global_(),       MemRegion.shared())).to(i64(150)),        # ~150 cycles for global→SMEM load
    set_(transfer_cost(MemRegion.shared(),        MemRegion.register_file())).to(i64(20)),  # ~20 cycles for SMEM→register load
    set_(transfer_cost(MemRegion.global_(),       MemRegion.register_file())).to(i64(350)), # ~350 cycles for uncached global→register
    set_(transfer_cost(MemRegion.register_file(), MemRegion.register_file())).to(i64(0)),   # free: already in registers

    # Guarded propagation (prevents infinite chains)
    rule(
        a.square(),
        eq(mem_location(a)).to(MemRegion.register_file()),
    ).then(
        union(mem_location(a.square())).with_(MemRegion.register_file()),
    ),
    rule(
        a.sum_reduce(),
        eq(mem_location(a)).to(MemRegion.register_file()),
    ).then(
        union(mem_location(a.sum_reduce())).with_(MemRegion.register_file()),
    ),
    rule(
        a.divide(b),
        eq(mem_location(a)).to(MemRegion.register_file()),
        eq(mem_location(b)).to(MemRegion.register_file()),
    ).then(
        union(mem_location(a.divide(b))).with_(MemRegion.register_file()),
    ),

    rule(gemm(a, b)).then(
        union(mem_location(gemm(a, b))).with_(MemRegion.register_file()),
    ),
    rule(fused_gemm_div(a, b, c)).then(
        union(mem_location(fused_gemm_div(a, b, c))).with_(MemRegion.register_file()),
    ),
)


# ── Fusion legality rules ───────────────────────────────────────

egraph.register(
    # Assumed small enough for SMEM; in production would be computed from tensor dims vs GPU SMEM capacity
    set_(fits_in_smem(X)).to(i64(1)),
    set_(fits_in_smem(Y)).to(i64(1)),

    rule(a.square(), eq(fits_in_smem(a)).to(i64(1))).then(
        set_(fits_in_smem(a.square())).to(i64(1)),
    ),
    rule(a.sum_reduce(), eq(fits_in_smem(a)).to(i64(1))).then(
        set_(fits_in_smem(a.sum_reduce())).to(i64(1)),
    ),

    rule(
        eq(mem_location(gemm(a, b))).to(MemRegion.register_file()),
        eq(mem_location(a)).to(MemRegion.register_file()),
    ).then(
        set_(needs_reload(a)).to(i64(0)),  # 0 = no reload needed, data already in registers
    ),
    rule(eq(mem_location(a)).to(MemRegion.global_())).then(
        set_(needs_reload(a)).to(i64(1)),  # 1 = reload needed, data stuck in global memory
    ),
)


# ── Kernel metric rules ─────────────────────────────────────────

egraph.register(
    # Unfused: separate kernels for divide, square, sum_reduce, gemm
    rule(gemm(a.divide(c), b)).then(
        set_(num_launches(gemm(a.divide(c), b))).to(i64(4)),    # 4 kernel launches (square + sum + div + gemm)
        set_(regs_per_thread(gemm(a.divide(c), b))).to(i64(32)),  # low register pressure, simple kernels
        set_(occupancy_pct(gemm(a.divide(c), b))).to(i64(100)),   # full occupancy since each kernel is small
        set_(mem_bw_pct(gemm(a.divide(c), b))).to(i64(40)),       # low bandwidth use, bottlenecked by launch overhead
    ),
    # Fused: single kernel does gemm + divide together
    rule(fused_gemm_div(a, b, c)).then(
        set_(num_launches(fused_gemm_div(a, b, c))).to(i64(1)),    # 1 launch, everything in one kernel
        set_(regs_per_thread(fused_gemm_div(a, b, c))).to(i64(128)), # high register pressure from fusing ops
        set_(occupancy_pct(fused_gemm_div(a, b, c))).to(i64(50)),    # lower occupancy due to register pressure
        set_(mem_bw_pct(fused_gemm_div(a, b, c))).to(i64(85)),       # high bandwidth use, no intermediate mem trips
    ),
)


# ── Warp specialization rules ───────────────────────────────────

ws_t = var("ws_t", Op)
ws_w = var("ws_w", WarpSpec)

egraph.register(
    rule(with_warp_spec(ws_t, ws_w)).then(
        union(result_of(with_warp_spec(ws_t, ws_w))).with_(result_of(ws_t))
    ),

    # Homogeneous: all warps do the same work, simple but underutilizes memory
    rule(with_warp_spec(a, WarpSpec.homogeneous())).then(
        set_(regs_per_thread(with_warp_spec(a, WarpSpec.homogeneous()))).to(i64(40)),   # low regs, simple uniform work
        set_(occupancy_pct(with_warp_spec(a, WarpSpec.homogeneous()))).to(i64(100)),     # full occupancy
        set_(mem_bw_pct(with_warp_spec(a, WarpSpec.homogeneous()))).to(i64(60)),         # moderate bandwidth
        set_(num_launches(with_warp_spec(a, WarpSpec.homogeneous()))).to(i64(1)),
        set_(composite_score(with_warp_spec(a, WarpSpec.homogeneous()))).to(i64(180)),   # occupancy + mem_bw = 100+60+20 (base)
    ),

    # Producer-consumer: some warps load data, others compute — best bandwidth
    rule(with_warp_spec(a, WarpSpec.producer_consumer())).then(
        set_(regs_per_thread(with_warp_spec(a, WarpSpec.producer_consumer()))).to(i64(56)),  # more regs for pipelining state
        set_(occupancy_pct(with_warp_spec(a, WarpSpec.producer_consumer()))).to(i64(75)),     # reduced by higher reg usage
        set_(mem_bw_pct(with_warp_spec(a, WarpSpec.producer_consumer()))).to(i64(95)),        # near-peak, overlaps loads + compute
        set_(num_launches(with_warp_spec(a, WarpSpec.producer_consumer()))).to(i64(1)),
        set_(composite_score(with_warp_spec(a, WarpSpec.producer_consumer()))).to(i64(209)),  # highest score: bandwidth wins here
    ),

    # Pingpong: double-buffered, warps alternate between load and compute phases
    rule(with_warp_spec(a, WarpSpec.pingpong())).then(
        set_(regs_per_thread(with_warp_spec(a, WarpSpec.pingpong()))).to(i64(48)),   # moderate regs for double buffering
        set_(occupancy_pct(with_warp_spec(a, WarpSpec.pingpong()))).to(i64(80)),      # decent occupancy
        set_(mem_bw_pct(with_warp_spec(a, WarpSpec.pingpong()))).to(i64(80)),         # good bandwidth from overlapping
        set_(num_launches(with_warp_spec(a, WarpSpec.pingpong()))).to(i64(1)),
        set_(composite_score(with_warp_spec(a, WarpSpec.pingpong()))).to(i64(192)),   # middle ground between homo and prod-con
    ),
)


# ── Warp-specialized variants ───────────────────────────────────

fused = fused_gemm_div(X, Y, X.square().sum_reduce())

homo    = egraph.let("homo",     with_warp_spec(fused, WarpSpec.homogeneous()))
prodcon = egraph.let("prodcon",  with_warp_spec(fused, WarpSpec.producer_consumer()))
pp      = egraph.let("pingpong", with_warp_spec(fused, WarpSpec.pingpong()))


# ── Run equality saturation ─────────────────────────────────────

egraph.run(10)  # 10 iterations is enough for this small graph to saturate
