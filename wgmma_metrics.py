"""Analytical metric estimation for wgmma operand sourcing.

Just for one wgmma instruction: 
A : M x K = 64 x K
B : K x N 

K is limited by the data type and the physical mult acc lanes available for it

Computes register usage, occupancy, and SMEM bandwidth for different
tile configurations and data types on H100 (SM90), then runs egglog
to pick the best wgmma mode for each configuration.
"""
from __future__ import annotations
from egglog import *


# ── H100 (SM90) hardware constants ────────────────────────────
# Source: CUDA Programming Guide, Table 15 (Compute Capability 9.0)

REGS_PER_SM = 65536          # 32-bit registers per SM
MAX_REGS_PER_THREAD = 255    # hard architectural cap
MAX_THREADS_PER_SM = 2048    # max resident threads per SM
SMEM_PER_SM_KB = 228         # max shared memory per SM (configurable up to 228 KB)
WARPGROUP_THREADS = 128      # 4 warps × 32 threads = 1 warpgroup (unit for wgmma)
SMEM_BW_BYTES_PER_CYCLE = 128  # SM90 shared memory: 128 bytes/cycle per SM partition

# wgmma instruction: M is always 64 on SM90 (fixed by hardware).
# N can be 8, 16, 32, 64, 128, 256 depending on datatype.
# K depends on datatype: 16 for FP16/BF16, 8 for TF32, 32 for FP8/INT8.
# Source: PTX ISA 8.0, wgmma.mma_async instruction reference.
WGMMA_M = 64


# ── Data type properties ──────────────────────────────────────

DTYPES = {
    "FP16":  {"input_bytes": 2, "accum_bytes": 4, "K": 16, "max_N": 256}, #accum always 4 bytes
    "BF16":  {"input_bytes": 2, "accum_bytes": 4, "K": 16, "max_N": 256},
    "TF32":  {"input_bytes": 4, "accum_bytes": 4, "K": 8,  "max_N": 256},
    "FP8":   {"input_bytes": 1, "accum_bytes": 4, "K": 32, "max_N": 256},
    "INT8":  {"input_bytes": 1, "accum_bytes": 4, "K": 32, "max_N": 256},
}


# ── Analytical formulas ──────────────────────────────────────

def accum_regs(M: int, N: int, accum_bytes: int) -> int:
    """Accumulator registers per thread.

    Each warpgroup (128 threads) holds the full M×N output tile.
    Each element is accum_bytes wide; each register is 4 bytes (32-bit).
    regs = (M * N * accum_bytes) / (128 threads * 4 bytes/reg)
    """
    return (M * N * accum_bytes) // (WARPGROUP_THREADS * 4)


def a_descriptor_regs(M: int, K: int, input_bytes: int) -> int:
    """Extra registers for A operand in reg_smem mode.

    In reg_smem mode, operand A (M×K) must be loaded into registers.
    Each thread in the warpgroup holds a portion:
    regs = (M * K * input_bytes) / (128 threads * 4 bytes/reg)
    Minimum 8 regs (hardware descriptor overhead).
    """
    data_regs = (M * K * input_bytes) // (WARPGROUP_THREADS * 4)
    return max(data_regs, 8)  # at least 8 for the matrix descriptor


OVERHEAD_REGS = 16  # pipeline bookkeeping (~8) + address computation (~8)
# Source: empirical from ptxas --print-register-usage on CUTLASS kernels.
# Varies by kernel complexity; 16 is a conservative baseline.


def total_regs_per_thread(accum: int, a_desc: int, overhead: int = OVERHEAD_REGS) -> int:
    """Total registers per thread, rounded up to allocation granularity.

    SM90 allocates registers in multiples of 8 per thread.
    Source: CUDA Programming Guide, "Register Allocation Granularity".
    """
    raw = accum + a_desc + overhead
    return ((raw + 7) // 8) * 8  # round up to next multiple of 8


def occupancy_by_regs(regs_per_thread: int, block_threads: int = 128) -> float:
    """Occupancy limited by register usage.

    regs_per_block = regs_per_thread × block_threads
    Registers per block are allocated in chunks of 256.
    max_blocks = REGS_PER_SM / regs_per_block (rounded)
    occupancy = (max_blocks × block_threads) / MAX_THREADS_PER_SM
    """
    regs_per_block = regs_per_thread * block_threads
    # SM90: register allocation granularity is 256 registers per block
    regs_per_block = ((regs_per_block + 255) // 256) * 256
    max_blocks_by_regs = REGS_PER_SM // regs_per_block
    active_threads = max_blocks_by_regs * block_threads
    return min(active_threads / MAX_THREADS_PER_SM, 1.0)


def smem_per_tile(M: int, N: int, K: int, input_bytes: int) -> int:
    """Shared memory bytes for one A tile + one B tile.

    A tile: M × K × input_bytes
    B tile: K × N × input_bytes
    Double-buffered: ×2 (one being loaded while the other is consumed).
    """
    a_bytes = M * K * input_bytes
    b_bytes = K * N * input_bytes
    return (a_bytes + b_bytes) * 2  # double buffered. next one loaded with TMA


def smem_fits(smem_bytes: int) -> bool:
    """Does the tile fit in SM90's configurable shared memory?"""
    return smem_bytes <= SMEM_PER_SM_KB * 1024


# ── Sweep configurations ─────────────────────────────────────

TILE_NS = [64, 128, 256]

print(f"{'Config':<22} {'Acc':>4} {'A_desc':>6} {'Tot_reg':>7} {'Tot_smem':>8} "
      f"{'Occ_reg%':>8} {'Occ_smem%':>9} {'SMEM_fit':>8} {'Winner':>12}")
print("-" * 100)

for dtype_name, props in DTYPES.items():
    for N in TILE_NS:
        K = props["K"]
        M = WGMMA_M

        # ── Register analysis ──
        acc = accum_regs(M, N, props["accum_bytes"])
        a_desc = a_descriptor_regs(M, K, props["input_bytes"])

        regs_smem_smem = total_regs_per_thread(acc, 0)           # no A in regs
        regs_reg_smem  = total_regs_per_thread(acc, a_desc)      # A loaded to regs

        # ── Occupancy analysis ──
        occ_smem_smem = occupancy_by_regs(regs_smem_smem)
        occ_reg_smem  = occupancy_by_regs(regs_reg_smem)

        # ── Shared memory analysis ──
        smem_bytes = smem_per_tile(M, N, K, props["input_bytes"])
        fits = smem_fits(smem_bytes)

        # ── Check register legality ──
        legal_ss = regs_smem_smem <= MAX_REGS_PER_THREAD
        legal_rs = regs_reg_smem <= MAX_REGS_PER_THREAD

        # For each config, build a fresh e-graph with computed costs.
        egraph = EGraph()

        class Tile(Expr):
            @classmethod
            def smem(cls, name: StringLike) -> Tile: ...

        class Mode(Expr):
            @classmethod
            def reg_smem(cls) -> Mode: ...
            @classmethod
            def smem_smem(cls) -> Mode: ...

        # ── Cost model: register pressure (current) ──
        # Fewer regs = cheaper. Minimizing regs maximizes occupancy.
        # smem_smem always wins here because it avoids loading A into registers.
        @function(cost=regs_reg_smem)
        def wgmma_reg_smem(a: Tile, b: Tile) -> Tile: ...

        @function(cost=regs_smem_smem)
        def wgmma_smem_smem(a: Tile, b: Tile) -> Tile: ...

        @function(cost=regs_smem_smem + regs_reg_smem)  # high cost so extractor prefers a lowered form
        def gemm(a: Tile, b: Tile) -> Tile: ...

        @function(cost=a_desc)  # cost of loading A to regs
        def load_to_regs(t: Tile) -> Tile: ...

        # ── Alternative: use SMEM bandwidth as cost instead of regs ──
        # a_bytes = M * K * input_bytes;  b_bytes = K * N * input_bytes
        # cost_ss = (a_bytes + b_bytes) // 128   # reads both A,B from SMEM
        # cost_rs = b_bytes // 128 + 20          # reads only B (+ldmatrix)
        # reg_smem wins for large N where halving SMEM traffic matters.

        a, b = vars_("a b", Tile)

        A = Tile.smem("A")
        B = Tile.smem("B")
        result = egraph.let("result", gemm(A, B))

        egraph.register(
            rewrite(gemm(a, b)).to(wgmma_reg_smem(load_to_regs(a), b)),
            rewrite(gemm(a, b)).to(wgmma_smem_smem(a, b)),
        )
        egraph.run(10)

        best = egraph.extract(result)
        winner = "reg_smem" if "reg_smem" in str(best) else "smem_smem"

        # Mark illegal configs
        if not legal_rs:
            winner += "*" if winner == "reg_smem" else ""
        if not legal_ss:
            winner += "*" if winner == "smem_smem" else ""

        config = f"{dtype_name} {M}x{N}x{K}"
        print(f"{config:<22} {acc:>4} {a_desc:>6} "
              f"{regs_reg_smem:>3}/{regs_smem_smem:<3} "
              f"{smem_bytes:>7}B "
              f"{occ_reg_smem:>7.0%} {occ_smem_smem:>8.0%} "
              f"{'yes' if fits else 'NO':>8} {winner:>12}")

print()
print("* = exceeds 255 regs/thread (illegal on SM90)")
print("Tot_reg shows reg_smem/smem_smem")
print("Costs include 16 overhead regs (pipeline + addressing), rounded to multiple of 8")
