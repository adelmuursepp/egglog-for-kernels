"""Report the optimal configuration from the saturated e-graph.

Run:  python test_poc_fusion.py
"""
from __future__ import annotations
import poc_fusion as pf


best = pf.egraph.extract(pf.original)
print(f"Original:  {pf.gemm(pf.X.divide(pf.X.square().sum_reduce()), pf.Y)}")
print(f"Optimized: {best}")

all_forms = pf.egraph.extract_multiple(pf.original, 10)
print(f"\n{len(all_forms)} equivalent forms:")
for i, form in enumerate(all_forms, 1):
    print(f"  {i}. {form}")

print("\n--- Warp Strategy Comparison ---")
print(f"{'Strategy':<20} {'Regs':>5} {'Occup%':>7} {'BW%':>5} {'Score':>6}")
best_name, best_score = "", 0
for name, expr in [("Homogeneous", pf.homo),
                    ("Producer-Consumer", pf.prodcon),
                    ("Pingpong", pf.pp)]:
    r  = int(pf.egraph.extract(pf.regs_per_thread(expr)))
    o  = int(pf.egraph.extract(pf.occupancy_pct(expr)))
    bw = int(pf.egraph.extract(pf.mem_bw_pct(expr)))
    sc = int(pf.egraph.extract(pf.composite_score(expr)))
    print(f"{name:<20} {r:>5} {o:>6}% {bw:>4}% {sc:>6}")
    if sc > best_score:
        best_score = sc
        best_name = name

print(f"\nWinner: {best_name} (score {best_score})")

# ── Lower-level: wgmma instruction selection ────────────────────
print("\n--- wgmma Instruction Selection ---")

# Build both lowered forms of the original gemm
wgmma_rr = pf.wgmma_reg_smem(pf.X, pf.Y)
wgmma_ss = pf.wgmma_smem_smem(pf.X, pf.Y)

def safe_int(expr):
    try:
        return int(pf.egraph.extract(expr))
    except Exception:
        return "N/A"

print(f"{'Variant':<20} {'Regs':>5} {'Occup%':>7} {'BW%':>5} {'RegLegal':>9}")
for name, expr in [("wgmma(reg,smem)", wgmma_rr),
                    ("wgmma(smem,smem)", wgmma_ss)]:
    r  = safe_int(pf.regs_per_thread(expr))
    o  = safe_int(pf.occupancy_pct(expr))
    bw = safe_int(pf.mem_bw_pct(expr))
    lg = safe_int(pf.reg_legal(expr))
    print(f"{name:<20} {r:>5} {o:>6}% {bw:>4}% {lg:>9}")
