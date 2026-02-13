"""Report optimal wgmma configuration from the saturated e-graph.

Run:  python test_wgmma_lowering.py
"""
from __future__ import annotations
import wgmma_lowering as wl


# ── Extraction: what does the cost model prefer? ──────────────
best = wl.egraph.extract(wl.result)
print(f"Optimal lowering: {best}")

all_forms = wl.egraph.extract_multiple(wl.result, 10)
print(f"\n{len(all_forms)} equivalent forms discovered:")
for i, form in enumerate(all_forms, 1):
    print(f"  {i}. {form}")


# ── Metrics comparison (from Mode sort — never merged) ────────
print(f"\n{'Variant':<25} {'Regs':>5} {'SMEM reads':>11} {'Occup%':>7}")
print("-" * 52)

for name, mode in [("wgmma(reg, smem)", wl.Mode.reg_smem()),
                    ("wgmma(smem, smem)", wl.Mode.smem_smem())]:
    r  = int(wl.egraph.extract(wl.regs_used(mode)))
    sr = int(wl.egraph.extract(wl.smem_reads(mode)))
    o  = int(wl.egraph.extract(wl.occupancy_pct(mode)))
    print(f"{name:<25} {r:>5} {sr:>11} {o:>6}%")

print(f"\nExtractor chose: {best}")
