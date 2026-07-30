"""WP1 figures from results.json — read, plot, never recompute.

    python -m experiments.wp1_economy.figures

Writes into the WP1 paper's figures/ dir. The paper lives in the Obsidian vault,
not this repo (writing stays out of the codebase); this path is the one seam.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIGS = os.path.join(
    os.path.expanduser("~"), "Documents", "Productivity", "Obsidian",
    "Research", "Projects", "CI Library", "papers", "wp1-economy", "figures")


def band(ax, results, s):
    b = results["e_star_pred"]["bands"][f"s={s}"]
    ax.axvspan(b["machines"], b["min_v"], alpha=0.12, color="tab:blue",
               label="pre-registered e* band" if s == 0.5 else None)


def fig_knee(results):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for s, marker in [(0.25, "s"), (0.5, "o")]:
        rows = [r for r in results["E1"] if r["reinvest_rate"] == s]
        es = [r["efficiency"] for r in rows]
        ks = [r["capital_late"]["point"] for r in rows]
        lo = [r["capital_late"]["lo"] for r in rows]
        hi = [r["capital_late"]["hi"] for r in rows]
        ax.fill_between(es, lo, hi, alpha=0.2)
        ax.plot(es, ks, marker + "-", label=f"s = {s}")
        band(ax, results, s)
    ax.set_xlabel("capital efficiency e")
    ax.set_ylabel("late private capital")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "e1_knee.pdf"))
    print("[saved] e1_knee.pdf")


def fig_decoupling(results):
    rows = sorted(results["E2"], key=lambda r: r["recycle"])
    rs = [r["recycle"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    y0 = results["people_only"]["output_late"]["point"]
    ax1.axhline(y0, ls="--", c="gray", lw=1, label="people-only")
    ax1.errorbar(rs, [r["output_late"]["point"] for r in rows],
                 yerr=[[r["output_late"]["point"] - r["output_late"]["lo"] for r in rows],
                       [r["output_late"]["hi"] - r["output_late"]["point"] for r in rows]],
                 marker="o")
    ax1.set_xlabel("recycled share r"); ax1.set_ylabel("late output"); ax1.legend(fontsize=8)
    ax2.errorbar(rs, [r["ai_wealth_share"]["point"] for r in rows],
                 yerr=[[r["ai_wealth_share"]["point"] - r["ai_wealth_share"]["lo"] for r in rows],
                       [r["ai_wealth_share"]["hi"] - r["ai_wealth_share"]["point"] for r in rows]],
                 marker="o", color="tab:red")
    ax2.set_xlabel("recycled share r"); ax2.set_ylabel("AI wealth share")
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "e2_decoupling.pdf"))
    print("[saved] e2_decoupling.pdf")


def fig_defenses(results):
    rows = results["E3"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    base = [r for r in rows if r["ownership"] == 0.0]
    taus = [r["tax_rate"] for r in base]
    ax1.errorbar(taus, [r["capital_late"]["point"] for r in base],
                 yerr=[[r["capital_late"]["point"] - r["capital_late"]["lo"] for r in base],
                       [r["capital_late"]["hi"] - r["capital_late"]["point"] for r in base]],
                 marker="o", label="private K")
    ax1b = ax1.twinx()
    ax1b.plot(taus, [r["output_late"]["point"] for r in base], "s--", c="gray", ms=4,
              label="output")
    ax1.set_xlabel("profit tax rate τ"); ax1.set_ylabel("late private capital")
    ax1b.set_ylabel("late output"); ax1.legend(fontsize=8, loc="upper right")
    for mir, c in [(False, "tab:green"), (True, "tab:purple")]:
        sel = [r for r in rows if r["ownership"] == 0.4 and r["pub_mirror"] == mir]
        ax2.errorbar([r["tax_rate"] for r in sel],
                     [r["pub_share"]["point"] for r in sel], marker="o", color=c,
                     label="mirror fund (ratchet)" if mir else "dividend fund")
    ax2.axhline(0.4, ls=":", c="gray", lw=1)
    ax2.set_xlabel("profit tax rate τ"); ax2.set_ylabel("public share of capital")
    ax2.set_ylim(0, 1.05); ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "e3_defenses.pdf"))
    print("[saved] e3_defenses.pdf")


def main():
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(HERE, "results.json")) as fh:
        results = json.load(fh)
    fig_knee(results)
    fig_decoupling(results)
    fig_defenses(results)


if __name__ == "__main__":
    main()
