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


def fig_headline(results):
    """Paper Fig. 2 — production grows while the human share of income falls."""
    h = results["headline"]
    t = list(range(len(h["production"])))
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.axhline(h["people_only_production"], ls="--", c="gray", lw=1,
               label="production, no AI")
    lo = [p - s for p, s in zip(h["production"], h["production_sd"])]
    hi = [p + s for p, s in zip(h["production"], h["production_sd"])]
    ax.fill_between(t, lo, hi, alpha=0.2, color="tab:blue")
    ax.plot(t, h["production"], c="tab:blue", label="production")
    ax.axvline(h["first_arrival"], ls=":", c="black", lw=1)
    ax.annotate("AI capital arrives", (h["first_arrival"] + 6, ax.get_ylim()[0]),
                fontsize=8, rotation=0, va="bottom")
    ax.set_xlabel("time"); ax.set_ylabel("production (gross output)")
    ax.set_ylim(0, None)
    axr = ax.twinx()
    axr.plot(t, h["human_share"], c="tab:green", label="human share of income")
    axr.set_ylabel("human share of income"); axr.set_ylim(0, 1.05)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = axr.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "headline.pdf"))
    print("[saved] headline.pdf")


def fig_money(results):
    """Paper Fig. 3 — (a) where the AI's income goes; (b) taxing it back."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))

    rows = sorted(results["E2"], key=lambda r: r["recycle"])
    rs = [r["recycle"] for r in rows]
    y0 = results["people_only"]["output_late"]["point"]
    ax1.axhline(y0, ls="--", c="gray", lw=1, label="no AI")
    ax1.errorbar(rs, [r["output_late"]["point"] for r in rows],
                 yerr=[[r["output_late"]["point"] - r["output_late"]["lo"] for r in rows],
                       [r["output_late"]["hi"] - r["output_late"]["point"] for r in rows]],
                 marker="o", c="tab:blue", label="production")
    ax1.set_xlabel("share of AI income spent back, $r$")
    ax1.set_ylabel("production"); ax1.set_ylim(0, None)
    ax1r = ax1.twinx()
    ax1r.plot(rs, [r["ai_wealth_share"]["point"] for r in rows], "s-",
              c="tab:red", ms=4, label="AI wealth share")
    ax1r.set_ylim(0, 1.05); ax1r.set_ylabel("AI share of wealth")
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1r.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="lower center")

    base = [r for r in results["E3"] if r["ownership"] == 0.0]
    taus = [r["tax_rate"] for r in base]
    ax2.axhline(results["people_only"]["hh_income_late"]["point"], ls="--",
                c="gray", lw=1, label="no AI")
    ax2.errorbar(taus, [r["hh_income_late"]["point"] for r in base],
                 yerr=[[r["hh_income_late"]["point"] - r["hh_income_late"]["lo"] for r in base],
                       [r["hh_income_late"]["hi"] - r["hh_income_late"]["point"] for r in base]],
                 marker="o", c="tab:blue", label="household income")
    ax2.set_xlabel("profit tax rate $\\tau$")
    ax2.set_ylabel("household income"); ax2.set_ylim(0, None)
    ax2r = ax2.twinx()
    ax2r.plot(taus, [r["ai_wealth_share"]["point"] for r in base], "s-",
              c="tab:red", ms=4, label="AI wealth share")
    ax2r.set_ylim(0, 1.05); ax2r.set_ylabel("AI share of wealth")
    l1, lb1 = ax2.get_legend_handles_labels()
    l2, lb2 = ax2r.get_legend_handles_labels()
    ax2.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="lower center")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "money.pdf"))
    print("[saved] money.pdf")


def fig_endstate(results):
    """Paper Fig. 4 — end-state human share against capability; e*/e law."""
    rows = results["E5"]
    es = [r["efficiency"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(es, [r["h_pred"] for r in rows], "--", c="gray",
            label="committed prediction $\\min(1, e^*/e)$")
    ax.errorbar(es, [r["human_share"]["point"] for r in rows],
                yerr=[[r["human_share"]["point"] - r["human_share"]["lo"] for r in rows],
                      [r["human_share"]["hi"] - r["human_share"]["point"] for r in rows]],
                marker="o", c="tab:green", label="measured")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("capital efficiency $e$ (log scale)")
    ax.set_ylabel("late human share of income")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "endstate.pdf"))
    print("[saved] endstate.pdf")


def fig_growth(results):
    """Paper Fig. 5 — collapse trajectories under the capability-growth models."""
    styles = {"static": ("tab:green", "-"), "slow": ("tab:blue", "-"),
              "central": ("tab:orange", "-"), "fast": ("tab:red", "-"),
              "rsi": ("tab:purple", "--")}
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for row in results["E6"]:
        c, ls = styles[row["label"]]
        lab = row["label"] if row["gamma"] == 0 else "RSI (second-order)"
        if row["g"] > 0 and row["gamma"] == 0:
            lab = f"{row['label']} (g={row['g']})"
        ax.plot(range(len(row["human_share_t"])), row["human_share_t"],
                ls, c=c, label=lab)
    ax.axvline(results["headline"]["first_arrival"], ls=":", c="black", lw=1)
    ax.set_xlabel("time"); ax.set_ylabel("human share of income")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "growth.pdf"))
    print("[saved] growth.pdf")


def fig_funds(results):
    """Appendix — the two public-fund designs (transfer vs ratchet)."""
    rows = results["E3"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for mir, c, lab in [(False, "tab:green", "dividend fund"),
                        (True, "tab:purple", "mirror fund (ratchet)")]:
        sel = [r for r in rows if r["ownership"] == 0.4 and r["pub_mirror"] == mir]
        ax.errorbar([r["tax_rate"] for r in sel],
                    [r["pub_share"]["point"] for r in sel], marker="o", color=c,
                    label=lab)
    ax.axhline(0.4, ls=":", c="gray", lw=1)
    ax.set_xlabel("profit tax rate $\\tau$")
    ax.set_ylabel("public share of capital")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "funds.pdf"))
    print("[saved] funds.pdf")


def main():
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(HERE, "results.json")) as fh:
        results = json.load(fh)
    fig_headline(results)
    fig_money(results)
    fig_endstate(results)
    fig_growth(results)
    fig_knee(results)
    fig_funds(results)


if __name__ == "__main__":
    main()
