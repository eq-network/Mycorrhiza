"""WP3 figures from results.json — read, plot, never recompute.

    python -m experiments.wp3_politics.figures

Writes into the WP3 paper's figures/ dir. The paper lives in the Obsidian vault,
not this repo (writing stays out of the codebase); this path is the one seam.

Style: WP1's compact-matplotlib house style; colors are the Okabe-Ito CVD-safe
set with FIXED assignment (organic #0072B2 blue / captured #D55E00 vermillion /
locked #000000 black; churn curves blue, orange, green, vermillion in fixed
order); heatmaps are single-hue sequential; never a dual axis.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIGS = os.path.join(
    os.path.expanduser("~"), "Documents", "Productivity", "Obsidian",
    "Research", "Projects", "CI Library", "papers", "wp3-politics", "figures")

COND = {"organic": "#0072B2", "captured": "#D55E00", "locked": "#000000"}
CHURN = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]        # fixed order


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, name))
    plt.close(fig)
    print(f"[saved] {name}")


def fig_headline(results):
    """Paper Fig. 1 — the political GD signature: after AI delegates arrive,
    human power share bends one way and policy detaches from the median."""
    h = results["headline"]
    t = np.arange(len(h["captured"]["policy"]))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.6, 4.6), sharex=True)

    for name in ("organic", "captured", "locked"):
        ax1.plot(t, h[name]["policy"], c=COND[name], lw=1.4, label=name,
                 ls="--" if name == "locked" else "-")
    p, sd = np.array(h["captured"]["policy"]), np.array(h["captured"]["policy_sd"])
    ax1.fill_between(t, p - sd, p + sd, color=COND["captured"], alpha=0.15, lw=0)
    ax1.axhline(h["citizen_median"], ls="--", c="gray", lw=1, label="citizen median")
    ax1.axvline(h["arrival"], ls=":", c="black", lw=1)
    ax1.set_ylabel("enacted tax rate")
    ax1.legend(fontsize=7, loc="center right", frameon=False)

    for name in ("organic", "captured", "locked"):
        ax2.plot(t, h[name]["human_power_share"], c=COND[name], lw=1.4,
                 ls="--" if name == "locked" else "-")
    s = np.array(h["captured"]["human_power_share"])
    sd = np.array(h["captured"]["human_power_share_sd"])
    ax2.fill_between(t, s - sd, s + sd, color=COND["captured"], alpha=0.15, lw=0)
    ax2.axvline(h["arrival"], ls=":", c="black", lw=1)
    ax2.annotate("AI delegates arrive", (h["arrival"] + 8, 0.45), fontsize=7)
    ax2.set_ylabel("human power share")
    ax2.set_xlabel("tick")
    ax2.set_ylim(0, 1.02)
    _save(fig, "headline.pdf")


def fig_e1_knee(results):
    """Human power share vs advantage, one curve per churn, with each curve's
    pre-registered a* as a dotted vertical in the matching color."""
    pred = {row["churn"]: row["a_star"] for row in results["a_star_pred"]["curve"]}
    churns = sorted({row["churn"] for row in results["E1"]})
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for i, r in enumerate(churns):
        rows = sorted([q for q in results["E1"] if q["churn"] == r],
                      key=lambda q: q["advantage"])
        a = [q["advantage"] for q in rows]
        y = [q["human_power_share"]["point"] for q in rows]
        lo = [q["human_power_share"]["lo"] for q in rows]
        hi = [q["human_power_share"]["hi"] for q in rows]
        ax.fill_between(a, lo, hi, color=CHURN[i], alpha=0.15, lw=0)
        ax.plot(a, y, "o-", c=CHURN[i], ms=3.5, lw=1.4, label=f"churn = {r}")
        if np.isfinite(pred[r]) and pred[r] <= max(a) + 0.3:
            ax.axvline(pred[r], ls=":", c=CHURN[i], lw=1)
    ax.set_xlim(0.7, 8.3)      # a*(0.1) = 20, a*(0.2) = inf are off-scale by design
    ax.set_xlabel("AI advantage a")
    ax.set_ylabel("human power share (late)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("dotted: pre-registered a*(churn)", fontsize=8)
    _save(fig, "e1_knee.pdf")


def fig_e1_phase(results):
    """(advantage, churn) plane: policy-median gap heatmap + the committed a*
    curve — tracking democracy left of the line, takeover right of it."""
    churns = sorted({row["churn"] for row in results["E1"]})
    advs = sorted({row["advantage"] for row in results["E1"]})
    gap = np.zeros((len(churns), len(advs)))
    for row in results["E1"]:
        gap[churns.index(row["churn"]), advs.index(row["advantage"])] = \
            row["policy_median_gap"]["point"]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    im = ax.pcolormesh(np.arange(len(advs) + 1), np.arange(len(churns) + 1),
                       gap, cmap="Oranges", vmin=0)
    for i in range(len(churns)):
        for j in range(len(advs)):
            dark = gap[i, j] > 0.6 * gap.max()
            ax.text(j + 0.5, i + 0.5, f"{gap[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if dark else "#444444")
    pred = {row["churn"]: row["a_star"] for row in results["a_star_pred"]["curve"]}
    xs, ys = [], []
    for i, r in enumerate(churns):
        if np.isfinite(pred[r]) and min(advs) <= pred[r] <= max(advs):
            j = np.interp(pred[r], advs, np.arange(len(advs)) + 0.5)
            xs.append(j)
            ys.append(i + 0.5)
    ax.plot(xs, ys, "k--o", ms=4, lw=1.2, label="pre-registered a*")
    ax.set_xticks(np.arange(len(advs)) + 0.5, [str(a) for a in advs])
    ax.set_yticks(np.arange(len(churns)) + 0.5, [str(r) for r in churns])
    ax.set_xlabel("AI advantage a")
    ax.set_ylabel("churn r")
    ax.legend(fontsize=7, loc="upper left", frameon=True)
    fig.colorbar(im, label="|policy − median| (late)")
    _save(fig, "e1_phase.pdf")


def fig_e2_hysteresis(results):
    """Left pair: human-power-share trajectories after the advantage shut-off,
    one line per lock-in level, churn-poor vs churn-rich panel. Right:
    collective responsiveness of the effective tax rate (from birth vs now)
    per condition — the GD gap as bars."""
    churns = sorted({row["churn"] for row in results["E2_shutoff"]})
    fig, axes = plt.subplots(1, len(churns) + 1, figsize=(10.5, 3.2))
    traj_axes, ax2 = axes[:-1], axes[-1]

    for k, (ax1, r) in enumerate(zip(traj_axes, churns)):
        rows = [q for q in results["E2_shutoff"] if q["churn"] == r]
        shades = plt.get_cmap("Blues")(np.linspace(0.45, 0.95, len(rows)))
        recovered = all(q["recovery_gap"]["point"] < 0.05 for q in rows)
        for row, c in zip(rows, shades):
            s = row["share_series"]
            ax1.plot(np.arange(len(s)), s, c=c, lw=1.4)
            if not recovered:      # converging lines get one note, not four labels
                ax1.annotate(f"lock-in {row['lockin']}", (len(s) - 1, s[-1]),
                             fontsize=6, color=c, ha="right",
                             xytext=(0, 4), textcoords="offset points")
        if recovered:
            ax1.annotate("all lock-in levels recover", (len(s) * 0.55, 0.62),
                         fontsize=7, color="#444444")
        ref = results["E2_organic_ref"][str(r)]["human_power_share"]
        ax1.axhspan(ref["lo"], ref["hi"], color="gray", alpha=0.2, lw=0)
        ax1.axhline(ref["point"], ls="--", c="gray", lw=1, label="organic reference")
        ax1.axvline(50, ls=":", c="black", lw=1)
        ax1.axvline(rows[0]["shutoff_t"], ls=":", c="black", lw=1)
        ax1.annotate("on", (54, 0.12), fontsize=7)
        ax1.annotate("off", (rows[0]["shutoff_t"] + 4, 0.12), fontsize=7)
        ax1.set_xlabel("tick")
        ax1.set_ylim(0, 1.02)
        ax1.set_title(f"churn = {r}", fontsize=9)
    traj_axes[0].set_ylabel("human power share")
    traj_axes[0].legend(fontsize=7, loc="lower right", frameon=False)

    conds = [r["condition"] for r in results["E2_responsiveness"]]
    x = np.arange(len(conds))
    for k, (key, label, c, hatch) in enumerate([
            ("influence_birth", "from birth", "#0072B2", None),
            ("influence_now", "now", "#56B4E9", "//")]):
        pts = [r[key]["point"] for r in results["E2_responsiveness"]]
        err = np.array([[r[key]["point"] - r[key]["lo"] for r in results["E2_responsiveness"]],
                        [r[key]["hi"] - r[key]["point"] for r in results["E2_responsiveness"]]])
        ax2.bar(x + 0.2 * (2 * k - 1), pts, width=0.36, color=c, hatch=hatch,
                label=label, yerr=err, capsize=2,
                error_kw={"lw": 0.8}, edgecolor="white", linewidth=1)
    ax2.set_xticks(x, conds)
    ax2.axhline(0, c="black", lw=0.8)
    ax2.set_ylabel("d(effective rate) / d(citizen ideals)")
    ax2.legend(fontsize=7, frameon=False)
    _save(fig, "e2_hysteresis.pdf")


def fig_e3_defenses(results):
    """Sortition cadence x influence cap heatmaps of human power share, early
    vs late defense onset — the reversibility window made visible."""
    onsets = sorted({row["onset"] for row in results["E3"]})
    cadences = list(dict.fromkeys(row["cadence"] for row in results["E3"]))
    caps = list(dict.fromkeys(row["cap_share"] for row in results["E3"]))

    fig, axes = plt.subplots(1, len(onsets), figsize=(7.4, 3.0), sharey=True,
                             constrained_layout=True)
    for ax, onset in zip(axes, onsets):
        M = np.zeros((len(cadences), len(caps)))
        for row in results["E3"]:
            if row["onset"] == onset:
                M[cadences.index(row["cadence"]), caps.index(row["cap_share"])] = \
                    row["human_power_share"]["point"]
        im = ax.pcolormesh(np.arange(len(caps) + 1), np.arange(len(cadences) + 1),
                           M, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(cadences)):
            for j in range(len(caps)):
                ax.text(j + 0.5, i + 0.5, f"{M[i, j]:.2f}", ha="center",
                        va="center", fontsize=7,
                        color="white" if M[i, j] > 0.6 else "#444444")
        ax.set_xticks(np.arange(len(caps)) + 0.5,
                      ["off" if c == 0 else str(c) for c in caps])
        ax.set_xlabel("influence cap share")
        ax.set_title(f"defense onset t = {onset}", fontsize=9)
    axes[0].set_yticks(np.arange(len(cadences)) + 0.5,
                       ["off" if c == 0 else str(c) for c in cadences])
    axes[0].set_ylabel("sortition cadence")
    fig.colorbar(im, ax=axes, label="human power share", fraction=0.04)
    fig.savefig(os.path.join(FIGS, "e3_defenses.pdf"))
    plt.close(fig)
    print("[saved] e3_defenses.pdf")


def fig_e4_floors(results):
    """The power floors, removed cumulatively. Left: the human share
    decomposed into its three sources per condition (stacked bars). Right:
    the share trajectories — the last condition reaches zero."""
    rows = results["E4_floors"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.1))

    comps = [("floor_kept", "franchise (kept votes)", "#0072B2"),
             ("floor_handback", "AI ballots handed back", "#56B4E9"),
             ("floor_c2c", "citizen-to-citizen", "#009E73")]
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for keyname, label, c in comps:
        vals = np.array([r[keyname] for r in rows])
        ax1.barh(y, vals, left=left, height=0.55, color=c, label=label,
                 edgecolor="white", linewidth=1)
        left += vals
    for yi, r in zip(y, rows):
        total = r["floor_kept"] + r["floor_handback"] + r["floor_c2c"]
        ax1.text(max(total, 0.001) + 0.015, yi,
                 f"{r['human_power_share']['point']:.2f}",
                 va="center", fontsize=8, color="#444444")
    ax1.set_yticks(y, [r["condition"] for r in rows])
    ax1.set_xlabel("human power share (end state, by source)")
    ax1.set_xlim(0, 0.55)
    ax1.legend(fontsize=7, frameon=False, loc="lower right")

    shades = plt.get_cmap("Oranges")(np.linspace(0.4, 0.95, len(rows)))
    for r, c in zip(rows, shades):
        s = r["share_series"]
        ax2.plot(np.arange(len(s)), s, c=c, lw=1.5, label=r["condition"])
    ax2.axvline(50, ls=":", c="black", lw=1)
    ax2.annotate("AI delegates arrive", (58, 0.93), fontsize=7)
    ax2.set_xlabel("tick")
    ax2.set_ylabel("human power share")
    ax2.set_ylim(-0.02, 1.0)
    ax2.legend(fontsize=7, frameon=False)
    _save(fig, "e4_floors.pdf")


def fig_e5_prominence(results):
    """E5a/E5c: the structural assumptions, across their ranges. Left/center:
    human power share and top-actor share vs gamma (does prominence compound?),
    organic and captured. Right: share vs franchise size with the kept-vote
    floor line."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 3.0))

    for cond, color in (("organic", COND["organic"]), ("captured", COND["captured"])):
        rows = sorted([r for r in results["E5_gamma"] if r["condition"] == cond],
                      key=lambda r: r["gamma"])
        g = [r["gamma"] for r in rows]
        for ax, metric in ((ax1, "human_power_share"), (ax2, "top_delegate_share")):
            y = [r[metric]["point"] for r in rows]
            lo = [r[metric]["lo"] for r in rows]
            hi = [r[metric]["hi"] for r in rows]
            ax.fill_between(g, lo, hi, color=color, alpha=0.15, lw=0)
            ax.plot(g, y, "o-", c=color, ms=3.5, lw=1.4,
                    label=cond if ax is ax1 else None)
    for ax, ylab in ((ax1, "human power share"), (ax2, "top actor's share")):
        ax.axvline(1.0, ls="--", c="gray", lw=1)
        ax.axvline(1.3, ls=":", c="black", lw=1)
        ax.set_xlabel("prominence exponent $\\gamma$")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.02)
    ax1.annotate("share-neutral", (0.99, 0.06), fontsize=6.5, color="gray", ha="right")
    ax1.annotate("default", (1.31, 0.06), fontsize=6.5)
    ax1.legend(fontsize=7, frameon=False, loc="center left")

    rows = sorted(results["E5_franchise"], key=lambda r: r["self_weight"])
    sw = [r["self_weight"] for r in rows]
    y = [r["human_power_share"]["point"] for r in rows]
    lo = [r["human_power_share"]["lo"] for r in rows]
    hi = [r["human_power_share"]["hi"] for r in rows]
    ax3.fill_between(sw, lo, hi, color=COND["captured"], alpha=0.15, lw=0)
    ax3.plot(sw, y, "o-", c=COND["captured"], ms=3.5, lw=1.4, label="captured")
    ax3.plot(sw, [r["kept_floor_pred"] for r in rows], "--", c="gray", lw=1,
             label="kept-vote floor $n_c s_w / N$")
    ax3.set_xlabel("franchise size $s_w$")
    ax3.set_ylabel("human power share")
    ax3.set_ylim(0, 1.02)
    ax3.legend(fontsize=7, frameon=False)
    _save(fig, "e5_prominence.pdf")


def fig_e5_objective(results):
    """E5b: the AI-objective plane (fidelity x bias) under capture — the policy
    gap follows the committed prediction (1-alpha)|b - median|; the power share
    is invariant across the whole plane (power never reads positions)."""
    alphas = sorted({r["alignment_ai"] for r in results["E5_objective"]})
    biases = sorted({r["ai_bias"] for r in results["E5_objective"]})
    gap = np.zeros((len(alphas), len(biases)))
    for r in results["E5_objective"]:
        gap[alphas.index(r["alignment_ai"]), biases.index(r["ai_bias"])] = \
            r["policy_median_gap"]["point"]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    im = ax.pcolormesh(np.arange(len(biases) + 1), np.arange(len(alphas) + 1),
                       gap, cmap="Oranges", vmin=0)
    for i in range(len(alphas)):
        for j in range(len(biases)):
            dark = gap[i, j] > 0.6 * max(gap.max(), 1e-9)
            ax.text(j + 0.5, i + 0.5, f"{gap[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if dark else "#444444")
    j_aligned = biases.index(0.4) if 0.4 in biases else None
    if j_aligned is not None:
        ax.plot([j_aligned, j_aligned + 1, j_aligned + 1, j_aligned, j_aligned],
                [0, 0, len(alphas), len(alphas), 0], c="#0072B2", lw=1.6)
        ax.annotate("aligned:\ncapture, no gap", (j_aligned + 0.5, len(alphas) + 0.15),
                    ha="center", fontsize=7, color="#0072B2")
    ax.set_xticks(np.arange(len(biases)) + 0.5, [str(b) for b in biases])
    ax.set_yticks(np.arange(len(alphas)) + 0.5, [str(a) for a in alphas])
    ax.set_xlabel("AI bias $b$ (its own pull)")
    ax.set_ylabel("AI fidelity $\\alpha$")
    inv = results["E5_share_invariance"]
    ax.set_title(f"power share invariant across the plane: "
                 f"[{inv['min']:.3f}, {inv['max']:.3f}]", fontsize=8)
    fig.colorbar(im, label="|policy − median| (late)")
    _save(fig, "e5_objective.pdf")


def fig_network_triptych(results):
    """The delegation graph at three moments: organic -> concentrating ->
    captured. Node size = power; citizens blue, AI delegates vermillion."""
    snap = results["snapshots"]
    types = np.array(snap["node_types"])
    n = len(types)
    n_c = int((types == 0).sum())
    theta = 2 * np.pi * np.arange(n_c) / n_c
    pos = np.zeros((n, 2))
    pos[:n_c, 0], pos[:n_c, 1] = np.cos(theta), np.sin(theta)
    n_ai = n - n_c
    phi = 2 * np.pi * np.arange(n_ai) / max(n_ai, 1) + np.pi / 4
    pos[n_c:, 0], pos[n_c:, 1] = 0.3 * np.cos(phi), 0.3 * np.sin(phi)

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    for ax, t, W, v in zip(axes, snap["t"], snap["delegation"], snap["influence"]):
        W, v = np.array(W), np.array(v)
        for i in range(n):
            for j in range(n):
                if i != j and W[i, j] > 0.05:
                    ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                            c="gray", lw=6 * W[i, j], alpha=0.25, zorder=1)
        for typ, color in ((0, "#0072B2"), (1, "#D55E00")):
            m = types == typ
            ax.scatter(pos[m, 0], pos[m, 1], s=8 + 3000 * v[m], c=color,
                       edgecolors="white", linewidths=0.5, zorder=2)
        share = float(v[types == 0].sum() / v.sum())
        ax.set_title(f"t = {t}   human share {share:.2f}", fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")
    _save(fig, "network_triptych.pdf")


def main():
    with open(os.path.join(HERE, "results.json")) as f:
        results = json.load(f)
    os.makedirs(FIGS, exist_ok=True)
    fig_headline(results)
    fig_e1_knee(results)
    fig_e1_phase(results)
    fig_e2_hysteresis(results)
    fig_e3_defenses(results)
    fig_e4_floors(results)
    fig_e5_prominence(results)
    fig_e5_objective(results)
    fig_network_triptych(results)


if __name__ == "__main__":
    main()
