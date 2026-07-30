"""WP2 figures from results.json + snapshots.npz — read, plot, never re-simulate.

    python -m experiments.wp2_culture.figures

Writes into the WP2 paper's figures/ dir. The paper lives in the Obsidian
vault, not this repo (writing stays out of the codebase); this path is the
one seam. The eigenmode search in fig3 is presentation-layer math on the
saved W snapshots — "never recompute" means never re-run the simulation; the
npz is the raw record.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .instruments import anchor_floor, mode_alignments, spring_layout

HERE = os.path.dirname(__file__)
FIGS = os.path.join(
    os.path.expanduser("~"), "Documents", "Productivity", "Obsidian",
    "Research", "Projects", "CI Library", "papers", "wp2-culture", "figures")


def load():
    with open(os.path.join(HERE, "results.json")) as fh:
        results = json.load(fh)
    snap = np.load(os.path.join(HERE, "snapshots.npz"))
    return results, snap["W"].astype(np.float64), snap["times"]


def draw_edges(ax, W, pos, top=None, color="0.5", max_alpha=0.9):
    """Directed weights drawn as lines, alpha proportional to attention weight."""
    n = W.shape[0]
    off = W.copy()
    np.fill_diagonal(off, 0.0)
    i, j = np.nonzero(off > 1e-3)
    w = off[i, j]
    if top is not None and len(w) > top:
        keep = np.argsort(w)[::-1][:top]
        i, j, w = i[keep], j[keep], w[keep]
    segs = np.stack([pos[i], pos[j]], axis=1)
    alphas = np.clip(w / w.max() * max_alpha, 0.05, max_alpha)
    lc = LineCollection(segs, colors=[color], linewidths=0.6, alpha=None, zorder=1)
    lc.set_alpha(None)
    rgba = np.zeros((len(w), 4))
    rgba[:, :3] = matplotlib.colors.to_rgb(color)
    rgba[:, 3] = alphas
    lc.set_color(rgba)
    ax.add_collection(lc)


def draw_nodes(ax, pos, n_c, citizen_colors="tab:blue", ai_colors="tab:red",
               size=55, **kw):
    """Citizens = circles, AI voices = squares (the shape IS the legend)."""
    sc = ax.scatter(pos[:n_c, 0], pos[:n_c, 1], c=citizen_colors, s=size,
                    marker="o", edgecolors="white", linewidths=0.5, zorder=2, **kw)
    ax.scatter(pos[n_c:, 0], pos[n_c:, 1], c=ai_colors, s=size * 1.6,
               marker="s", edgecolors="black", linewidths=0.7, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    return sc


def fig_graph(results, W, pos):
    """Fig 1 — the town on day 0: who listens to whom."""
    n_c = results["config"]["n_citizens"]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    draw_edges(ax, W[0], pos)
    draw_nodes(ax, pos, n_c)
    ax.scatter([], [], c="tab:blue", marker="o", label="person")
    ax.scatter([], [], c="tab:red", marker="s", label="AI voice")
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "graph.pdf"))
    print("[saved] graph.pdf")


def fig_field(results, W, pos):
    """Fig 2 — the field filling up: how machine-filled each head is, day 0 vs T."""
    cfgr = results["config"]
    n_c, lam = cfgr["n_citizens"], cfgr["susceptibility"]
    panels = [("day 0", W[0], np.array(results["exemplar"]["ai_share_t0"])),
              (f"day {cfgr['T']}", W[-1], np.array(results["exemplar"]["ai_share_tT"]))]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, (title, w, shares) in zip(axes, panels):
        # person->AI attention flows: the channels amplification carves
        wf = w.copy()
        mask = np.zeros_like(wf, dtype=bool)
        mask[:n_c, n_c:] = True
        wf[~mask] = 0.0
        draw_edges(ax, wf, pos, top=40, color="tab:red", max_alpha=0.7)
        sc = draw_nodes(ax, pos, n_c, citizen_colors=shares, size=48,
                        cmap="viridis", vmin=0.0, vmax=lam)
        ax.set_title(title, fontsize=9)
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("share of percept supplied by AI", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(os.path.join(FIGS, "field.pdf"), bbox_inches="tight")
    print("[saved] field.pdf")


def fig_modes(results, W, times, pos):
    """Fig 3 — mode alignment: the human-AI split climbing the mode ladder."""
    cfgr = results["config"]
    n_c = cfgr["n_citizens"]
    n = n_c + cfgr["n_ai"]
    types = np.arange(n) >= n_c

    ranks, phis_best, vec_best = [], [], []
    for f in range(W.shape[0]):
        _, vecs, phis = mode_alignments(W[f], types)
        k = int(np.argmax(phis)) + 1
        ranks.append(k)
        phis_best.append(float(phis[k - 1]))
        vec_best.append(vecs[:, k])

    fig = plt.figure(figsize=(7.4, 3.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 1.0], hspace=0.35)
    for p, t in enumerate(results["exemplar"]["checkpoints"]):
        f = int(np.argmin(np.abs(times - t)))
        ax = fig.add_subplot(gs[0, p])
        side = vec_best[f] >= 0
        # orient every panel the same way: AI voices on the orange side
        if side[n_c:].mean() < 0.5:
            side = ~side
        colors = np.where(side, "tab:orange", "tab:blue")
        draw_edges(ax, W[f], pos, top=120)
        draw_nodes(ax, pos, n_c, citizen_colors=list(colors[:n_c]),
                   ai_colors=list(colors[n_c:]), size=28)
        ax.set_title(f"day {times[f]}\nmode {ranks[f]}/{n - 1} · "
                     f"match {phis_best[f]:.2f}", fontsize=7)

    axb = fig.add_subplot(gs[1, :])
    axb.step(times, ranks, where="post", c="tab:blue", label="mode rank of the human-AI split")
    axb.axvline(results["config"]["amp_onset"], ls=":", c="black", lw=1)
    axb.set_ylim(max(ranks) + 1.5, 0.5)  # inverted: climbing toward mode 1 reads upward
    axb.set_xlabel("day", fontsize=8)
    axb.set_ylabel("mode rank", fontsize=8)
    axr = axb.twinx()
    axr.plot(times, phis_best, c="tab:green", lw=1.2, label="match with the split")
    axr.set_ylim(0, 1.05)
    axr.set_ylabel("match", fontsize=8)
    l1, lb1 = axb.get_legend_handles_labels()
    l2, lb2 = axr.get_legend_handles_labels()
    axb.legend(l1 + l2, lb1 + lb2, fontsize=7, loc="center right")
    axb.tick_params(labelsize=7)
    axr.tick_params(labelsize=7)
    fig.savefig(os.path.join(FIGS, "modes.pdf"), bbox_inches="tight")
    print(f"[saved] modes.pdf (final: mode {ranks[-1]}, match {phis_best[-1]:.2f})")


def fig_power(results):
    """Fig 4 — the quantitative claim: amplification drives the human share down."""
    cfgr = results["config"]
    h = results["headline"]
    t = h["time"]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for name, color, label in [("amp_off", "tab:blue", "amplification off"),
                               ("amp_on", "tab:red", "amplification on")]:
        ax.fill_between(t, h[name]["lo"], h[name]["hi"], alpha=0.2, color=color)
        ax.plot(t, h[name]["mean"], c=color, label=label)
    ax.axhline(cfgr["floor"], ls="--", c="gray", lw=1,
               label=r"anchor floor $1-\lambda$")
    ax.axvline(cfgr["amp_onset"], ls=":", c="black", lw=1)
    ax.annotate("amplification begins", (cfgr["amp_onset"] + 6, 0.06), fontsize=8)
    ax.set_xlabel("day")
    ax.set_ylabel("human share of percepts")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "dynamics.pdf"))
    print("[saved] dynamics.pdf")


def fig_dials(results):
    """Fig 5 — the non-anchor dials across their ranges; defaults are one point."""
    d = results["dial_sweeps"]
    cfgr = results["config"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))

    def panel(ax, rows, default, xlabel):
        xs = [r["value"] for r in rows]
        ax.errorbar(xs, [r["share"]["point"] for r in rows],
                    yerr=[[r["share"]["point"] - r["share"]["lo"] for r in rows],
                          [r["share"]["hi"] - r["share"]["point"] for r in rows]],
                    marker="o", c="tab:blue")
        ax.axvline(default, ls=":", c="black", lw=1)
        ax.annotate("value in figs 1-4", (default, 0.92), fontsize=7,
                    rotation=90, va="top", ha="right")
        ax.axhline(cfgr["floor"], ls="--", c="gray", lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.0)

    panel(ax1, d["amplification"], d["defaults"]["amplification"],
          "amplification")
    ax1.set_xscale("log", base=2)
    ax1.set_ylabel("human share of percepts, final day")
    panel(ax2, d["drift"], d["defaults"]["drift"], "attention step $u$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "dials.pdf"))
    print("[saved] dials.pdf")


def fig_floor(results):
    """Fig 6 — remove the anchors and the share follows the analytic floor to zero."""
    rows = results["floor_sweep"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    styles = {0.15: ("tab:blue", "o"), 0.0: ("tab:red", "s")}
    lo_lam = min(r["lam"] for r in rows)
    dense = np.linspace(lo_lam, 1.0, 200)
    for s_w in sorted({r["self_weight"] for r in rows}, reverse=True):
        sel = sorted([r for r in rows if r["self_weight"] == s_w],
                     key=lambda r: r["lam"])
        c, m = styles.get(s_w, ("tab:green", "^"))
        xs = [r["lam"] for r in sel]
        ax.errorbar(xs, [r["share"]["point"] for r in sel],
                    yerr=[[r["share"]["point"] - r["share"]["lo"] for r in sel],
                          [r["share"]["hi"] - r["share"]["point"] for r in sel]],
                    marker=m, c=c, label=f"measured, self-attention {s_w}")
        ax.plot(dense, [anchor_floor(l, s_w) for l in dense], ls="--", c=c,
                lw=1, label=f"floor, self-attention {s_w}")
    ax.set_xlabel(r"susceptibility $\lambda$")
    ax.set_ylabel("human share of percepts, final day")
    ax.set_ylim(0, 0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "floor.pdf"))
    print("[saved] floor.pdf")


def main():
    os.makedirs(FIGS, exist_ok=True)
    results, W, times = load()
    n_c = results["config"]["n_citizens"]
    pos = spring_layout(W[0], seed=0)  # ONE layout, shared by figs 1-3
    fig_graph(results, W, pos)
    fig_field(results, W, pos)
    fig_modes(results, W, times, pos)
    fig_power(results)
    fig_dials(results)
    fig_floor(results)


if __name__ == "__main__":
    main()

