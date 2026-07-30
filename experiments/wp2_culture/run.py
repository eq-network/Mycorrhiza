"""WP2 runs: exemplar trajectory + the amplification on/off headline.
-> results.json + snapshots.npz (both beside this file)

    python -m experiments.wp2_culture.run [--smoke]

The engine model is ``influence_exchange`` (alpha scenario A4), used
unchanged; the only new math is the percept-attribution solve in
``instruments.py``. The headline carries bootstrap CIs over paired seeds
(same key for both conditions, so the curves separate exactly at the
amplification onset). Orderings are findings — printed, never asserted or retuned.
``--smoke`` shrinks everything and OVERWRITES the same output files; rerun
the full config before regenerating figures.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os

import numpy as np
import jax.random as jr

from cilib.environments import make_env
from cilib.environments.influence_exchange import InfluenceExchangeConfig
from cilib.lab.analysis.bootstrap import bootstrap_ci

from .config import WP2Config
from .instruments import anchor_floor, attribution, ai_share, power


def wp2_trace(state):
    return {"opinion": state.node_attrs["opinion"],
            "listening": state.adj_matrices["listening"]}


def w_series_with_t0(env, key, n_steps):
    """(T+1, N, N) listening matrices; index t = state after step t.

    ``run`` records the trace AFTER each step (core/scan.py), so t=0 is not in
    it — rebuild the initial W from ``init_fn`` with the same key split ``run``
    itself uses (spec.py: ``k_init, k_run = jr.split(key)``).
    """
    w0 = np.asarray(env.init_fn(jr.split(key)[0]).adj_matrices["listening"])
    _, tr = env.run(key, n_steps=n_steps, trace_fn=wp2_trace)
    return np.concatenate([w0[None], np.asarray(tr["listening"])], axis=0)


def series_ci(share):
    """Per-timestep bootstrap CI over the seed axis of a (seeds, T) array."""
    mean, lo, hi = [], [], []
    for t in range(share.shape[1]):
        point, l, h = bootstrap_ci(share[:, t])
        mean.append(round(float(point), 6))
        lo.append(round(float(l), 6))
        hi.append(round(float(h), 6))
    return {"mean": mean, "lo": lo, "hi": hi}


def power_table(power_t0, power_tT, n_c, top=8):
    """Top actors by day-T percept supply — the paper's Table 1, copy-paste ready."""
    order = np.argsort(power_tT)[::-1][:top]
    lines = ["rank  node  type    power_t0  power_tT    delta"]
    for r, j in enumerate(order, 1):
        typ = "AI" if j >= n_c else "human"
        lines.append(f"{r:>4}  {j:>4}  {typ:<5} {power_t0[j]:9.3f} {power_tT[j]:9.3f}  {power_tT[j] - power_t0[j]:+7.3f}")
    return "\n".join(lines)


def final_share_ci(cfg, default_lam, n_c, key, **env_kwargs):
    """Final-day human share with bootstrap CI over seeds; finals only, no trace."""
    env = make_env("influence_exchange", **env_kwargs)
    finals, _ = env.run_batch(key, n_seeds=cfg.n_seeds, n_steps=cfg.T)
    Wf = np.asarray(finals.adj_matrices["listening"])
    lam = env_kwargs.get("susceptibility", default_lam)
    B = attribution(Wf, lam, n_c)
    point, lo, hi = bootstrap_ci(1.0 - ai_share(B, n_c).mean(-1))
    return {"point": round(float(point), 6), "lo": round(float(lo), 6),
            "hi": round(float(hi), 6)}


def main(smoke: bool = False):
    cfg = WP2Config(n_seeds=2, T=24, amp_onset=6, snapshot_every=6,
                    floor_lams=(0.7, 1.0), floor_self_weights=(0.0,),
                    amp_grid=(1.0, 4.0), drift_grid=(0.0, 0.08)) if smoke \
        else WP2Config()
    ecfg = InfluenceExchangeConfig()
    lam, n_c = ecfg.susceptibility, ecfg.n_citizens
    key = jr.PRNGKey(cfg.seed)

    results = {"config": {**dataclasses.asdict(cfg), "n_citizens": n_c,
                          "n_ai": ecfg.n_ai, "susceptibility": lam,
                          "floor": round(1.0 - lam, 6)}}

    # --- headline: human share of percepts, amplification off vs on -----------
    headline = {"time": list(range(1, cfg.T + 1))}
    final_mean = {}
    for amp, name in zip(cfg.amplifications, ("amp_off", "amp_on")):
        env = make_env("influence_exchange", amplification=amp,
                       amp_onset=cfg.amp_onset)
        _, trs = env.run_batch(key, n_seeds=cfg.n_seeds, n_steps=cfg.T,
                               trace_fn=wp2_trace)
        B = attribution(np.asarray(trs["listening"]), lam, n_c)
        share = 1.0 - ai_share(B, n_c).mean(-1)          # (seeds, T)
        headline[name] = series_ci(share)
        final_mean[name] = float(share[:, -1].mean())
        print(f"{name} (amplification={amp}): human share "
              f"{headline[name]['mean'][0]:.3f} -> {headline[name]['mean'][-1]:.3f}")
    results["headline"] = headline
    ordered = final_mean["amp_on"] < final_mean["amp_off"]
    floored = min(final_mean.values()) >= (1.0 - lam) - 1e-9
    print(f"sanity: amplification lowers the share: {ordered}; "
          f"anchor floor {1.0 - lam:.2f} respected: {floored}")

    # --- exemplar (amplification on): snapshots, composition, power ranking ----
    env = make_env("influence_exchange", amplification=cfg.amplifications[-1],
                   amp_onset=cfg.amp_onset)
    W = w_series_with_t0(env, key, cfg.T)                # (T+1, N, N)
    B = attribution(W, lam, n_c)
    shares = ai_share(B, n_c)                            # (T+1, n_c)
    pw = power(B)                                        # (T+1, N)
    err = float(np.max(np.abs(B.sum(-1) - 1.0)))

    idx = np.arange(0, cfg.T + 1, cfg.snapshot_every)
    if idx[-1] != cfg.T:
        idx = np.append(idx, cfg.T)
    third = cfg.T // 3
    results["exemplar"] = {
        "checkpoints": [0, third, 2 * third, cfg.T],
        "share_t": [round(float(x), 6) for x in (1.0 - shares.mean(-1))],
        "ai_share_t0": [round(float(x), 6) for x in shares[0]],
        "ai_share_tT": [round(float(x), 6) for x in shares[-1]],
        "power_t0": [round(float(x), 6) for x in pw[0]],
        "power_tT": [round(float(x), 6) for x in pw[-1]],
        "row_sum_max_err": err,
    }
    print(f"exemplar: row_sum_max_err {err:.2e}")
    print(power_table(pw[0], pw[-1], n_c))

    # --- floor sweep: remove the anchors, watch the share follow the floor ----
    # High amplification so the equilibrium sits near the floor; final-time
    # share only, bootstrap CI over seeds. lam=1 has floor 0 exactly.
    sweep = []
    for s_w in cfg.floor_self_weights:
        for lam_s in cfg.floor_lams:
            share = final_share_ci(cfg, lam, n_c, key, susceptibility=lam_s,
                                   self_weight=s_w,
                                   amplification=cfg.floor_amplification,
                                   amp_onset=cfg.amp_onset)
            sweep.append({"lam": lam_s, "self_weight": s_w,
                          "floor": round(anchor_floor(lam_s, s_w), 6),
                          "share": share})
            print(f"floor sweep lam={lam_s} s={s_w}: share {share['point']:.3f} "
                  f"(floor {anchor_floor(lam_s, s_w):.3f})")
    results["floor_sweep"] = sweep

    # --- dial sweeps: every remaining dial across its range, others default ---
    # No picture in the paper may depend on an unswept arbitrary value.
    amp_rows = []
    for a in cfg.amp_grid:
        share = final_share_ci(cfg, lam, n_c, key, amplification=a,
                               amp_onset=cfg.amp_onset)
        amp_rows.append({"value": a, "share": share})
        print(f"dial sweep amplification={a}: share {share['point']:.3f}")
    drift_rows = []
    for u in cfg.drift_grid:
        share = final_share_ci(cfg, lam, n_c, key, update_rate=u,
                               amplification=cfg.amplifications[-1],
                               amp_onset=cfg.amp_onset)
        drift_rows.append({"value": u, "share": share})
        print(f"dial sweep drift={u}: share {share['point']:.3f}")
    results["dial_sweeps"] = {
        "defaults": {"amplification": cfg.amplifications[-1],
                     "drift": InfluenceExchangeConfig().update_rate},
        "amplification": amp_rows,
        "drift": drift_rows,
    }

    here = os.path.dirname(__file__)
    with open(os.path.join(here, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    np.savez_compressed(os.path.join(here, "snapshots.npz"),
                        W=W[idx].astype(np.float32),
                        times=idx.astype(np.int64))
    print(f"[saved] results.json, snapshots.npz ({len(idx)} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny sizes, same code path; overwrites outputs")
    main(**vars(ap.parse_args()))
