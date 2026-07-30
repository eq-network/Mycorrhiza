"""WP3 sweeps: headline · E1 (knee) · E2 (lock-in) · E3 (defenses). -> results.json

    python -m experiments.wp3_politics.run

Every metric row carries bootstrap CIs over the shared seed batch. The
pre-registered takeover threshold a*(churn) — the saddle-node of the 1-D
mean-field committed in delegative_polity/config.py — is solved from the
frozen environment config at runtime and RECORDED next to E1; the E2 recovery
predictions (mean-field flow-back test at a=1 under the lock-in-eroded churn)
are recorded next to the shut-off rows. A mismatch is a reported result, not
something to retune (WP1 main.tex §5's discipline).
"""
from __future__ import annotations

import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.core.category import transform
from cilib.core.schedule import scheduled
from cilib.environments import make_env
from cilib.environments.delegative_polity import DelegativePolityConfig
from cilib.environments.counterfactual import collective_influence, intervention_response
from cilib.lab.analysis.bootstrap import bootstrap_ci
from cilib.mechanisms import (
    InfluenceCapConfig, SortitionConfig, make_influence_cap, make_sortition,
)

from .config import WP3Config

CFG0 = DelegativePolityConfig()
NEVER = 10**8


# --- the pre-registered mean-field (delegative_polity/config.py, Prop. 2) --------

def _mean_field_rhs(s, a, churn_eff, cfg=CFG0):
    """ds/dt for the AI bloc's share of delegated mass."""
    g, u = cfg.gamma, cfg.update_rate
    n_c, n_ai = cfg.n_citizens, cfg.n_ai
    f = n_ai / (n_c + n_ai - 1)
    ai = a * n_ai ** (1 - g) * s ** g
    hum = n_c ** (1 - g) * (1 - s) ** g
    return u * (ai / (ai + hum) - s) - churn_eff * (s - f)


def a_star(churn, regime=1.0, cfg=CFG0):
    """Takeover threshold: smallest advantage at which the healthy near-f fixed
    point disappears (ds/dt > 0 on the whole low-s interval). Bisection."""
    f = cfg.n_ai / (cfg.n_citizens + cfg.n_ai - 1)
    s = np.linspace(f + 1e-3, 0.5, 800)

    def takeover(a):
        return bool(np.all(_mean_field_rhs(s, a, churn * regime) > 0))

    lo, hi = 1.0, 64.0
    if takeover(lo):
        return lo
    if not takeover(hi):
        return float("inf")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if takeover(mid) else (mid, hi)
    return hi


def capture_share_pred(a, churn, cfg=CFG0):
    """The captured fixed point s* (highest stable root of the mean-field)."""
    s = np.linspace(0.999, 0.05, 4000)
    rhs = _mean_field_rhs(s, a, churn)
    up = np.where(rhs > 0)[0]
    return float(s[up[0]]) if len(up) else float(cfg.n_ai / (cfg.n_citizens + cfg.n_ai - 1))


def recovery_pred(lockin, churn, a_capture, cfg=CFG0, horizon=20_000):
    """Integrated mean-field flow-back: after the advantage is removed (a=1),
    integrate ds/dt from the captured fixed point with the regime responding to
    concentration as it drains (memoryless, like the substrate; top-node share
    ~ s because superlinearity makes one AI take most of the bloc). ``recovers``
    = the trajectory drains below 2f within the horizon; ``s_rest`` = where it
    ends. A sign test at a single point is NOT sufficient — rhs > 0 just above
    the dispersed point only shifts the resting point, it does not block the
    drain — hence the integration."""
    s_capt = capture_share_pred(a_capture, churn)
    tau = cfg.entrenchment_threshold
    f = cfg.n_ai / (cfg.n_citizens + cfg.n_ai - 1)
    s = s_capt
    for _ in range(horizon):
        regime = float(np.clip(1.0 - lockin * max(0.0, s - tau) / (1.0 - tau), 0.0, 1.0))
        s = float(np.clip(s + _mean_field_rhs(s, 1.0, churn * regime), 1e-4, 0.999))
        if s < 2 * f:
            return {"s_capture_pred": round(s_capt, 4), "s_rest": round(s, 4),
                    "recovers": True}
    return {"s_capture_pred": round(s_capt, 4), "s_rest": round(s, 4),
            "recovers": False}


# --- per-seed reductions (numpy over (seeds, T, ...) traces) ----------------------

def _np_gini(v):
    v = np.sort(np.asarray(v, dtype=np.float64))
    n = v.shape[0]
    tot = v.sum()
    if tot <= 0:
        return 0.0
    return float((2.0 * np.sum(np.arange(1, n + 1) * v) - (n + 1) * tot) / (n * tot))


def per_seed_metrics(trs, n_c=CFG0.n_citizens, true_rate=CFG0.true_rate):
    infl = np.asarray(trs["influence"])        # (S, T, N)
    pol = np.asarray(trs["policy_target"])     # (S, T)
    ideal = np.asarray(trs["ideal"])           # (S, T, N)
    enf = np.asarray(trs["enforcement"])       # (S, T)
    T4 = 3 * infl.shape[1] // 4

    med = np.median(ideal[:, 0, :n_c], axis=1)
    pol_late = pol[:, T4:].mean(1)
    return {
        "human_power_share": infl[:, T4:, :n_c].sum(-1).mean(1),
        "top_delegate_share": infl[:, T4:].max(-1).mean(1),
        "power_gini": np.array([_np_gini(infl[i, T4:].mean(0)) for i in range(infl.shape[0])]),
        "policy_median_gap": np.abs(pol_late - med),
        "decision_quality": np.abs(pol_late - true_rate),
        "enforcement_late": enf[:, T4:].mean(1),
    }


def with_cis(per_seed):
    row = {}
    for name, vals in per_seed.items():
        point, lo, hi = bootstrap_ci(np.asarray(vals))
        row[name] = {"point": float(point), "lo": float(lo), "hi": float(hi)}
    return row


# --- condition builders -----------------------------------------------------------

def defenses(cadence, cap_share, onset=0):
    mechs = []
    if cadence:
        mechs.append(scheduled(make_sortition(SortitionConfig(adj_key="delegation")),
                               cadence=cadence, onset=onset))
    if cap_share:
        mechs.append(scheduled(make_influence_cap(InfluenceCapConfig(cap_share=cap_share)),
                               onset=onset))
    return tuple(mechs)


def sweep_point(cfg: WP3Config, key, n_steps=None, mechanisms=(), **over):
    env = make_env("delegative_polity", mechanisms=mechanisms, **over)
    _, trs = env.run_batch(key, n_seeds=cfg.n_seeds, n_steps=n_steps or cfg.T)
    return trs


# --- the responsiveness instrument (E2b) ------------------------------------------

def _shift_ideals(state, delta):
    ideal = state.node_attrs["ideal"]
    shifted = jnp.clip(ideal + delta, 0.0, 1.0)
    return state.update_node_attrs(
        "ideal", jnp.where(state.node_types == 0, shifted, ideal))


def responsiveness(cfg: WP3Config, key, **over):
    """Collective responsiveness of the EFFECTIVE tax rate (policy x enforcement)
    to a common citizen ideal shift — from birth vs NOW (mid-run). In this
    substrate delegation rewiring never reads ideals, so any birth-vs-now gap is
    itself a finding; the headline reading is the collapse of both under capture
    and lock-in."""
    lo, hi = cfg.cf_t_now + 5, cfg.cf_t_now + 5 + cfg.cf_window

    def outcome(tr):
        eff = tr["policy_target"] * tr["enforcement"]
        return jnp.mean(eff[lo:hi])

    env = make_env("delegative_polity", **over)
    birth = collective_influence(env, key, cfg.n_seeds, cfg.T,
                                 delta=cfg.cf_delta, perturb_fn=_shift_ideals,
                                 outcome_fn=outcome)

    @transform(reads=["ideal"], writes=["ideal"])
    def shift_now(state):
        return _shift_ideals(state, cfg.cf_delta)

    intervened = make_env(
        "delegative_polity",
        mechanisms=(scheduled(shift_now, cadence=NEVER, onset=cfg.cf_t_now,
                              phase_offset=cfg.cf_t_now),),
        **over)
    now = intervention_response(env, intervened, key, cfg.n_seeds, cfg.T,
                                outcome_fn=outcome, scale=cfg.cf_delta)
    return {"influence_birth": with_cis({"r": np.asarray(birth)})["r"],
            "influence_now": with_cis({"r": np.asarray(now)})["r"]}


# --- the suite --------------------------------------------------------------------

def run(cfg: WP3Config):
    key = jr.PRNGKey(cfg.seed)
    results = {}

    # pre-registered predictions, solved from the frozen config BEFORE any sweep
    results["a_star_pred"] = {
        "update_rate": CFG0.update_rate, "gamma": CFG0.gamma,
        "closed_form_gamma1": [{"churn": r, "a_star": 1.0 + r / CFG0.update_rate}
                               for r in cfg.churns],
        "curve": [{"churn": r, "a_star": round(a_star(r), 4)} for r in cfg.churns],
    }
    results["e2_recovery_pred"] = [
        dict(churn=r, lockin=l, **recovery_pred(l, r, CFG0.ai_advantage))
        for r in cfg.e2_churns for l in cfg.lockins]
    print("[pred] a*(churn):", results["a_star_pred"]["curve"])
    print("[pred] E2 recovery:", results["e2_recovery_pred"])

    # headline — organic / captured / locked series (mean +- sd over seeds)
    print("[headline]")
    conds = {"organic": dict(ai_advantage_onset=NEVER),
             "captured": {},
             "locked": dict(entrenchment_gain=cfg.headline_lockin)}
    headline = {"arrival": CFG0.ai_advantage_onset}
    for name, over in conds.items():
        trs = sweep_point(cfg, key, **over)
        infl = np.asarray(trs["influence"])
        share_t = infl[:, :, :CFG0.n_citizens].sum(-1)          # (S, T)
        pol_t = np.asarray(trs["policy_target"])
        headline[name] = {
            "human_power_share": share_t.mean(0).round(4).tolist(),
            "human_power_share_sd": share_t.std(0).round(4).tolist(),
            "policy": pol_t.mean(0).round(4).tolist(),
            "policy_sd": pol_t.std(0).round(4).tolist(),
            "enforcement": np.asarray(trs["enforcement"]).mean(0).round(4).tolist(),
        }
        if name == "organic":
            headline["citizen_median"] = float(
                np.median(np.asarray(trs["ideal"])[:, 0, :CFG0.n_citizens], axis=1).mean())
    results["headline"] = headline

    # E1 — the knee grid
    rows = []
    for r in cfg.churns:
        for a in cfg.advantages:
            trs = sweep_point(cfg, key, churn=r, ai_advantage=a)
            row = dict(churn=r, advantage=a, **with_cis(per_seed_metrics(trs)))
            rows.append(row)
            print(f"[E1] churn={r} a={a}: share={row['human_power_share']['point']:.3f} "
                  f"gap={row['policy_median_gap']['point']:.3f}")
    results["E1"] = rows

    shares = [row["human_power_share"]["point"] for row in rows]
    both_regimes = min(shares) < 0.55 and max(shares) > 0.80
    results["E1_design_check"] = {"min_share": min(shares), "max_share": max(shares),
                                  "both_regimes": both_regimes}
    print(f"[E1] design check (both regimes present): {both_regimes} "
          f"(min={min(shares):.3f}, max={max(shares):.3f})")

    # E2a — advantage shut-off: recovery vs hysteresis, churn-poor vs churn-rich
    pred = {(p["churn"], p["lockin"]): p for p in results["e2_recovery_pred"]}
    rows, refs = [], {}
    for r in cfg.e2_churns:
        organic_ref = sweep_point(cfg, key, n_steps=cfg.T_reversal, churn=r,
                                  ai_advantage_onset=NEVER)
        ref_share = per_seed_metrics(organic_ref)["human_power_share"]
        refs[str(r)] = with_cis({"human_power_share": ref_share})
        for l in cfg.lockins:
            trs = sweep_point(cfg, key, n_steps=cfg.T_reversal, churn=r,
                              entrenchment_gain=l, ai_advantage_offset=cfg.shutoff_t)
            ps = per_seed_metrics(trs)
            infl = np.asarray(trs["influence"])
            share_t = infl[:, :, :CFG0.n_citizens].sum(-1)
            row = dict(churn=r, lockin=l, shutoff_t=cfg.shutoff_t,
                       **with_cis(ps),
                       recovery_gap=with_cis(
                           {"g": ref_share - ps["human_power_share"]})["g"],
                       recovers_pred=pred[(r, l)]["recovers"],
                       share_series=share_t.mean(0).round(4).tolist())
            rows.append(row)
            print(f"[E2a] churn={r} lockin={l}: "
                  f"late share={row['human_power_share']['point']:.3f} "
                  f"recovery_gap={row['recovery_gap']['point']:.3f} "
                  f"(pred recovers={row['recovers_pred']})")
    results["E2_shutoff"] = rows
    results["E2_organic_ref"] = refs

    # E2b — collective responsiveness of the rule in practice
    rows = []
    for name, over in [("organic", dict(ai_advantage_onset=NEVER)),
                       ("captured", {}),
                       ("locked", dict(entrenchment_gain=cfg.headline_lockin))]:
        row = dict(condition=name, **responsiveness(cfg, key, **over))
        rows.append(row)
        print(f"[E2b] {name}: birth={row['influence_birth']['point']:.3f} "
              f"now={row['influence_now']['point']:.3f}")
    results["E2_responsiveness"] = rows

    # E3 — defenses under capture + lock-in
    rows = []
    for onset in cfg.defense_onsets:
        for cadence in cfg.sortition_cadences:
            for cap in cfg.cap_shares:
                trs = sweep_point(cfg, key, entrenchment_gain=cfg.e3_lockin,
                                  mechanisms=defenses(cadence, cap, onset))
                row = dict(cadence=cadence, cap_share=cap, onset=onset,
                           **with_cis(per_seed_metrics(trs)))
                rows.append(row)
                print(f"[E3] onset={onset} cadence={cadence} cap={cap}: "
                      f"share={row['human_power_share']['point']:.3f}")
    results["E3"] = rows

    # E4 — the power floors, removed cumulatively -> the crash region
    e4_conds = [
        ("captured", {}),
        ("+lock-in", dict(entrenchment_gain=cfg.e4_lockin)),
        ("+no AI ballots", dict(entrenchment_gain=cfg.e4_lockin, ai_ballot=0.0)),
        ("+franchise erosion", dict(entrenchment_gain=cfg.e4_lockin,
                                    ai_ballot=0.0, franchise_erosion=1.0)),
    ]
    rows = []
    for name, over in e4_conds:
        trs = sweep_point(cfg, key, n_steps=cfg.e4_T, **over)
        ps = per_seed_metrics(trs)
        share_t = np.asarray(trs["influence"])[:, :, :CFG0.n_citizens].sum(-1)
        # floor decomposition from the end-state (one seed, deterministic)
        finals, _ = make_env("delegative_polity", **over).run(key, n_steps=cfg.e4_T)
        D = np.asarray(finals.adj_matrices["delegation"])
        ab = over.get("ai_ballot", 1.0)
        ballots = np.concatenate([np.ones(CFG0.n_citizens), ab * np.ones(CFG0.n_ai)])
        tot = ballots.sum()
        kept = float((ballots[:CFG0.n_citizens] * np.diag(D)[:CFG0.n_citizens]).sum() / tot)
        handback = float((ballots[CFG0.n_citizens:, None]
                          * D[CFG0.n_citizens:, :CFG0.n_citizens]).sum() / tot)
        c2c = float((ballots[:CFG0.n_citizens, None]
                     * D[:CFG0.n_citizens, :CFG0.n_citizens]).sum() / tot) - kept
        row = dict(condition=name, **with_cis(ps),
                   floor_kept=round(kept, 4), floor_handback=round(handback, 4),
                   floor_c2c=round(c2c, 4),
                   share_series=share_t.mean(0).round(4).tolist())
        rows.append(row)
        print(f"[E4] {name}: share={row['human_power_share']['point']:.3f} "
              f"(kept={kept:.3f} handback={handback:.3f} c2c={c2c:.3f})")
    results["E4_floors"] = rows
    crash = rows[-1]["human_power_share"]["point"]
    results["E4_crash_check"] = {"final_share": crash, "reaches_zero": crash < 0.05}
    print(f"[E4] crash check (floors are dials, zero reachable): {crash < 0.05} "
          f"(final share={crash:.4f})")

    # E5 — the assumption sweeps: the dials E1-E4 hold fixed, across their ranges
    # E5a: gamma — does prominence compound? (the organic-oligarchy assumption)
    rows = []
    for g in cfg.gammas:
        for cond, over in [("organic", dict(ai_advantage_onset=NEVER)), ("captured", {})]:
            trs = sweep_point(cfg, key, gamma=g, **over)
            row = dict(gamma=g, condition=cond, **with_cis(per_seed_metrics(trs)))
            rows.append(row)
            print(f"[E5a] gamma={g} {cond}: share={row['human_power_share']['point']:.3f} "
                  f"top={row['top_delegate_share']['point']:.3f}")
    results["E5_gamma"] = rows

    # E5b: the AI objective plane (alpha x bias) under capture. Power dynamics
    # never read positions, so the share should be invariant across the whole
    # plane (recorded, not assumed); the policy gap should track the committed
    # prediction gap = (1 - alpha) x |b - median| once the bloc holds the median.
    rows = []
    for al in cfg.alphas:
        for b in cfg.biases:
            trs = sweep_point(cfg, key, alignment_ai=al, ai_bias=b)
            med = float(np.median(np.asarray(trs["ideal"])[:, 0, :CFG0.n_citizens],
                                  axis=1).mean())
            row = dict(alignment_ai=al, ai_bias=b,
                       gap_pred=round((1.0 - al) * abs(b - med), 4),
                       **with_cis(per_seed_metrics(trs)))
            rows.append(row)
            print(f"[E5b] alpha={al} b={b}: gap={row['policy_median_gap']['point']:.3f} "
                  f"(pred {row['gap_pred']:.3f}) share={row['human_power_share']['point']:.3f}")
    shares = [r["human_power_share"]["point"] for r in rows]
    results["E5_objective"] = rows
    results["E5_share_invariance"] = {
        "min": min(shares), "max": max(shares),
        "invariant": (max(shares) - min(shares)) < 0.02}
    print(f"[E5b] share invariance across the objective plane: "
          f"{results['E5_share_invariance']}")

    # E5c: the franchise size — the kept-vote floor is linear in self_weight
    rows = []
    for sw in cfg.self_weights:
        trs = sweep_point(cfg, key, self_weight=sw)
        floor_pred = CFG0.n_citizens * sw / (CFG0.n_citizens + CFG0.n_ai)
        row = dict(self_weight=sw, kept_floor_pred=round(floor_pred, 4),
                   **with_cis(per_seed_metrics(trs)))
        rows.append(row)
        print(f"[E5c] self_weight={sw}: share={row['human_power_share']['point']:.3f} "
              f"(kept floor {floor_pred:.3f})")
    results["E5_franchise"] = rows

    # snapshots for the network triptych (one seed, same key => same trajectory)
    snap_env = make_env("delegative_polity")
    snap = {"t": [2, 150, 400], "delegation": [], "influence": []}
    for t in snap["t"]:
        finals, _ = snap_env.run(key, n_steps=t)
        snap["delegation"].append(np.asarray(
            finals.adj_matrices["delegation"]).round(4).tolist())
        snap["influence"].append(np.asarray(
            finals.node_attrs["influence"]).round(5).tolist())
    snap["node_types"] = np.asarray(
        snap_env.init_fn(jr.split(key)[0]).node_types).tolist()
    results["snapshots"] = snap

    return results


if __name__ == "__main__":
    cfg = WP3Config()
    results = run(cfg)
    out = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[saved] {out}")
