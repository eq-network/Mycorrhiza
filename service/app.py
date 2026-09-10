"""RemoteEngine v0 — live ledger_society runs behind one stateless endpoint
(docs/remote-engine-design.md, 2026-07-31).

    pip install -e .[service]
    uvicorn service.app:app --port 8100

A game in progress is nothing but (scenario, seed, plan). POST /run carries
the accumulated plan as piecewise segments; the engine re-runs the whole
trajectory from t=0 (shared-seed determinism: the past never redraws) and the
response is the contract-shaped payload up to ``upto`` plus a run-record
manifest. One jitted runner per (scenario, T) shape — the plan is a dynamic
argument, so requests never recompile.

The lever whitelist IS the assumptions card: values outside the declared
ranges are refused with the range in the error, never clamped silently
(the transform clips too, as a second line of defence). v0 caps: one
scenario, T <= 800, no arbitrary config passthrough. Rate limiting is the
deployment's job (Fly/Cloud Run), not implemented here — README.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.metadata
import json
from typing import Dict, List

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cilib.core.scan import run_scan
from cilib.environments import make_env
from cilib.mechanisms import PLAN_LEVERS, PolicyLeverConfig
from cilib.mechanisms.families import (
    CULTURE_LEVERS, ECONOMY_LEVERS, POLITICS_LEVERS,
    CultureLeverConfig, EconomyLeverConfig, PoliticsLeverConfig,
    culture_upkeep, external_intensity_of,
)

T_MAX = 800
ROUND_DECIMALS = 4

# named scenarios only — config never crosses the wire
SCENARIOS: Dict[str, dict] = {
    "ledger-society-default": {"regime_rate": 0.02},
}

_LEVERS = PolicyLeverConfig()
LEVER_RANGES: Dict[str, tuple] = {
    "levy_rate": (0.0, _LEVERS.levy_max),
    "repair_rate": (0.0, _LEVERS.repair_max),
    "reach_cut": (0.0, _LEVERS.reach_cut_max),
    "sortition_rate": (0.0, _LEVERS.sortition_max),
}

# --- the three lever families (docs/gd-game-three-families.md) --------------
# Same contract as the four-lever closure above: named levers, declared ranges
# that are both this whitelist and the in-transform clip, values off the range
# refused rather than clamped. The families are alternative closures to
# `policy_horizon` over the same channels — a request uses one or the other.

_ECON, _CULT, _POL = (EconomyLeverConfig(), CultureLeverConfig(),
                      PoliticsLeverConfig())

FAMILY_LEVERS: Dict[str, tuple] = {
    "economy": ECONOMY_LEVERS,
    "culture": CULTURE_LEVERS,
    "politics": POLITICS_LEVERS,
}

# Column 4 of the politics plan is not a player control: it is the bill the
# other two families run up, assembled server-side by `external_intensity_of`
# so the coupling cannot be bypassed by a client that simply omits it.
DERIVED_LEVERS = {("politics", "external_intensity")}

FAMILY_RANGES: Dict[str, Dict[str, tuple]] = {
    "economy": {
        # bands are deltas on the neutral allocation point; the transform
        # clips the resulting band, so the delta range is the band width
        "d_consume": (_ECON.band_lo[0] - 0.70, _ECON.band_hi[0] - 0.70),
        "d_invest": (_ECON.band_lo[1] - 0.05, _ECON.band_hi[1] - 0.05),
        "d_broadcast": (_ECON.band_lo[2] - 0.03, _ECON.band_hi[2] - 0.03),
        "d_lobby": (_ECON.band_lo[3] - 0.02, _ECON.band_hi[3] - 0.02),
        "levy_rate": (0.0, _ECON.levy_max),
    },
    "culture": {
        "gamma_w_delta": (-_CULT.gamma_w_delta_max, _CULT.gamma_w_delta_max),
        "update_rate_w_delta": (_CULT.update_rate_w_delta_lo,
                                _CULT.update_rate_w_delta_hi),
        "reach_cut": (0.0, _CULT.reach_cut_max),
    },
    "politics": {
        "repair_spend_rate": (0.0, _POL.repair_spend_max),
        "sortition_rate": (0.0, _POL.sortition_max),
        "repair_rate": (0.0, _POL.repair_rate_max),
        "entrenchment_gain": (0.0, _POL.entrenchment_gain_max),
        "external_intensity": (0.0, _POL.external_intensity_max),
    },
}

WHITELIST_GLOBAL = ("efficiency", "policy_target", "enforcement",
                    "gamma_w_now", "update_rate_w_now", "reach_cut_now",
                    "repair_rate_now", "entrenchment_gain_now",
                    # the four ledger shares, per tick — what a client plots
                    # its curve and its ghost from
                    "human_wealth_share", "human_income_share",
                    "human_attention_share", "human_power_share")
WHITELIST_NODE = ("wealth", "listen_influence", "influence",
                  "intervention_spend", "top_listen_target",
                  "top_delegate_target")

app = FastAPI(title="cilib RemoteEngine", version="0.1")
# The view side runs on 4321 by default and on 4322 when that port is taken;
# a browser reports a disallowed origin exactly like a service that is down, so
# both the localhost and 127.0.0.1 spellings of both ports are listed here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://eq-network.org",
                   "http://localhost:4321", "http://127.0.0.1:4321",
                   "http://localhost:4322", "http://127.0.0.1:4322"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class PlanSegment(BaseModel):
    from_tick: int = Field(ge=0)
    levers: Dict[str, float]


class RunRequest(BaseModel):
    scenario_id: str = "ledger-society-default"
    seed: int = Field(default=0, ge=0, le=2 ** 31 - 1)
    T: int = Field(default=400, ge=1, le=T_MAX)
    plan: List[PlanSegment] = []
    upto: int = Field(default=400, ge=1)
    # The three-family closure. When present, `plan` must be empty: the two
    # closures write the same channels and cannot compose (build_game refuses).
    families: Dict[str, List[PlanSegment]] = {}


@functools.lru_cache(maxsize=8)
def _runner(scenario_id: str, T: int):
    env = make_env("ledger_society", policy_horizon=T,
                   **SCENARIOS[scenario_id])

    def _run(key, plan):
        k_init, k_run = jr.split(key)
        state = env.init_fn(k_init).update_global_attr("policy_plan", plan)
        return run_scan(env.round_fn, state, T, k_run, trace_fn=env.trace_fn)

    return env, jax.jit(_run)


@functools.lru_cache(maxsize=8)
def _family_runner(scenario_id: str, T: int):
    """One warm program for the three-family closure. All three families are
    always present at horizon T; a family the client did not steer simply
    carries its neutral (all-zero) plan, which is bit-identical to the family
    being absent. That keeps ONE compiled program for every combination of
    tabs the player touches."""
    env = make_env("ledger_society", economy_horizon=T, culture_horizon=T,
                   politics_horizon=T, **SCENARIOS[scenario_id])

    def _run(key, econ, cult, pol):
        k_init, k_run = jr.split(key)
        state = env.init_fn(k_init)
        state = state.update_global_attr("economy_plan", econ)
        state = state.update_global_attr("culture_plan", cult)
        state = state.update_global_attr("politics_plan", pol)
        return run_scan(env.round_fn, state, T, k_run, trace_fn=env.trace_fn)

    return env, jax.jit(_run)


def _expand_family(segments: List[PlanSegment], family: str,
                   T: int) -> np.ndarray:
    """Piecewise segments -> a (T, P) plan for one family, validated against
    that family's declared ranges. Derived columns are refused: a client may
    not set the bill its siblings owe."""
    levers = FAMILY_LEVERS[family]
    ranges = FAMILY_RANGES[family]
    plan = np.zeros((T, len(levers)), dtype=np.float32)
    for seg in sorted(segments, key=lambda s: s.from_tick):
        if seg.from_tick >= T:
            raise HTTPException(422, f"from_tick {seg.from_tick} >= T {T}")
        for name, value in seg.levers.items():
            if name not in ranges:
                raise HTTPException(
                    422, f"unknown {family} lever {name!r}; "
                         f"levers: {[l for l in levers if (family, l) not in DERIVED_LEVERS]}")
            if (family, name) in DERIVED_LEVERS:
                raise HTTPException(
                    422, f"{family}.{name} is derived from the other families' "
                         f"plans and is assembled by the server; it cannot be set")
            lo, hi = ranges[name]
            if not lo <= value <= hi:
                raise HTTPException(
                    422, f"{family}.{name}={value} outside its declared range "
                         f"[{lo}, {hi}] — off-card values are refused, "
                         f"never clamped")
            plan[seg.from_tick:, levers.index(name)] = value
    return plan


def _bill_the_siblings(econ: np.ndarray, cult: np.ndarray,
                       pol: np.ndarray) -> np.ndarray:
    """Fill the politics plan's `external_intensity` column from the economy
    and culture plans, per tick. This is the coupling made unavoidable: the
    levy and the cultural levers are political acts, and the office pays for
    them whether or not the player is looking at the politics tab."""
    col = POLITICS_LEVERS.index("external_intensity")
    levy_col = ECONOMY_LEVERS.index("levy_rate")
    out = pol.copy()
    for t in range(pol.shape[0]):
        out[t, col] = float(external_intensity_of(
            levy_rate=float(econ[t, levy_col]),
            culture_row=jnp.asarray(cult[t]), cfg=_POL,
            levy_max=_ECON.levy_max))
    return out


def _expand_plan(segments: List[PlanSegment], T: int) -> np.ndarray:
    plan = np.zeros((T, len(PLAN_LEVERS)), dtype=np.float32)
    for seg in sorted(segments, key=lambda s: s.from_tick):
        if seg.from_tick >= T:
            raise HTTPException(422, f"from_tick {seg.from_tick} >= T {T}")
        for name, value in seg.levers.items():
            if name not in LEVER_RANGES:
                raise HTTPException(422, f"unknown lever {name!r}; "
                                         f"levers: {list(LEVER_RANGES)}")
            lo, hi = LEVER_RANGES[name]
            if not lo <= value <= hi:
                raise HTTPException(
                    422, f"{name}={value} outside its declared range "
                         f"[{lo}, {hi}] — off-card values are refused, "
                         f"never clamped")
            plan[seg.from_tick:, PLAN_LEVERS.index(name)] = value
    return plan


def _series(arr: np.ndarray) -> list:
    if np.issubdtype(arr.dtype, np.integer):
        return arr.tolist()
    return np.round(arr.astype(np.float64), ROUND_DECIMALS).tolist()


@app.get("/health")
def health():
    return {"ok": True, "scenarios": list(SCENARIOS),
            "levers": {k: list(v) for k, v in LEVER_RANGES.items()},
            "families": {
                fam: {name: list(rng) for name, rng in ranges.items()
                      if (fam, name) not in DERIVED_LEVERS}
                for fam, ranges in FAMILY_RANGES.items()},
            "derived": [f"{fam}.{name}" for fam, name in sorted(DERIVED_LEVERS)]}


@app.post("/run")
def run(req: RunRequest):
    if req.scenario_id not in SCENARIOS:
        raise HTTPException(422, f"unknown scenario {req.scenario_id!r}")
    upto = min(req.upto, req.T)

    if req.families:
        return _run_families(req, upto)

    plan = _expand_plan(req.plan, req.T)
    env, fn = _runner(req.scenario_id, req.T)
    _, trace = fn(jr.PRNGKey(req.seed), jnp.asarray(plan))

    n = env.config.n_humans + env.config.n_ai
    out_global = {k: _series(np.asarray(trace[k][:upto]))
                  for k in WHITELIST_GLOBAL if k in trace}
    out_node = {k: _series(np.asarray(trace[k][:upto]).reshape(upto * n))
                for k in WHITELIST_NODE if k in trace}
    manifest = {
        "suite": "gradual_disempowerment",
        "scenario_id": req.scenario_id,
        "env": "ledger_society",
        "seed": req.seed,
        "T": req.T,
        "upto": upto,
        "plan": [seg.model_dump() for seg in req.plan],
        "engine": {
            "package": "collective-intelligence-library",
            "version": importlib.metadata.version(
                "collective-intelligence-library"),
            "jax_version": jax.__version__,
        },
        "config_hash": hashlib.sha256(json.dumps(
            {"config": dataclasses.asdict(env.config), "seed": req.seed,
             "plan": plan.tolist()},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    return {"global": out_global, "node": out_node,
            "meta": {"gameId": "ledger_society", "T": upto, "N": n,
                     "n_humans": env.config.n_humans,
                     "n_ai": env.config.n_ai,
                     "seed": req.seed},
            "manifest": manifest}


def _run_families(req: RunRequest, upto: int):
    """The three-family closure. Every request carries the whole plan and the
    engine re-runs from t=0, so the past never redraws and there is no session
    to store — the same contract as the four-lever path above."""
    if req.plan:
        raise HTTPException(
            422, "`plan` (the four-lever closure) and `families` write the "
                 "same channels and cannot compose — send one or the other")
    unknown = set(req.families) - set(FAMILY_LEVERS)
    if unknown:
        raise HTTPException(422, f"unknown families {sorted(unknown)}; "
                                 f"families: {sorted(FAMILY_LEVERS)}")

    econ = _expand_family(req.families.get("economy", []), "economy", req.T)
    cult = _expand_family(req.families.get("culture", []), "culture", req.T)
    pol = _expand_family(req.families.get("politics", []), "politics", req.T)
    pol = _bill_the_siblings(econ, cult, pol)

    env, fn = _family_runner(req.scenario_id, req.T)
    _, trace = fn(jr.PRNGKey(req.seed), jnp.asarray(econ), jnp.asarray(cult),
                  jnp.asarray(pol))

    n = env.config.n_humans + env.config.n_ai
    out_global = {k: _series(np.asarray(trace[k][:upto]))
                  for k in WHITELIST_GLOBAL if k in trace}
    out_node = {k: _series(np.asarray(trace[k][:upto]).reshape(upto * n))
                for k in WHITELIST_NODE if k in trace}
    metrics = {name: float(fn_(trace))
               for name, fn_ in env.metrics.items()
               if name.startswith("journey_")}
    manifest = {
        "suite": "gradual_disempowerment",
        "closure": "three-families",
        "scenario_id": req.scenario_id,
        "env": "ledger_society",
        "seed": req.seed,
        "T": req.T,
        "upto": upto,
        "families": {fam: [seg.model_dump() for seg in segs]
                     for fam, segs in req.families.items()},
        # the bill the server assembled, not something the client sent
        "external_intensity": _series(
            pol[:, POLITICS_LEVERS.index("external_intensity")]),
        "engine": {
            "package": "collective-intelligence-library",
            "version": importlib.metadata.version(
                "collective-intelligence-library"),
            "jax_version": jax.__version__,
        },
        "config_hash": hashlib.sha256(json.dumps(
            {"config": dataclasses.asdict(env.config), "seed": req.seed,
             "economy": econ.tolist(), "culture": cult.tolist(),
             "politics": pol.tolist()},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    return {"global": out_global, "node": out_node, "metrics": metrics,
            "meta": {"gameId": "ledger_society", "T": upto, "N": n,
                     "n_humans": env.config.n_humans,
                     "n_ai": env.config.n_ai,
                     "seed": req.seed},
            "manifest": manifest}
