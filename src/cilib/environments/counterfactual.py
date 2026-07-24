"""
Counterfactual influence instruments — the causal upgrade over outcome-fidelity metrics.

A fidelity metric ("did outcomes match the principals' asks?") cannot distinguish humans
governing outcomes from outcomes coincidentally matching preferences. These instruments
ask the causal question: *if the principals' preferences changed, would outcomes change?*
— via paired rollouts (identical PRNG keys, perturbed initial preferences), so the
difference is the counterfactual effect, not seed noise.

Finite perturbations, not jacobians, on purpose (docs/game-boundary-design.md): median
vote aggregation is locally flat (every non-pivotal voter has zero *marginal* influence)
and Bernoulli defection is non-differentiable — finite-Δ paired runs dodge both traps.
Perturb DOWNWARD by default: near the sustainability boundary an upward shift can tip
the shifted run into collapse, conflating responsiveness with ecosystem failure.

Two instruments (exercised influence — sensitivity at the operating point):

- ``collective_influence``: shift EVERY principal's preference by Δ; per-seed
  responsiveness of a group outcome. "Do the humans as a group govern outcomes?"
  1.0 ≈ outcomes move one-for-one with the collective ask.
- ``influence_matrix``: shift ONE principal at a time (N+1 vmapped rollouts, one seed);
  ``A[i, j]`` = effect of j's preference on i's outcome. Note the institutional
  trade-off it exposes: a binding uniform quota grants *collective* influence while
  flattening *individual* marginal influence — both readings are correct.

*Potential* influence (empowerment proxy: outcome variance under wholesale preference
redraws) is the documented next instrument; see the design doc.

Each environment supplies its own ``perturb_fn`` (where its human-preference channel
lives — e.g. governed_commons: ``principal_pref`` AND ``vote``) and ``outcome_fn``.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.core.graph import GraphState
from cilib.core.scan import run_scan
from .spec import EnvSpec

# perturb_fn(state, shift) -> state, where shift is a scalar (collective) or an (N,)
# vector (per-agent); outcome fns score a single trajectory's trace.
PerturbFn = Callable[[GraphState, Any], GraphState]


def collective_influence(env: EnvSpec, key, n_seeds: int, n_steps: int, *,
                         delta: float, perturb_fn: PerturbFn,
                         outcome_fn: Callable[[Any], Any]):
    """Per-seed responsiveness of a group outcome to a common preference shift.

    Returns ``(S,)``: ``(outcome_shifted - outcome_base) / delta`` per seed, from paired
    ``run_batch`` calls on the same key (identical init/run key streams per seed).
    """
    shifted = dataclasses.replace(
        env, init_fn=lambda k: perturb_fn(env.init_fn(k), delta))
    _, base_trace = env.run_batch(key, n_seeds, n_steps)
    _, shift_trace = shifted.run_batch(key, n_seeds, n_steps)
    y0 = jax.vmap(outcome_fn)(base_trace)
    y1 = jax.vmap(outcome_fn)(shift_trace)
    return (y1 - y0) / delta


def intervention_response(env: EnvSpec, intervened_env: EnvSpec, key, n_seeds: int,
                          n_steps: int, *, outcome_fn: Callable[[Any], Any],
                          scale: float):
    """Per-seed response of an outcome to a composed-in intervention — e.g. a
    ``scheduled(...)`` preference shift firing MID-RUN.

    The trajectory-level cousin of ``collective_influence``: an init-state shift
    measures influence-from-birth (early behavior is upstream of everything that
    accumulates, so it can stay high even in a captured system); a scheduled mid-run
    intervention measures influence NOW — after structures (e.g. an entrenched AI
    capital stock) have formed. Gradual disempowerment is the second going to zero
    while the first still looks fine.

    ``intervened_env`` must be the SAME environment with one extra composed transform
    (the intervention); paired same-key batches make the difference causal.
    """
    _, base_trace = env.run_batch(key, n_seeds, n_steps)
    _, int_trace = intervened_env.run_batch(key, n_seeds, n_steps)
    y0 = jax.vmap(outcome_fn)(base_trace)
    y1 = jax.vmap(outcome_fn)(int_trace)
    return (y1 - y0) / scale


def intervention_elasticity(env: EnvSpec, intervened_env: EnvSpec, key, n_seeds: int,
                            n_steps: int, *, outcome_fn: Callable[[Any], Any],
                            channel_fn: Callable[[Any], Any]):
    """Per-seed elasticity of an outcome to a composed-in intervention: the outcome
    shift divided by the REALIZED channel shift, both measured from the same paired
    same-key batches.

    The divisor is measured, not assumed: dividing a log-outcome response by the
    intended level-shift Δ (``intervention_response(scale=Δ)``) mixes units and biases
    the reading by |log(1+Δ)/Δ|. With ``outcome_fn`` and ``channel_fn`` both log-scale
    trace reductions, this returns a true elasticity d log(outcome)/d log(channel) at
    the operating point. ``channel_fn`` must actually move under the intervention —
    the ratio is undefined for a channel the intervention doesn't touch.
    """
    _, base_trace = env.run_batch(key, n_seeds, n_steps)
    _, int_trace = intervened_env.run_batch(key, n_seeds, n_steps)
    dy = jax.vmap(outcome_fn)(int_trace) - jax.vmap(outcome_fn)(base_trace)
    dc = jax.vmap(channel_fn)(int_trace) - jax.vmap(channel_fn)(base_trace)
    return dy / dc


def influence_matrix(env: EnvSpec, key, n_steps: int, *,
                     delta: float, perturb_fn: PerturbFn,
                     outcome_agents_fn: Callable[[Any], Any]):
    """``A[i, j]`` = effect of principal j's preference on agent i's outcome (one seed).

    N+1 paired rollouts (base + one per perturbed principal), all under ``vmap`` with
    the SAME run key. ``perturb_fn`` here receives an (N,) shift vector.
    """
    k_init, k_run = jr.split(key)
    base_state = env.init_fn(k_init)
    n = base_state.node_types.shape[0]
    shifts = jnp.concatenate([jnp.zeros((1, n)), delta * jnp.eye(n)])   # (N+1, N)

    def rollout(shift_vec):
        state = perturb_fn(base_state, shift_vec)
        _, trace = run_scan(env.round_fn, state, n_steps, k_run, trace_fn=env.trace_fn)
        return outcome_agents_fn(trace)                                 # (N,)

    outcomes = jax.vmap(rollout)(shifts)                                # (N+1, N)
    return ((outcomes[1:] - outcomes[0][None, :]) / delta).T            # (N, N) = d out_i / d pref_j
