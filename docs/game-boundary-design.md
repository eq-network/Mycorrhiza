# Design: the agent↔environment boundary (GameSpec)

*Deposited 2026-07-13 from a forest-walk design session (Stage 1–2, exploratory — verify
against Gymnax/JaxMARL interfaces and the open-games literature before treating as
settled). Direction set by Jonas: environments must be MDP/Gym-sense game forms awaiting
policies, not closed dynamical systems. This doc is the concrete shape.*

**Status (2026-07-13, same day):** migration steps 1–4 implemented — `GameSpec`/`close`/
`validate_reads` (`environments/game.py`), `governed_commons` split through the boundary
(`ai_delegate` is the closing policy), `policy_target` rename, and the counterfactual
instruments (`environments/counterfactual.py`: `collective_influence`,
`influence_matrix`; potential-influence proxy still open). First result: quota-only
restores ~0 causal influence at T=200 (knife-edge regime), quota+sanctions ~0.91 —
the exercised-vs-fidelity gap the design predicted. Step 5 (build A2 against the
boundary, then freeze) is next.

**FROZEN (2026-07-14):** `compute_economy` shipped as the second consumer with zero
interface changes — `GameSpec`/`close()`/`validate_reads` are now the standard. The
predicted friction materialized exactly as documented: heterogeneous populations
(households vs AI actors) are handled by masking inside `step_fn`; `close_multi`
remains deferred until A3's persuader/member split forces it. The instrument family
grew one member: `intervention_response` (mid-run scheduled interventions —
influence-NOW vs influence-from-birth; see `docs/abm-suite-design.md`).

## The one-sentence design

Split every round at the action: a **`GameSpec`** is an open game — `observe_fn` +
`step_fn(state, actions, key)` + reward view — and **`close(game, policy) → EnvSpec`**
recovers today's closed environments as a special case, so nothing downstream breaks.

## Why (recap of the failure)

`governed_commons` welds the delegate's decision rule into `dynamics.py::make_decide`.
Consequences: policies aren't pluggable, learning agents can't drop in, strategic
behavior can't be tested, the `ai_delegate` catalog entry is decorative, and influence
can only be measured correlationally (outcome fidelity) instead of counterfactually
(perturb the input, measure the outcome), because preferences are buried in `init_fn`
instead of being explicit inputs at a boundary.

## The interface

```python
# src/cilib/environments/game.py  (sibling of spec.py — a catalog contract, not core)

@dataclasses.dataclass(frozen=True)
class GameSpec:
    """An OPEN environment: a game form awaiting policies."""
    name: str
    config: Any
    init_fn: Callable[[Key], GraphState]           # key -> state (vmap-safe)
    observe_fn: Callable[[GraphState], Obs]        # state -> per-agent obs, leading axis N
    step_fn: Callable[[GraphState, Actions, Key], GraphState]
                                                   # pure; internally compile_pipeline(...)
    trace_fn: Optional[TraceFn] = None
    metrics: Dict[str, MetricFn] = ...

    def rewards(self, state) -> Array:             # a VIEW of state, default:
        return state.node_attrs["last_reward"]     # payoff-rule mechanisms transform state


def close(game: GameSpec, policy: Policy, params=None) -> EnvSpec:
    """Attach policies to an open game -> today's closed EnvSpec. Backward-compat bridge."""
    def round_fn(state, t, key):
        k_act, k_step = jr.split(key)
        obs = game.observe_fn(state)
        actions = jax.vmap(policy, in_axes=(None, 0, 0))(params, obs,
                                                         jr.split(k_act, n_agents))
        return game.step_fn(state, actions, k_step)
    return EnvSpec(name=game.name, config=game.config, init_fn=game.init_fn,
                   round_fn=round_fn, trace_fn=game.trace_fn, metrics=game.metrics)
```

Design commitments (each argued on the walk):

1. **Parallel (simultaneous-move) API only.** All five scenarios are population games.
   Actions/obs are arrays with a leading agent axis (vmap-friendly), never dict-of-dicts.
2. **Mechanisms stay `Config -> Transform` catalog entries** spliced into `step_fn`'s
   pipeline. Do NOT reintroduce Gym wrappers: wrappers compose opaquely; transforms
   declare reads/writes and the compiler orders them. That property is the moat.
3. **Rewards live in state** (`node_attrs["last_reward"]`); the reward return is a view.
   This is what makes payoff-rule mechanisms (sanctions, taxation) ordinary transforms.
4. **Observation is always a pure view of state.** Information rules (monitoring,
   provenance) are transforms that write observable state fields (e.g. `public_signal`)
   — never wrappers around `observe_fn`. One mechanism type, no second composition system.
5. **No `done`s in v0.** Fixed-horizon benchmark discipline; masking later if needed.

## Ostrom's rule taxonomy = where mechanisms hook

| IAD rule type | Hook | Existing/planned entry |
|---|---|---|
| aggregation | transform writing a policy target from votes | `quota_vote` (rename write field `harvest_target` → generic `policy_target` before A2) |
| payoff | transform on `last_reward` / transfers | `graduated_sanction`, A2 taxation |
| information | transform writing observable state fields | monitoring (backlog) |
| choice | action bound applied in `step_fn`'s apply-actions stage | quota-as-cap (today inside harvest) |
| position | **policy composition** — who occupies the acting seat | `ai_delegate` |
| boundary | activation masks on nodes | A2 actor arrivals |

Mechanism entries should carry a `rule_type` tag (docstring at minimum) so the catalog
reads as an institutional taxonomy, not a grab bag.

## Delegation = policy composition

```python
# agents/delegate.py becomes load-bearing: the default policy that closes the commons.
# principal ∘ delegate ∘ game — a lens between the principal's signal and the game.
delegate = DelegatePolicy(greedy_target, action_noise)
env = close(game, delegate)                        # undefended baseline, literally
```

The future "delegated voting" dial is re-plumbing which composite the aggregation rule
reads from — a game variant, not a new environment.

## The influence instrument (replaces fidelity as the headline)

Two inequivalent quantities; report both. The gap between them IS gradual disempowerment
(a delegate that gives you what you want, in a system where you could get nothing else:
high fidelity, zero empowerment).

- **Exercised influence** (per household j): finite-Δ paired counterfactual —
  rerun the rollout with `pref_j += Δ` (Δ ≈ one preference std), SAME keys, measure
  outcome shifts. N+1 rollouts = one `vmap` over the perturbation index.
  NOT an infinitesimal jacobian: the median aggregator is locally flat (zero marginal
  influence for every non-median voter — a real trap identified on the walk), and
  Bernoulli defection is non-differentiable. Finite-Δ paired runs dodge both.
- **Potential influence** (empowerment proxy): redraw preferences wholesale, measure how
  much outcome variance the preference channel carries vs the noise channels.
- **Influence matrix / spectral thread**: `A[i, j]` = effect of j's preference on i's
  outcome. Collective influence = aggregate of A's human rows/cols; the diagnostics
  thread's operator is this A (or its per-step linearization) — spectral margin
  `1 − ρ(A)`, coupled lock-in when ρ > 1. The boundary makes A generic: one instrument
  in `lab/analysis/`, runs on every scenario, because preferences are explicit inputs.

Keep `influence_fidelity` as a secondary descriptive metric, renamed
(`preference_fidelity`) — it answers "did outcomes match asks," not "did humans govern."

## Migration plan (each step leaves pytest green)

1. `environments/game.py`: `GameSpec` + `close()` (~60 lines) + tests.
2. `governed_commons`: split `make_decide` into `observe_fn` (obs = `[principal_pref,
   alignment]` per household) + the `ai_delegate` policy from the agents catalog;
   `build_governed_commons(mechanisms, **cfg)` becomes `close(build_game(...), delegate)`
   — signature unchanged, all existing tests and the benchmark keep passing.
3. Rename `quota_vote`'s write field to a generic `policy_target`; add the ~20-line
   `validate_composition(init_state, mechanisms)` check (reads/writes metadata already
   exists for this).
4. `lab/analysis/counterfactual.py`: the paired-perturbation influence instrument,
   generic over `GameSpec`; wire exercised/potential influence into the scorecard.
5. Build A2 (`compute_economy`) **against the boundary** as the second consumer; only
   then freeze the interface (APIs generalized from one example generalize wrong).

## Deferred, door open

- Learning-policy runner (params in the scan carry; who owns the update rule — one
  design pass when the learning delegate lands).
- Standard graph observation helper (own attrs + neighbor aggregates) — when A3 forces it.
- Constitutional layer: rule *choice* as a slower-cadence action arena
  (`core.schedule` finally earns its keep) — the adaptive-capacity benchmark, A6-ish.
