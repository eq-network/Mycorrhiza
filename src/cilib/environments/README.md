# Environments catalog

Reusable, governance-agnostic simulation substrates. Pick one by name from `REGISTRY`
(in `__init__.py`); `make_env` / `list_envs` are thin conveniences over it.

**Type function:** `EnvFactory = (**cfg) -> EnvSpec` (see `spec.py`). An `EnvSpec` is a
fully-specified, runnable environment: `config`, `init_fn`, `round_fn`, `trace_fn`,
`metrics`, with `run` / `run_batch` / `evaluate` helpers.

**Entries:** `commons_harvest` (spatial tragedy-of-the-commons, Melting-Pot idiom),
`governed_commons` (non-spatial aggregate-stock commons with AI-delegate households —
alpha scenario 1), `compute_economy` (CES production, scheduled AI arrivals, labor-share
decay — alpha scenario 2; first entry of the three-model economy register — see
`docs/model-register-design.md` and the colocated `ASSUMPTIONS.md` card),
`io_economy` (Leontief recipe network, AI-by-recipe-rewiring, demand-attribution
share — register entry R2, the σ=0 bracket; card colocated), `task_economy`
(task-frontier automation with endogenous adoption — register flagship R3, skeleton;
card colocated), `value_contagion` (culture as contagion on a homophily-dialed
friendship network — cultural register entry C2, the (S, P) phase-diagram model;
card colocated; first entry with an agent-agent adjacency). These take a `mechanisms=` kwarg of
`cilib.mechanisms` Transforms (schedule-wrapped as needed) and also expose an OPEN
`build_game(...) -> GameSpec` (see `game.py`: `close(game, policy)` plugs any policy in).
Shared helpers: `networks.py` (graph generators, plus `to_sparse`/`sparse_nse_bound`/
`row_sums` for storing an adjacency as a BCOO), `counterfactual.py` (causal influence
instruments), `commons_metrics.py` (GovSim suite).

**Sparse adjacencies.** A dense `(N, N)` matrix costs O(N²) in every `W @ x` no matter
how empty it is — XLA cannot infer sparsity from a dense array. Environments with a
*static* network can store it as `jax.experimental.sparse.BCOO` instead;
`value_contagion` does, behind `sparse_friendship=True`. It is a representation swap,
not a model change (`test_sparse_equivalence.py` pins the trajectories bit-for-bit).
Worth it above N ~ 2-3k: memory drops immediately (~300x at N=6000, mean degree 6),
wall-clock only crosses over once the N² term clears the per-call overhead floor —
`examples/07_sparse_scaling.py` measures both. Networks that *rewire* (e.g.
`influence_exchange`) are not candidates: preferential-attachment drift densifies them
regardless of how sparse they start, so they need a degree cap first.

```python
from cilib.environments import make_env, list_envs
env = make_env("commons_harvest", n_agents=16, grid=(18, 18))
finals, traces = env.run_batch(jr.PRNGKey(0), n_seeds=64, n_steps=1000)
scores = env.evaluate(traces)        # GovSim metric suite
```

**Add one:**
1. Implement the `EnvSpec` fields in a subpackage (config / state / dynamics / metrics).
2. Define a `build_<name>(**cfg) -> EnvSpec` builder; import it in `__init__.py`.
3. Add one line to `REGISTRY`: `"<name>": build_<name>`; add a substrate test.
