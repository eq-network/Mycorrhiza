# Collective Intelligence Library

CI Lib is a JAX-native framework for multi-agent simulation: state is an immutable
pytree (`GraphState`), every step is a pure function (`Transform = GraphState ->
GraphState`), and a simulation compiles to a single `lax.scan` that `vmap`s over
thousands of seeds. You compose transforms instead of writing an update loop, and you
get typed dependency ordering, JIT, and batched sweeps for free.

**Why you might care (by field):**

- *Computational social science:* swap aggregation rules, delegation topologies, or
  election schedules without rewriting the game. Mechanisms compose.
- *Multi-agent AI:* agents are pure functions over a shared `GraphState`; runs are
  batched via `jax.vmap`, making 1000-seed adversarial sweeps feasible on one GPU.
- *Mechanism design:* every pipeline carries declared read/write sets, so
  institutional dependencies are a typed DAG, not a tangle of update hooks.

> ### Status: launch-ready core, evolving catalogs, active research payload
>
> `cilib.core` is the stable promise — build on it without expecting churn. The
> catalogs (`cilib.{agents,transformations,mechanisms,environments,metrics}`) work
> today and grow by addition, but are thin and their shapes may still move.
> `cilib.lab` is the maintainer's research code behind in-progress papers — real,
> tested, not held to library stability. The table in
> [Where this fits](#where-this-fits) says which promise applies where.

## Installation

```bash
git clone https://github.com/eq-network/Collective-Intelligence-Library.git
cd Collective-Intelligence-Library
pip install -e .            # editable install; `import cilib` now resolves
# with JAX CUDA 12:
pip install -e ".[cuda]"
```

The library installs as `cilib` (distribution name `collective-intelligence-library`).

**Requirements**: Python 3.10+, JAX 0.4.20+, NumPy, matplotlib. GPU optional (via `jax[cuda12]`).

## Quick example

```python
import jax
import jax.numpy as jnp
from cilib.core import GraphState, transform, run_scan

state = GraphState(
    node_types=jnp.zeros(5, dtype=jnp.int32),
    node_attrs={"score": jnp.ones(5)},      # per-agent, evolving
    adj_matrices={},
    global_attrs={},
)

@transform(reads=["score"], writes=["score"])   # declared effects -> the compiler can order it
def decay(state):
    return state.update_node_attrs("score", state.node_attrs["score"] * 0.9)

final, trace = run_scan(lambda s, t, key: decay(s), state, n_steps=50,
                        key=jax.random.PRNGKey(0),
                        trace_fn=lambda s: s.node_attrs["score"].mean())
print(final.node_attrs["score"])                # decayed toward 0, in one lax.scan
```

**Start with [`examples/`](examples/)** — three short scripts, one idea each, on one
growing toy domain:

1. [`01_first_transform.py`](examples/01_first_transform.py) — `GraphState` + one
   `@transform` + `run_scan`; watch a commons collapse.
2. [`02_typed_pipeline.py`](examples/02_typed_pipeline.py) — four transforms;
   `compile_pipeline` derives the execution order and finds the parallel batch itself.
3. [`03_vmap_seeds_and_learning.py`](examples/03_vmap_seeds_and_learning.py) —
   per-agent Q-learning, 200 seeds as one `vmap(lax.scan)` program.

## Where this fits

| Ring | Package | Promise | The PR test |
|---|---|---|---|
| 0 — engine | `cilib.core` | **Stable.** `GraphState`, `Transform`, `@transform`, `compile_pipeline`, `run_scan`/`run_scan_batch` won't break under you without a changelog entry. | Would we review a PR here like a library API change? Yes. |
| 1 — catalogs | `cilib.{agents, transformations, mechanisms, environments, metrics}` | **Works, evolving.** Entries are real and tested; catalogs are thin and grow by addition, so names/signatures near the edges may still shift. | Would we merge a stranger's new catalog entry? Yes — that's the point, see [EXTENDING.md](EXTENDING.md). |
| 2 — lab | `cilib.lab.{paradigms, analysis}` | **Research payload, no stability promise.** The code behind specific papers (polycentric causal emergence, active inference). Read before depending on it. | Would we maintain a stranger's PR here like a library? No — we'd point them at `experiments/`. |

## Repository structure

```
collective-intelligence-library/
├── src/cilib/
│   ├── core/                 Ring 0 — the engine (GraphState, @transform,
│   │                         compile_pipeline, scan/vmap runners, schedule, protocols)
│   ├── agents/  transformations/  mechanisms/
│   │   environments/  metrics/                  Ring 1 — catalogs (plain REGISTRY dicts)
│   └── lab/
│       ├── paradigms/        Ring 2 — composed research models (active_inference, polycentric)
│       └── analysis/         Ring 2 — offline research math (effective information,
│                             causal emergence, bootstrap CIs)
├── examples/                 START HERE — imports ring 0/1 only, never cilib.lab
├── experiments/              research studies (config/run/figures/README contract);
│   │                         may import cilib.lab
│   ├── basin_stability/         PDD/PRD/PLD democracy under adversarial pressure
│   ├── polycentric_emergence/   governance as causal emergence (paper study)
│   └── governed_harvest/        earlier harvest prototype
├── docs/                     Manifesto (the "why") + dated changelogs
└── ARCHITECTURE.md  EXTENDING.md  CONTRIBUTING.md  CLAUDE.md  LICENSE
```

Each catalog's contents are a plain `REGISTRY` dict in its `__init__.py`. To build a
simulation you select catalog entries, compile a pipeline, and sweep it — see
[EXTENDING.md](EXTENDING.md) for the recipes and `experiments/_template/` for the shape.

## The functional imperatives

These aren't style preferences — the scan/vmap execution model breaks *silently*, not
loudly, if you violate them.

1. **State lives in `GraphState`, never in objects.** Agents are pure factories; stash
   evolving data on `self` and it won't be there when JAX retraces you.
2. **Transforms are pure — `GraphState -> GraphState`, no mutation.** Composition
   (`sequential`, `compile_pipeline`) assumes it; an in-place mutation breaks under
   `jit`/`vmap` without raising.
3. **Static config is closed over by factories, never stored in `global_attrs`.**
   Swept or per-step data in `global_attrs` forces silent recompiles or freezes a value
   at trace time.
4. **No data-dependent Python `if` inside a transform — use `jnp.where`/`lax.cond`.**
   A traced value can't drive Python control flow; the branch taken at trace time gets
   locked in.
5. **Declare `.reads`/`.writes` via `@transform`.** `compile_pipeline` derives execution
   order from these sets; an undeclared transform is invisible to the compiler.

The full convention list (for anyone extending the engine itself) is in
[CLAUDE.md](CLAUDE.md).

## Research built on this engine

The stable/evolving/research split above isn't a way of saying the research is
second-class — it's the opposite: these studies are the reason the engine has the
shape it has.

- **Polycentric causal emergence** (`cilib.lab.paradigms.polycentric`,
  [`experiments/polycentric_emergence/`](experiments/polycentric_emergence/)) —
  governance regimes re-described as causal emergence over a common-pool resource
  game, measured with effective information against honest surrogate nulls.
- **Basin stability of democratic mechanisms**
  ([`experiments/basin_stability/`](experiments/basin_stability/)) — direct,
  representative, and liquid democracy (PDD/PRD/PLD) with Q-learning agents under
  adversarial pressure, measured as survival probability across a 500-seed sweep.

These live outside the stable catalogs because they move at the pace of a thesis
chapter, not a library release. If you want to see the engine under real research
load, read one after `examples/`.

## Documentation

| Doc | Purpose |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | The pattern map — read this first |
| [EXTENDING.md](EXTENDING.md) | How to add a building block (agent, transform, mechanism, …) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution path, house style, doc cadence, contact |
| [CLAUDE.md](CLAUDE.md) | Working conventions (for humans and coding agents) |
| [docs/Manifesto.md](docs/Manifesto.md) | The "why": process-centric thinking + category theory framing |
| [docs/changelogs/](docs/changelogs/) | Dated engineering notes + typeset roll-ups |
| [Excalidraw diagrams](https://excalidraw.com/#room=f4116b0ba2d8d5095d85,zSDwGDuqMZI4uxu4CTQuHg) | Live visual architecture overview (external whiteboard) |

## Contributing

New building blocks are one file plus one `REGISTRY` line — see
[CONTRIBUTING.md](CONTRIBUTING.md). Interested in the ideas or collaborating?
Reach out: [Jonas Hallgren](https://github.com/spiralling) · Uppsala University ·
`jonas@eq-network.org`.

## License

[MIT](LICENSE)
