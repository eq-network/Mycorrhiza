# wp2_culture — the culture model spec's runs and figures

**Hypothesis.** When machine voices receive algorithmic amplification, the
share of every citizen's settled percepts attributable to human sources
falls toward the anchor floor `1 − λ`, and the human–AI boundary becomes the
sharpest structure in the listening graph's mode ladder.

**Model.** `cilib.environments.influence_exchange` (alpha scenario A4), used
unchanged — this experiment adds only offline instruments (`instruments.py`):
the percept-attribution solve, per-mode Laplacian alignments, and a spring
layout for the figures.

**Finding worth knowing before reading fig3:** the human–AI split locks onto
the *top* of the mode ladder (mode 33/33, match 1.00 within ~30 days of the
amplification onset), not the bottom. The machines do not become a separate
community (that would be the lowest modes) — they become the hub everyone
orbits. Reported as measured; not retuned.

| Artifact | Figure (semantic names, wp1 convention) | Paper placement |
|---|---|---|
| `results.json` `headline` | `dynamics.pdf` (share over time, amplification on/off, CIs) | main body |
| `results.json` `exemplar` | `field.pdf` colors, checkpoints; power ranking (stdout only, no paper table) | main body |
| `results.json` `dial_sweeps` | `dials.pdf` (amplification 1–32 concave/saturating; drift a switch at zero) | appendix |
| `results.json` `floor_sweep` | `floor.pdf` (share follows `(1-λ)/(1-λs)` to zero at λ=1) | appendix |
| `snapshots.npz` (41 W frames) | `graph.pdf`, `field.pdf` edges, `modes.pdf` mode ladder | graph/field main body; modes appendix |

Discipline: graph, field, and dynamics are illustrations at the config
defaults — the main body's "here is the model" material. Everything
measured (dial ranges, floor sweep, the mode-geometry finding) lives in the
paper's appendices, with the defaults appearing as marked points.

The floor sweep is the honesty instrument: susceptibility × self-attention at
32× amplification. Measured shares track the floor within one percentage
point, and at λ=1 the share is exactly zero — with no anchors, humans are
relays, not sources. The spec's protective assumptions are A2 (anchor), A3
(fixed self-attention), A7 (threat cannot grow with success).

## Coupling-grammar contract (`docs/ledger-design.md`, `docs/gd-suite-v0.1.md`)

Sorted by the grammar's kinds — ledger / port / rate layer / internal —
with cross-domain channels typed by the four-type catalog (flow,
modulation, capture, rewiring):

| quantity | kind | contract |
|---|---|---|
| attention (rows of `listening`) | **ledger** | row-stochastic every step; `rewire` reallocates, never mints; no sources/sinks |
| self-attention (diagonal) | reserved ledger slice | fixed 0.15; unreachable by the drift |
| percept `opinion` | internal field | anchored averaging; bounded; nothing spent when it moves; not readable across domains |
| anchor `signal` | internal fixed stock | written by nothing |
| `influence` | **port** (in-engine reduction) | long-run attention weight; sums to 1 |
| `amplification` | rate layer (institutional residue) | clock-written; see seam below |
| `B`, `power`, human share | **candidate ports** (offline, this experiment) | B rows sum to 1 (asserted); power sums to n_citizens; share ≥ analytic floor |

**The inbound seam, typed honestly:** amplification is today a *scheduled
modulation* with a hardcoded beneficiary — two declared deviations from the
grammar (channels should be funded and value-agnostic; a human voice here
cannot buy the machines' reach). The ledger rewrite replaces it with a
*funded rewiring*: reach bought from the money ledger by whoever spends,
entering `rewire`'s attractiveness as bought attention. That change deletes
spec assumption A7, and fig5 prices what follows.

**Outbound:** the ports above are the whole surface (the port rule — nothing
else is legitimately readable across a boundary). Candidate ports get
promoted to in-engine reductions the day another domain reads them. Belief
is internal: culture is an attribution over the attention flow, not a
transferable stock. The attention object read as delegation is
`delegative_polity`'s kernel — the culture→politics channel is the shared
kernel, not a new transform.

Commands (repo root):

```bash
python -m experiments.wp2_culture.run          # full: T=400, 8 paired seeds x 2 amplifications
python -m experiments.wp2_culture.run --smoke  # tiny, same code path; OVERWRITES outputs
python -m experiments.wp2_culture.figures      # 4 PDFs -> vault papers/wp2-culture/figures/
```

Smoke overwrites `results.json`/`snapshots.npz` — rerun the full config before
regenerating figures. The spec document lives in the Obsidian vault
(`Research/Projects/CI Library/papers/wp2-culture/`); the repo stays code-only.
