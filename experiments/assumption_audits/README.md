# Assumption audits

One-shot receipt scripts behind the "we checked" claims on eq-network.org/lab and the
playground's per-tab modelling-assumptions notes. Deliberately **not** the `_template`
shape (config/run/figures): these are probes, not sweep studies — each answers one
question and prints its numbers. Run from the repo root:
`python -m experiments.assumption_audits.<script>`.

Rescued from the 2026-07-27 session scratchpad that produced the lab-content review
(eq-network repo, `docs/lab-content-review-2026-07-27.md`) and the cold-read panel
(`cold-read/lab-assumptions-cards-2026-07-27/`). Scripts are kept verbatim as run that
day; the numbers column records what they printed then. If coupled/political/cultural
dynamics change, re-run and update the review doc's floor table **and** the website
copy in the same commit (review rule 6).

| Script | Question | Headline numbers (2026-07-27) | Backs |
|---|---|---|---|
| `collapse_audit.py` | Are the coupled model's floors structural or manufactured? | culture share 0.54–0.59 at defaults → **0.008** at `recovery=0` (manufactured: constant native reversion); influence floor **~0.45–0.50** insensitive to `self_weight` 0.15→0.01 and amplification 8 (structural: frozen AI listening rows); income share **0.07** (T=500) → **0.018** (T=1500, all-out) — no floor | every per-tab floor line |
| `tune_a4.py` | Do the influence_exchange defense constants separate the four conditions? | human share 0.87 organic / 0.54 amplified / ~0.74 defended, over a (cap, sortition share, cadence) grid | political tab anchors; "swept, not asserted" |
| `schedule_scan.py` | Which per-domain timescale structures are stable? | cadences {1,2,4}³ at κ=0.8, defended + undefended: composite, late-window std, correlated decline per cell | combined tab robustness claims |
| `sched_probe.py` | Quick cadence probes (scouting for `schedule_scan`) | econ/politics/culture cadence 2–3 vs lockstep | — |
| `a5_numbers.py` | Coupled headline numbers, pre-flywheel (3 couplings) | four conditions + transfer gaps | superseded by `a5_numbers2` |
| `a5_numbers2.py` | Post-flywheel headline numbers (5 couplings), income-share leg | income 0.12 → 0.07 coupled-undefended; transfer gap 0.128 defended / 0.109 undefended; corr-decline −0.02 → 0.60 | combined tab `anchorsNote`; lab.ts engineNote |
