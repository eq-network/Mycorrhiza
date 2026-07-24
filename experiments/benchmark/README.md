# benchmark — the multi-scenario benchmark harness (alpha A1+)

Runs defense portfolios against each scenario's undefended baseline and scores them on
**influence preserved** — how much of the humans' lost *causal* influence over outcomes
a defense recovers. The headline is counterfactual, never correlational: paired
same-key rollouts with a perturbed human-preference channel, instrument defined
per scenario in [`scenarios.py`](scenarios.py). Each scorecard row is one
(scenario × portfolio) leaderboard entry.

## Run

```bash
python -m experiments.benchmark.run                                # commons (32 x 200)
python -m experiments.benchmark.run --scenario compute_economy     # economy (32 x 300)
python -m experiments.benchmark.run --smoke                        # tiny sanity run
```

Prints the scorecard and writes `results/scorecard_<scenario>_<timestamp>.json`, with
bootstrap CIs on every metric and every condition run on the same seeds as its baseline.

## Scenarios & conditions

**governed_commons** (scenario 1): `baseline` / `quota_voting` (median vote → policy
target, every 5 ticks) / `graduated_sanctions` (+ over-quota penalty & confiscation).
Influence instrument: collective ask-shift from t=0, outcome = per-capita harvest.

**compute_economy** (scenario 2): `baseline` / `ai_revenue_tax` (50% on capital income,
redistributed — switching on at t=50) / `tax_and_ownership_cap` (+ 35% cap on any
actor's share of AI compute). Influence instrument: **one-shot work-preference shift at
2T/3** (influence *now*, after AI capital entrenches), outcome = late log-output — its
responsiveness is the output elasticity of human labor ("how much the economy still
needs people").

Conditions are **(mechanism, config, schedule) triples**: mechanisms are pure catalog
rules; `ScheduleSpec` owns cadence / phase / onset (`onset` is the regime-shift dial).

## Adding a defense or a scenario

- Defense: catalog the mechanism per [EXTENDING.md](../../EXTENDING.md), then add one
  triple to the scenario's conditions dict in `scenarios.py`.
- Scenario: build the environment (EXTENDING.md), define its influence instrument
  (perturbation + outcome), add one `ScenarioSpec` to `SCENARIOS`.

## Headline results (2026-07-14)

**governed_commons** (32 × 200): baseline 0.00 / quota-only **−0.07** / +sanctions
**0.91** influence preserved. The correlational fidelity score (0.49/0.63) hides that
unenforced quota voting restores almost no causal influence — at the sustainability
knife-edge the ecology governs, not the vote; enforcement is what makes the vote
channel causally effective.

**compute_economy** (32 × 300): baseline 0.00 / tax **0.69** / tax+cap **0.73**
influence preserved (exercised: 0.34 / 0.79 / 0.83). Two counterfactual readings
matter here: influence-from-birth stays ≈1.0 in ALL regimes (early labor is upstream
of the AI capital stock itself), while influence-*now* collapses undefended — **the gap
between historical and current influence is the gradual-disempowerment signature.**
Descriptive indicators move with it: labor share 0.33→0.63, human income share
0.33→0.81, income Gini 0.69→0.11 across the defense portfolio.

Caveats, stated as loudly as the numbers: these are toy models — calibrations chosen so
each dynamic appears clearly (`docs/abm-suite-design.md` has the validation ladder);
scores are candidate indicators, not measurements of the world. Agents are deliberately
classical (fixed rules, no learning): mechanisms are tested for structural effects, not
robustness to strategic adaptation — that's the roadmap, not the claim.
