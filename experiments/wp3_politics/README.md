# WP3 — political disempowerment: delegation, taxation, and the takeover threshold

The paper's substrate is `delegative_polity` (see its `ASSUMPTIONS.md`): citizens
delegate ballots to humans or AI delegates, the power-weighted median sets a tax
rate, preferential attachment fights re-delegation churn, and an off-by-default
lock-in dial lets concentrated power erode enforcement and re-delegation freedom.

**Hypothesis.** Delegation capture has a threshold: below the pre-registered
a*(churn) (the mean-field saddle-node; `1 + churn/update_rate` at gamma=1) the
polity tracks its median voter indefinitely; above it the AI bloc takes the
weighted median and policy detaches. Lock-in converts capture from a state into
a trap (hysteresis under advantage shut-off).

| Sweep | Axes | Headline readouts |
|---|---|---|
| headline | organic / captured / locked series | human power share, policy vs median, enforcement |
| E1 knee | ai_advantage x churn | human_power_share, policy_median_gap, a* overlay |
| E2a shut-off | entrenchment_gain, advantage off at t=200 | recovery_gap, share trajectories |
| E2b responsiveness | organic / captured / locked | d(effective rate)/d(citizen ideals), birth vs now |
| E3 defenses | sortition cadence x cap share x onset | human_power_share, decision_quality |
| E4 floors | cumulative floor removal (lock-in, AI ballots, franchise) | share decomposition, crash to zero |

One role per file: `config.py` freezes every parameter and the committed
prediction; `run.py` sweeps -> `results.json` (bootstrap CIs over the shared
seed batch); `figures.py` reads `results.json` and writes the paper's PDFs into
the vault — read, plot, never recompute. Design-failure check: E1 must show BOTH
regimes or the grid is revised before any interpretation.

    python -m experiments.wp3_politics.run
    python -m experiments.wp3_politics.figures
