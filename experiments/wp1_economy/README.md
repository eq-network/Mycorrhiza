# WP1 experiments — the knee, the decoupling, the defenses

The sweep suite behind `papers/wp1-economy/` (referee-gated spec; predictions
committed before these runs — `results.json` records the pre-registered e*
band next to E1).

| Sweep | Axis | Backs |
|---|---|---|
| E1 | efficiency × reinvest | Prop 1 (survival threshold; both regimes shown) |
| E2 | recycled share r | Prop Decouple (concentration robust, output closure-dependent) |
| E3 | τ × ω × fund design | Prop 4 (eradication band; dividend split vs mirror ratchet) |

```bash
python -m experiments.wp1_economy.run      # -> results.json (bootstrap CIs, 8 seeds)
python -m experiments.wp1_economy.figures  # -> ../../papers/wp1-economy/figures/*.pdf
```

Headline numbers land in the paper's results section; anything that contradicts
a committed prediction is reported as a finding (two examples already recorded:
the inventory-stock invariant and the two-fund-designs correction; a third
candidate from E2 — the output minimum is *interior* in r, because extreme
stalling starves the automation itself).
