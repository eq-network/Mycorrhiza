"""Which timescale structures are stable? Sweep per-domain cadences over
{1,2,4}^3 at kappa=0.8, defended and undefended; report the composite human
share, the correlated-decline index, and a stability proxy (late-window std of
the per-tick composite)."""
import itertools

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from cilib.core.schedule import scheduled
from cilib.environments import make_env
from cilib.mechanisms import (
    EnforcedAITaxConfig, InfluenceCapConfig, SortitionConfig,
    make_enforced_ai_tax, make_influence_cap, make_sortition,
)

KEY = jr.PRNGKey(0)
T, SEEDS = 400, 4


def defenses():
    return (scheduled(make_enforced_ai_tax(EnforcedAITaxConfig()), onset=50),
            scheduled(make_sortition(SortitionConfig()), cadence=15),
            make_influence_cap(InfluenceCapConfig()))


def composite_series(trace, H=20):
    labor = (trace["wage"] * trace["labor_supply"].sum(-1)) / np.maximum(trace["output"], 1e-6)
    culture = (1.0 - trace["culture"][..., :H]).mean(-1)
    influence = trace["influence"][..., :H].sum(-1)
    return (np.clip(labor, 0, 1) + culture + influence) / 3.0   # (seeds, T)


rows = []
for ec, cc, pc in itertools.product([1, 2, 4], repeat=3):
    for label, mechs in [("undef", ()), ("def", defenses())]:
        env = make_env("coupled_society", mechanisms=mechs,
                       econ_cadence=ec, culture_cadence=cc, politics_cadence=pc)
        _, tr = env.run_batch(KEY, SEEDS, T)
        tr = {k: np.asarray(v) for k, v in tr.items()}
        comp = composite_series(tr)
        late = comp[:, 3 * T // 4:]
        corr = float(jnp.mean(jax.vmap(env.metrics["correlated_decline"])(
            {k: jnp.asarray(v) for k, v in tr.items()})))
        rows.append((ec, cc, pc, label, float(late.mean()), float(late.std(axis=1).mean()), corr))
        print(f"econ {ec} culture {cc} politics {pc} {label:5s} "
              f"composite {rows[-1][4]:.3f}  lateStd {rows[-1][5]:.4f}  corrDecline {rows[-1][6]:.2f}")

print("\nmost stable defended (low std, high composite):")
for r in sorted([r for r in rows if r[3] == "def"], key=lambda r: r[5])[:5]:
    print(f"  cadences ({r[0]},{r[1]},{r[2]}): composite {r[4]:.3f}, std {r[5]:.4f}")
print("least stable overall:")
for r in sorted(rows, key=lambda r: -r[5])[:5]:
    print(f"  cadences ({r[0]},{r[1]},{r[2]}) {r[3]}: composite {r[4]:.3f}, std {r[5]:.4f}")
