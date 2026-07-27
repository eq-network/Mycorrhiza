"""Post-flywheel numbers: four conditions + transfer gaps, income-share leg."""
import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.core.schedule import scheduled
from cilib.environments import make_env
from cilib.environments.coupled_society import defense_transfer_gap
from cilib.mechanisms import (
    EnforcedAITaxConfig, InfluenceCapConfig, SortitionConfig,
    make_enforced_ai_tax, make_influence_cap, make_sortition,
)

KEY = jr.PRNGKey(0)
T, SEEDS = 500, 8


def defenses():
    return (scheduled(make_enforced_ai_tax(EnforcedAITaxConfig()), onset=50),
            scheduled(make_sortition(SortitionConfig()), cadence=15),
            make_influence_cap(InfluenceCapConfig()))


def report(name, env):
    _, tr = env.run_batch(KEY, SEEDS, T)
    m = {k: float(jnp.mean(jax.vmap(fn)(tr))) for k, fn in env.metrics.items()}
    print(f"{name:22s} income {m['human_income_share']:.3f}  (factor {m['labor_share']:.3f})"
          f"  culture {m['human_culture_share']:.3f}  influence {m['human_influence_share']:.3f}"
          f"  composite {m['composite_human_share']:.3f}  corr {m['correlated_decline']:.2f}")
    return m


report("sealed undefended", make_env("coupled_society", kappa=0.0))
report("sealed defended", make_env("coupled_society", kappa=0.0, mechanisms=defenses()))
report("coupled undefended", make_env("coupled_society"))
report("coupled defended", make_env("coupled_society", mechanisms=defenses()))

for label, mechs in [("undefended", ()), ("defended", defenses())]:
    gap = defense_transfer_gap(
        make_env("coupled_society", mechanisms=mechs),
        make_env("coupled_society", mechanisms=mechs, kappa=0.0),
        KEY, SEEDS, T)
    print(f"transfer gap ({label}): {float(gap):.3f}")
