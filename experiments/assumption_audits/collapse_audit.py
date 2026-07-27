"""Collapse audit: are the coupled model's floors structural, or manufactured
by hand-set constants? Push each suspected floor-setter in the coupled
undefended world and see whether full collapse becomes reachable."""
import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.environments import make_env

KEY = jr.PRNGKey(0)


def probe(name, T=500, **kw):
    env = make_env("coupled_society", **kw)
    _, tr = env.run_batch(KEY, 6, T)
    m = {k: float(jnp.mean(jax.vmap(f)(tr))) for k, f in env.metrics.items()}
    print(f"{name:42s} income {m['human_income_share']:.3f}  culture {m['human_culture_share']:.3f}"
          f"  influence {m['human_influence_share']:.3f}  composite {m['composite_human_share']:.3f}")
    return m


print("--- reference ---")
probe("coupled undefended (defaults, T=500)")
probe("same, T=1500 (plateau or still sliding?)", T=1500)

print("--- culture floor: constant native recovery ---")
probe("recovery 0.15 -> 0.05", recovery=0.05)
probe("recovery 0.15 -> 0.0", recovery=0.0)

print("--- influence floor: self-weight anchor ---")
probe("self_weight 0.15 -> 0.05", self_weight=0.05)
probe("self_weight 0.15 -> 0.01", self_weight=0.01)

print("--- consensus floor: FJ susceptibility ---")
probe("susceptibility 0.7 -> 0.95", susceptibility=0.95)

print("--- economy: max the loop ---")
probe("kappa 1.0 + capture_gain 1.0", kappa=1.0, capture_gain=1.0)

print("--- everything at once (is full collapse reachable?) ---")
probe("kappa 1, recovery .02, sw .03, lambda .95",
      kappa=1.0, recovery=0.02, self_weight=0.03, susceptibility=0.95,
      capture_gain=1.0, amplification=4.0)
probe("same, T=1500", T=1500,
      kappa=1.0, recovery=0.02, self_weight=0.03, susceptibility=0.95,
      capture_gain=1.0, amplification=4.0)
