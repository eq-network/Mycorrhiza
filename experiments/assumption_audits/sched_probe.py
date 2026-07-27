import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.environments import make_env

KEY = jr.PRNGKey(0)
for name, kw in [("lockstep", {}), ("econ cad 2", {"econ_cadence": 2}),
                 ("econ cad 3", {"econ_cadence": 3}),
                 ("politics cad 3", {"politics_cadence": 3}),
                 ("culture cad 3", {"culture_cadence": 3})]:
    env = make_env("coupled_society", **kw)
    _, tr = env.run_batch(KEY, 6, 400)
    m = {k: float(jnp.mean(jax.vmap(f)(tr))) for k, f in env.metrics.items()}
    print(f"{name:14s} labor {m['labor_share']:.3f}  culture {m['human_culture_share']:.3f}"
          f"  influence {m['human_influence_share']:.3f}  composite {m['composite_human_share']:.3f}")
