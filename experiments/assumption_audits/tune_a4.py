"""Defense-constant sweep for influence_exchange: find (cap_share,
sortition share, cadence) where the four conditions separate cleanly."""
import itertools

import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.core.schedule import scheduled
from cilib.environments import make_env
from cilib.mechanisms import (
    InfluenceCapConfig, SortitionConfig, make_influence_cap, make_sortition,
)

KEY = jr.PRNGKey(0)


def share_and_err(env):
    _, tr = env.run_batch(KEY, n_seeds=6, n_steps=400)
    s = float(jnp.mean(jax.vmap(env.metrics["human_influence_share"])(tr)))
    e = float(jnp.mean(jax.vmap(env.metrics["consensus_error"])(tr)))
    return s, e


s_org, e_org = share_and_err(make_env("influence_exchange", amp_onset=10_000))
s_amp, e_amp = share_and_err(make_env("influence_exchange"))
s_dis, e_dis = share_and_err(make_env("influence_exchange", update_rate=0.0, amp_onset=10_000))
print(f"organic   share {s_org:.3f}  err {e_org:.3f}")
print(f"amplified share {s_amp:.3f}  err {e_amp:.3f}")
print(f"dispersed share {s_dis:.3f}  err {e_dis:.3f}   (wisdom margin {e_amp - e_dis:.3f})")
print()

for cap, sshare, cad in itertools.product([0.04, 0.06, 0.08], [0.35, 0.5], [10, 15, 25]):
    mechs = (scheduled(make_sortition(SortitionConfig(share=sshare)), cadence=cad),
             make_influence_cap(InfluenceCapConfig(cap_share=cap)))
    s_def, e_def = share_and_err(make_env("influence_exchange", mechanisms=mechs))
    ok = "OK " if s_def > s_amp + 0.15 else "   "
    print(f"{ok}cap {cap:.2f} share {sshare:.2f} cadence {cad:2d} -> defended {s_def:.3f}  err {e_def:.3f}")
