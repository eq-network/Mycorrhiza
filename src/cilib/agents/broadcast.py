"""
Broadcast policy — constant per-agent transmission effort.

``value_contagion`` (cultural register C2) keeps its agent boundary open the
same way the commons does: the per-tick decision is how loudly you broadcast
your current cultural variant — a scalar effort in [0, 1] scaling your
transmissibility *as a source*. Effort 1.0 recovers the classic contagion count
rule exactly; effort 0.0 is a silent population (culture frozen — the boundary
test). C4's strategic persuaders later modulate this same channel; that is the
planned ``close_multi`` seam, not this catalog entry.
"""
import jax.numpy as jnp


class BroadcastPolicy:
    """Constant broadcast effort, ignoring observations.

    ``obs = [own_culture, local_ai_exposure_share]`` is exposed for future
    strategic variants; this default rule uses none of it.
    """

    def __init__(self, effort: float = 1.0):
        self.effort = effort

    def __call__(self, obs: jnp.ndarray, key) -> jnp.ndarray:
        return jnp.asarray(self.effort, dtype=jnp.float32)
