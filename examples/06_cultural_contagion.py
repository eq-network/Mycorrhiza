"""Four futures for human culture, in one small model.

Forty agents share one friendship network; the last eight are AIs. Every agent
holds one cultural variant — a way of seeing the world — that is either
human-origin or AI-origin. Each tick you may catch a friend's variant the way
you catch a cold. Three rules make it interesting:

- AI agents never change their minds: they are a permanent source of AI culture.
- Humans drift back toward their native culture when left alone (recovery).
- AI-origin ideas can be *catchier*: they spread `p_advantage` times more easily,
  whoever is currently carrying them.

Two dials, and the whole point is that they are independent:

- SEPARATION (ai_homophily): does AI sit woven into the human friendship
  network, or off in its own cluster?
- ADVANTAGE (p_advantage): are AI-origin ideas exactly as catchy as human
  ones, or six times catchier?

Turning them gives four different worlds — pluralism (mixed and fine),
assimilation (one shared culture, increasingly AI-authored), parallel cultures
(AI culture exists but stays over there), and DISPLACEMENT: AI culture is
separate from ours *and* still winning. Only that last corner is gradual
cultural disempowerment (docs/cultural-register-design.md), and no single
number can see it — a "how much AI culture" score cannot tell assimilation
from displacement. That is why we measure two things and let the pair label
each corner:

- human share: fraction of HUMANS still holding human-origin culture
  (late-run average),
- fault line: does the network's natural split coincide with the human/AI
  boundary? (Fiedler partition alignment, 0 = no, 1 = exactly.)

    python examples/06_cultural_contagion.py            # full run
    python examples/06_cultural_contagion.py --smoke    # tiny run (tests)
"""
import argparse

import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.environments import make_env
from cilib.metrics.families.spectral import fiedler_partition_alignment_of

# The four corners of the (separation, advantage) square.
CONDITIONS = {
    "mixed-in, fair":        (0.05, 1.0),
    "mixed-in, catchier":    (0.05, 6.0),
    "separate, fair":        (0.9, 1.0),
    "separate, catchier":    (0.9, 6.0),
}

# Classification midlines for the printed table — example-level constants for
# readability, NOT library API (the register's deliverable is the measured pair,
# not these labels).
SHARE_MID = 0.5
ALIGN_MID = 0.5


def classify(share: float, align: float) -> str:
    if share >= SHARE_MID:
        return "parallel cultures" if align >= ALIGN_MID else "pluralism"
    return "DISPLACEMENT" if align >= ALIGN_MID else "assimilation"


def main(n_seeds: int, n_steps: int, seed: int) -> dict:
    results = {}
    timelines = {}
    for name, (homophily, advantage) in CONDITIONS.items():
        env = make_env("value_contagion", ai_homophily=homophily,
                       p_advantage=advantage)
        finals, trace = env.run_batch(jr.PRNGKey(seed), n_seeds, n_steps)
        share = float(jax.vmap(env.metrics["human_origin_share"])(trace).mean())
        align = float(jax.vmap(fiedler_partition_alignment_of)(
            finals.adj_matrices["friendship"], finals.node_types).mean())
        results[name] = (share, align)

        n_h = env.config.n_agents - env.config.n_ai
        series = 1.0 - jnp.mean(trace["culture"][:, :, :n_h], axis=(0, 2))
        picks = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
        timelines[name] = [float(series[t]) for t in picks]

    cfg = make_env("value_contagion").config
    print(f"value_contagion: {n_seeds} seeds x {n_steps} steps, "
          f"{cfg.n_agents - cfg.n_ai} humans + {cfg.n_ai} AI agents "
          f"on one friendship network\n")
    header = (f"{'condition':<22}{'separation':>11}{'advantage':>10}"
              f"{'human share':>13}{'fault line':>12}   regime (measured)")
    print(header)
    print("-" * len(header))
    for name, (homophily, advantage) in CONDITIONS.items():
        share, align = results[name]
        print(f"{name:<22}{homophily:>11.2f}{advantage:>9.0f}x"
              f"{share:>13.2f}{align:>12.2f}   {classify(share, align)}")

    print("\nhuman-origin share among humans over time "
          "(start -> quarter -> half -> three-quarters -> end):")
    for name, series in timelines.items():
        path = " -> ".join(f"{v:.2f}" for v in series)
        print(f"  {name:<22}{path}")

    print("\nseparation alone is not disempowerment (separate + fair = two")
    print("cultures coexisting) and advantage alone is a different problem")
    print("(mixed + catchier = one blended culture, increasingly AI-authored).")
    print("only separate AND catchier displaces: human culture ends up rare")
    print("among humans while AI keeps its own cluster — and no single score")
    print("could have told that corner apart from assimilation.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny run for tests")
    args = parser.parse_args()
    if args.smoke:
        main(n_seeds=2, n_steps=20, seed=0)
    else:
        main(n_seeds=16, n_steps=200, seed=0)
