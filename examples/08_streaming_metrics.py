"""The second memory wall: the trajectory, not the state.

`examples/07_sparse_scaling.py` made the STATE cheap — a mean-degree-6 network
on 5000 agents drops from a 144 MB dense matrix to a 0.45 MB sparse one. That
does not make the run possible, because the state was never the big object:

    value_contagion, N=5000, 16 seeds, T=2000
        sparse `friendship` in the state ......    0.45 MB
        stacked trace .........................    1.28 GB      ~2800x

Every environment's `default_trace` records raw per-agent arrays and defers
reduction to `metrics.py`, so a run materializes O(T x N) floats in order for a
metric to walk them once and return, usually, one float. `human_origin_share`
builds a (T, N) array to produce a single number.

A `Reducer` (see `core/reduce.py`) folds that metric INTO the scan instead: the
accumulator lives in the carry, so the cost is O(1) rather than O(T x N). Same
arithmetic, same answer to float32 rounding, no trajectory retained.

The catch worth knowing: reducing the AGENT axis is what saves memory. A (T,)
scalar series is ~8 KB at T=2000, so timelines are nearly free and worth
keeping in `trace_fn` — this script keeps one and shows the reducer agreeing
with it.

    python examples/08_streaming_metrics.py            # full run
    python examples/08_streaming_metrics.py --smoke    # tiny run (tests)
"""
import argparse
import time

import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.environments import make_env
from cilib.environments.value_contagion import make_reducers

BYTES_PER_F32 = 4
TRACE_FIELDS = 2          # value_contagion traces `culture` and `broadcast_effort`


def projected_trace_bytes(n_agents: int, n_steps: int, n_seeds: int) -> int:
    """What the stacked (B, T, N) trace would cost. Computed, not allocated —
    the whole point is that the large cases cannot be materialized."""
    return n_seeds * n_steps * n_agents * BYTES_PER_F32 * TRACE_FIELDS


def state_bytes(env, key) -> int:
    W = env.init_fn(key).adj_matrices["friendship"]
    return int(W.data.nbytes + W.indices.nbytes) if hasattr(W, "data") else int(W.nbytes)


def main(n_agents: int, n_steps: int, n_seeds: int, verify_n: int) -> dict:
    print("the two memory walls, for value_contagion (mean degree 6)\n")
    header = (f"{'N':>7}{'sparse state':>15}{'trace (B,T,N)':>16}"
              f"{'reducer':>10}{'trace/state':>13}")
    print(header)
    print("-" * len(header))
    for n in (500, 1000, 5000, 20000):
        env = make_env("value_contagion", n_agents=n, n_ai=n // 5,
                       sparse_friendship=True)
        sb = state_bytes(env, jr.PRNGKey(0))
        tb = projected_trace_bytes(n, 2000, 16)
        # one f32 accumulator + one f32 count, per seed, per metric
        rb = 16 * 2 * BYTES_PER_F32
        print(f"{n:>7}{sb / 1e6:>14.2f}M{tb / 1e9:>15.2f}G{rb:>9}B{tb / sb:>12.0f}x")
    print("  (trace column is projected at T=2000, 16 seeds -- not allocated)")

    # --- the two paths agree, at a size where both fit ---------------------------
    print(f"\nverifying the streaming metric against the trace metric "
          f"(N={verify_n}, T=200, 4 seeds):")
    env = make_env("value_contagion", n_agents=verify_n, n_ai=verify_n // 5,
                   sparse_friendship=True, p_advantage=6.0)
    reducers = make_reducers(env.config)

    _, trace = env.run_batch(jr.PRNGKey(0), 4, 200)
    from_trace = jax.vmap(env.metrics["human_origin_share"])(trace)
    _, _, reduced = env.run_reduced_batch(jr.PRNGKey(0), 4, 200, reducers)
    from_reducer = reduced["human_origin_share"]

    held = sum(int(v.nbytes) for v in trace.values())
    print(f"  trace path   : {float(from_trace.mean()):.6f}   "
          f"({held / 1e6:.1f} MB of trajectory held)")
    print(f"  reducer path : {float(from_reducer.mean()):.6f}   "
          f"({from_reducer.nbytes} B held)")
    print(f"  max |difference| = {float(jnp.max(jnp.abs(from_trace - from_reducer))):.2e}"
          f"   (float32 rounding: a streaming sum reduces in step order)")

    # --- the run the trace path cannot do ----------------------------------------
    would_need = projected_trace_bytes(n_agents, n_steps, n_seeds)
    print(f"\nnow the run that motivated this: N={n_agents}, T={n_steps}, "
          f"{n_seeds} seeds")
    print(f"  trace path would need {would_need / 1e9:.2f} GB of trajectory "
          f"-- not attempted")
    env = make_env("value_contagion", n_agents=n_agents, n_ai=n_agents // 5,
                   sparse_friendship=True, p_advantage=6.0)
    reducers = make_reducers(env.config)

    t0 = time.perf_counter()
    _, series, reduced = env.run_reduced_batch(
        jr.PRNGKey(0), n_seeds, n_steps, reducers,
        # a (T,) scalar timeline is ~8 KB -- cheap enough to keep
        trace_fn=lambda s: {"share": jnp.mean(1.0 - s.node_attrs["culture"])})
    jax.block_until_ready(reduced["human_origin_share"])
    elapsed = time.perf_counter() - t0

    share = reduced["human_origin_share"]
    kept = int(share.nbytes + sum(v.nbytes for v in series.values()))
    print(f"  reducer path : human_origin_share = {float(share.mean()):.4f} "
          f"+/- {float(share.std()):.4f}  in {elapsed:.1f}s")
    print(f"  memory held  : {kept / 1e3:.1f} KB "
          f"({share.nbytes} B of metric + a (T,) timeline per seed)")
    print(f"  reduction    : {would_need / kept:,.0f}x less than the trace path")

    print("\nsparsity fixed the state; streaming fixed the recording. neither")
    print("alone gets you a large, long run -- fixing one just moves the wall.")
    return {"share": float(share.mean()), "held_bytes": kept,
            "trace_bytes": would_need}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--verify-n", type=int, default=200,
                        help="size at which BOTH paths are run and compared")
    parser.add_argument("--smoke", action="store_true", help="tiny run for tests")
    args = parser.parse_args()

    if args.smoke:
        main(n_agents=200, n_steps=20, n_seeds=2, verify_n=40)
    else:
        main(args.agents, args.steps, args.seeds, args.verify_n)
