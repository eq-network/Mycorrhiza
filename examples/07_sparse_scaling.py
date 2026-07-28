"""What does a dense adjacency actually cost? — dense vs sparse `friendship`.

A friendship network with mean degree 6 on 5000 agents has ~30k edges out of 25
million possible pairs: 99.88% of the matrix is zero. XLA cannot see that. A
dense `(N, N)` array is dense as far as the compiler is concerned, so every
`W @ x` in the contagion step pays the full O(N^2) — the zeros are multiplied
and added like any other number. That is the ceiling on large influence and
empowerment runs, and it is a representation problem, not an algorithm one.

`value_contagion` can store the same graph as a sparse BCOO
(`sparse_friendship=True`, see `environments/networks.to_sparse`). Same
generator, same key, same graph, same trajectory — the equivalence is pinned by
`test_sparse_equivalence.py`. Only the cost changes: work tracks EDGES (~N*d)
instead of node PAIRS (N^2), so doubling N roughly doubles the sparse cost while
quadrupling the dense one.

This script measures that, rather than asserting it. Compile time is reported
separately from steady-state time because they scale differently and only the
latter is what a long run pays.

Two things the measurements say that the argument above does not, and which are
the reason this file exists rather than a paragraph claiming a speedup:

- **Memory wins immediately; time does not.** The stored-bytes column improves by
  20-300x from N=400 up, but wall-clock only crosses over around N ~ 2-3k. Below
  that, sparse is neutral or slightly SLOWER: a dense matvec runs a vectorized
  BLAS kernel, while BCOO runs gather/scatter-add, which is memory-bound and has
  a much worse constant factor. Sparsity buys asymptotics, not a free lunch.
- **Short runs measure dispatch, not arithmetic.** At `--steps 50` there is a
  ~0.2s floor of per-call overhead that hides the N^2 term entirely and makes
  both columns look flat. The defaults below are chosen to clear that floor; if
  you shorten the run, expect the exponents to collapse toward zero and do not
  read that as the matvec being cheap.

    python examples/07_sparse_scaling.py                 # 1000 / 3000 / 6000
    python examples/07_sparse_scaling.py --sizes 500 5000 --steps 100
    python examples/07_sparse_scaling.py --smoke         # tiny run (tests)
"""
import argparse
import time

import jax
import jax.numpy as jnp
import jax.random as jr

from cilib.environments import make_env

MEAN_DEGREE = 6.0        # fixed: sparsity is the point, so degree must not grow with N


def adjacency_bytes(W) -> int:
    """Bytes actually stored for the matrix — measured off the arrays, not a
    formula. A BCOO holds `data` (nse,) plus `indices` (nse, 2)."""
    if hasattr(W, "data"):
        return int(W.data.nbytes + W.indices.nbytes)
    return int(W.nbytes)


def time_run(n_agents: int, n_steps: int, sparse: bool, repeats: int):
    """Returns (compile_seconds, best_warm_seconds, adjacency_bytes).

    `best` rather than mean: we want the cost of the computation, and noise on a
    shared machine only ever adds time.
    """
    env = make_env("value_contagion", n_agents=n_agents, n_ai=n_agents // 5,
                   mean_degree=MEAN_DEGREE, sparse_friendship=sparse)
    key = jr.PRNGKey(0)

    t0 = time.perf_counter()
    finals, trace = env.run(key, n_steps)
    jax.block_until_ready(trace["culture"])
    compile_s = time.perf_counter() - t0          # first call: trace + compile + run

    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        _, tr = env.run(key, n_steps)
        jax.block_until_ready(tr["culture"])
        best = min(best, time.perf_counter() - t0)

    return compile_s, best, adjacency_bytes(finals.adj_matrices["friendship"])


def main(sizes, n_steps: int, repeats: int) -> dict:
    print(f"value_contagion: {n_steps} steps, mean degree {MEAN_DEGREE:.0f}, "
          f"1 seed, best of {repeats} warm runs\n")
    header = (f"{'N':>6}{'density':>10}{'dense (s)':>12}{'sparse (s)':>12}"
              f"{'speedup':>10}{'dense mem':>12}{'sparse mem':>12}{'mem x':>8}")
    print(header)
    print("-" * len(header))

    results = {}
    for n in sizes:
        c_d, t_d, m_d = time_run(n, n_steps, sparse=False, repeats=repeats)
        c_s, t_s, m_s = time_run(n, n_steps, sparse=True, repeats=repeats)
        results[n] = {"dense_s": t_d, "sparse_s": t_s, "speedup": t_d / t_s,
                      "dense_bytes": m_d, "sparse_bytes": m_s,
                      "compile_dense_s": c_d, "compile_sparse_s": c_s}
        print(f"{n:>6}{MEAN_DEGREE / (n - 1):>10.4f}{t_d:>12.3f}{t_s:>12.3f}"
              f"{t_d / t_s:>9.1f}x{m_d / 1e6:>11.1f}M{m_s / 1e6:>11.3f}M"
              f"{m_d / m_s:>7.0f}x")

    print("\ncompile time (first call: trace + compile + run), reported separately:")
    for n, r in results.items():
        print(f"  N={n:<6} dense {r['compile_dense_s']:.2f}s   "
              f"sparse {r['compile_sparse_s']:.2f}s")

    # The claim is about SCALING, not a single ratio: report observed growth
    # exponents. Dense matvec is O(N^2), sparse is O(N*degree) = O(N), so the
    # fitted slopes on a log-log plot should straddle ~2 and ~1.
    if len(sizes) >= 2:
        lo, hi = sizes[0], sizes[-1]
        span = jnp.log(hi / lo)
        print(f"\nobserved cost growth from N={lo} to N={hi} "
              f"(exponent p in cost ~ N^p):")
        for label, k in (("dense", "dense_s"), ("sparse", "sparse_s")):
            p = float(jnp.log(results[hi][k] / results[lo][k]) / span)
            print(f"  {label:<7} N^{p:.2f}   ({results[lo][k]:.3f}s -> "
                  f"{results[hi][k]:.3f}s)")
        print("  (dense ~ N^2 is the O(N^2) matvec; sparse ~ N^1 tracks edges.)")

    print("\nthe graph is identical in both columns: same generator, same key,")
    print("same trajectory (test_sparse_equivalence.py). only the storage differs.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # defaults chosen to clear the ~0.2s per-call overhead floor (see docstring):
    # below ~150 steps or ~2k agents the fixed cost hides the scaling entirely.
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 3000, 6000])
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="tiny run for tests")
    args = parser.parse_args()

    if args.smoke:
        main(sizes=[60, 120], n_steps=5, repeats=1)
    else:
        main(sizes=args.sizes, n_steps=args.steps, repeats=args.repeats)
