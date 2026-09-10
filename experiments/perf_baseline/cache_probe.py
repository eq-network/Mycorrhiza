"""Does JAX's persistent compilation cache rescue the eager (un-jitted) path?

``perf_baseline.run`` shows an eager ``env.run`` re-traces and re-compiles on
every call. The persistent cache is keyed on the compiled module's fingerprint,
not on jaxpr identity, so an identical eager call can hit it after the first
compile, both within one process and across processes. What it cannot remove
is the re-trace and the cache round trip, which is why the README reports the
three numbers side by side: eager repeat without cache, eager repeat with
cache, and a jitted repeat.

Run it three times to reproduce the README's row: once with no cache dir, then
twice with the same cache dir (cold, then warm).

    python -m experiments.perf_baseline.cache_probe                # no cache
    python -m experiments.perf_baseline.cache_probe /tmp/jaxcache  # cold
    python -m experiments.perf_baseline.cache_probe /tmp/jaxcache  # warm
"""
import sys
import time

import jax
import jax.random as jr

cache = sys.argv[1] if len(sys.argv) > 1 else None
if cache:
    jax.config.update("jax_compilation_cache_dir", cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

from cilib.environments import make_env  # noqa: E402  (after the cache config)

env = make_env(sys.argv[2] if len(sys.argv) > 2 else "ledger_society")
key = jr.PRNGKey(0)
T = 200
for i in range(3):
    t0 = time.perf_counter()
    _, tr = env.run(key, T)
    jax.block_until_ready(tr)
    print(f"eager call {i + 1}: {time.perf_counter() - t0:.2f}s")
f = jax.jit(lambda k: env.run(k, T))
t0 = time.perf_counter(); jax.block_until_ready(f(key)); print(f"jit call 1: {time.perf_counter() - t0:.2f}s")
t0 = time.perf_counter(); jax.block_until_ready(f(key)); print(f"jit call 2: {time.perf_counter() - t0:.3f}s")
print("cache:", cache or "off")
