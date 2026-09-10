# Mechanisms catalog

Composed institutions — markets, networks, democracies — expressed as typed
transforms. A mechanism is internally a composition of `transformations` (e.g. a
market is `elicit -> match -> clear -> settle`). Pick one by name from `REGISTRY`.

**Type function:** `TransformFactory = Config -> Transform`. The returned Transform
**declares its `.reads` / `.writes`** via `@transform`, so `compile_pipeline` derives
execution order from effects. Entries within a *family* keep **disjoint write sets**
so `parallel(market, network)` is always valid.

**Families:** `market` · `network` · `democracy` · `fiscal`.
**Entries today:** `market`; `quota_vote` + `graduated_sanction` (`democracy` —
quantile vote → policy target; over-quota penalty + confiscation); `ai_revenue_tax` +
`ownership_cap` (`fiscal` — tax-and-redistribute capital income; cap any actor's share
of AI compute). The variants — double-auction, sealed-bid, trust-weighted / gossip
networks, liquid / representative democracy — are the first open-source follow-ups,
each a new file + one `REGISTRY` line.

**`families/` — the GD game's three lever bundles** (`economy_levers`,
`culture_levers`, `politics_levers`): each one composed transform reading one
`(T, P)` plan array carried in `global_attrs`, so plan *values* are data and one
compiled program serves every plan. They are a **sequenced bundle, not a
parallel family** — economy and politics share the `wealth` write on purpose.
See [families/README.md](families/README.md) for levers, ranges, the write map
and the neutrality guarantee.

**Timing belongs to the schedule, not the mechanism:** entries are pure rules; wrap
with `core.schedule.scheduled(mech, cadence, phase_offset, onset)` to control when they
fire (`onset` = regime-shift dial). Benchmark conditions are (mechanism, config,
schedule) triples.

**Add one:**
1. Compose `transformations` into a `make_<mech>(cfg) -> Transform` factory.
2. `@transform(reads=[...], writes=[...])` with writes disjoint from its family siblings.
3. Add one line to `REGISTRY`; add a swap test (it composes where its siblings do).

**Swap test (the payoff):** replacing one family member with another in a pipeline
must keep the pipeline compiling — same read/write contract, different dynamics.
