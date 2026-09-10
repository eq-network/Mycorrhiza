# The three lever families — economy · culture · politics

The GD game's continuous controls over `environments/ledger_society`, one
composed transform per family, built 2026-07-31 to
[docs/gd-game-three-families.md](../../../../docs/gd-game-three-families.md).
Each family reads **one plan array** — a `(T, P)` float32 dynamic pytree child
of `global_attrs`, indexed by the traced `step` — so plan *values* are data and
one compiled program serves every plan. Registered as `economy_levers`,
`culture_levers`, `politics_levers` in `mechanisms.REGISTRY`.

| family | verb | plan key | shape | what it is |
|---|---|---|---|---|
| economy | **allocate** | `economy_plan` | `(T, 5)` | steers the five-column allocation vector every household spends from |
| culture | **shape** | `culture_plan` | `(T, 3)` | sets the attention kernel's parameters; you move how attention moves, not attention |
| politics | **spend** | `politics_plan` | `(T, 5)` | draws on `enforcement`, a bounded stock refilled only by repair |

## Levers

`economy` — `ECONOMY_LEVERS`. Columns 0–3 are **deltas** on the human rows of
`alloc_pref`, neutral at 0; the resulting band is clipped to its range and the
residual (`save`) absorbs exactly what the bands took, so a row's sum is
preserved and never silently renormalised.

| col | lever | resulting band | typing |
|---|---|---|---|
| 0 | `d_consume` | 0.30–0.85 (neutral point 0.70) | arbitrary-but-swept |
| 1 | `d_invest` | 0.00–0.40 (neutral point 0.05) | arbitrary-but-swept; **above 0.25 is beyond the probe** |
| 2 | `d_broadcast` | 0.00–0.30 (neutral point 0.03) | arbitrary-but-swept |
| 3 | `d_lobby` | 0.00–0.20 (neutral point 0.02) | arbitrary-but-swept |
| 4 | `levy_rate` | 0.0–0.3 | tuned-for-legibility (`PolicyLeverConfig.levy_max`) |

`culture` — `CULTURE_LEVERS`. Columns 0–1 are **deltas** on config bases.

| col | lever | resulting value | typing |
|---|---|---|---|
| 0 | `gamma_w_delta` | γ ∈ [0.6, 1.4], base 1.0 | base **anchored** (linear attachment is share-neutral); range arbitrary-but-swept |
| 1 | `update_rate_w_delta` | r ∈ [0.03, 0.15], base 0.08 | base **anchored** to `influence_exchange`; range arbitrary-but-swept |
| 2 | `reach_cut` | 0–1 fraction of the money→attention dial removed | tuned-for-legibility |

`politics` — `POLITICS_LEVERS`. Two live player sliders, two scenario dials,
one inbound bill.

| col | lever | range | typing |
|---|---|---|---|
| 0 | `repair_spend_rate` | 0–0.05 of each hoard per tick | tuned-for-legibility |
| 1 | `sortition_rate` | 0–0.20 | tuned-for-legibility |
| 2 | `repair_rate` | 0–0.05, **0 = hold** the config value | arbitrary-but-swept; probed only 0.02→0.04 |
| 3 | `entrenchment_gain` | 0–0.05, **0 = hold** | arbitrary-but-swept; probed only 0→0.02 |
| 4 | `external_intensity` | 0–4.0 | the siblings' bill, built by `external_intensity_of` |

Ranges are both the server's whitelist and the in-transform clip. Every probe
number behind this table is 6 seeds, one field at a time, medians, **no CIs** —
ordering claims only; magnitudes live in the deposit next to that sentence.

## The write map

    economy   alloc_pref, wealth
    culture   gamma_w_now, update_rate_w_now, reach_cut_now
    politics  wealth, intervention_spend, enforcement, delegation,
              repair_rate_now, entrenchment_gain_now

Culture is disjoint from both siblings. Economy and politics share exactly one
field, `wealth`, **deliberately**: the levy is a conserving AI→human transfer of
the stock, the office drip removes part of the stock into the declared
`intervention_spend` sink. Funding the office competes with investing the hoard,
which is the design's third failure mode — the overlap is the mechanic, not a
naming accident. `compile_pipeline` turns it into a plain WAW edge oriented by
program order, so pass the mechanisms as `(economy, culture, politics)` and the
drip is taken from post-levy wealth. Because of that overlap the three are a
**sequenced bundle, not a `parallel`-composable family**.

`enforcement` has exactly one writer, politics. The other two families' political
acts are billed to it as data through column 4, priced off their own plan rows by
`external_intensity_of` and `culture_upkeep`.

None of the three composes with `make_policy_levers` or `make_interventions` —
the card game, the four-lever policy game and the three-family game are
alternative closures over the same channels.

## The neutrality guarantee

A zero plan row is exactly neutral for every lever in every family, and a run
carrying all three neutral plans is **bit-identical** to a run with no lever
transforms and no plan arrays on the state — every node attribute, both
adjacency ledgers, every global. Verified across seeds under `lax.scan`, which
is the tier that matters: the eager tier does not fuse across the mechanism
slot and so cannot see the failure mode. Politics' ballot write is
`D + sort_r * delta` rather than the algebraically equal decompose-and-recompose
form for exactly this reason — see the sortition block in `politics.py`.

`ledger_society`'s state factory does not create the three plan keys, so attach
them before `t=0` with `attach_economy_plan` / `attach_culture_plan` /
`attach_politics_plan`, and compose through `build_step_fn`. `build_game`'s
`validate_reads` rejects the transforms until `make_state` carries the keys.

## Adding a lever

Widen the family's plan array by one column, extend `<FAMILY>_LEVERS`, clip it
in-transform against a typed range on the config dataclass, and add a behavioral
test plus the neutrality rung. A lever that needs a substrate parameter to vary
mid-run needs that parameter promoted to a `global_attrs` port in
`ledger_society/state.py` first — six exist today.
