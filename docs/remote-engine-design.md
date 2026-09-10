# The remote engine — live runs behind an endpoint

> **ABANDONED 2026-08-01. Do not build on this, and do not deploy the service.**
> Jonas's verdict on what this plan produced: really bad and essentially
> worthless, and the work was not worth it. This is route B of two dead routes —
> live levers in policy tabs, which gives the player a form full of parameters
> they do not understand. Making plans data widened the engine's input surface
> without making it live; a request-response batch simulator is not real-time
> editing. The record and the rule against restarting it are in
> [docs/gd-game-postmortem.md](gd-game-postmortem.md). Everything below is kept
> as the history of what was tried, not as a plan.

*Deposited 2026-07-31, decisions by Jonas the same day: the policy-tab game
the GD game is growing into cannot be precomputed, and the answer is a live
engine server, with the tree-backed experience kept as the guided opening.
Siblings: docs/gd-game-design.md (the tree game), docs/dial-lattice-design.md
§4 hosting ladder v3 (which anticipated this endpoint), CLAUDE.md ⟦BOUNDARY⟧
(which this design preserves by construction), docs/observability-design.md
(every served run is a run record — the server is the observability layer's
first live producer). Status: R0 in implementation; the eq-network no-backend
ADR revision is drafted in that repo and awaits sign-off there.*

## 1. Why precompute ends here

The branch tree survives few windows × one card because affordability prunes
the space to tens of paths. The target experience — setup choices, then tabs
of policies adjustable over time — is exponential in checkpoints and dies at
any honest resolution. Precompute remains correct for the guided opening;
free policy play needs the engine itself answering.

## 2. The design in one paragraph

A game in progress is nothing but `(scenario, seed, policy plan so far)`. The
server is stateless: every request carries the accumulated plan, the engine
re-runs the whole trajectory from t=0 — trivial at this model size — and the
response returns the requested segment of the contract-v1.1 payload. Shared-
seed determinism guarantees the past never redraws. There are no sessions,
no database, and nothing to migrate; a cached response is exact; every
response is addressable as a run record.

## 3. The one real engine change (R0): plans are data, not config

Interventions today close over static config (onsets and rates are Python
constants), so every distinct plan is a new compiled program. The server
cannot recompile per request. R0 makes the policy plan a **(T, P) float
array carried in `global_attrs`** — a dynamic pytree child, like `step` — so
one jitted runner takes `(key, plan)` and serves every plan at the same
shape with zero recompilation.

Levers (P = 4 in v0), each read per tick from the plan row, each exactly
neutral at 0 (the sealing convention, bit-identity tested):

| lever | range | effect per tick | lineage |
|---|---|---|---|
| levy_rate | [0, 0.3] | conserving transfer of AI wealth to humans | the levy card, made continuous |
| repair_rate | [0, 0.05] | human wealth drip into `intervention_spend`, bounded enforcement uplift | the fund-repair card |
| reach_cut | [0, 1] | scales the money→attention channel via the `reach_cut_now` global (one-tick lag, documented) | the influence-cap card |
| sortition_rate | [0, 0.2] | blends citizen delegation rows toward uniform-over-citizens | the sortition card, made continuous |

**Cost model:** continuous political levers pay upkeep instead of one-shot
debits — enforcement drains per tick in proportion to lever intensity
(coefficients typed tuned-for-legibility; the erosion-of-capacity feedback
stays the game's spine). The card game's one-shot debit model is unchanged
and coexists.

All lever effects live in ONE composed transform (same-family disjoint
writes), values clipped in-transform to the ranges above as a second line of
defence behind the server's validation. Provenance: this is a model
extension under the same dated exception as the intervention events
(ledger_society ASSUMPTIONS.md, 2026-07-31).

## 4. The service (R1)

`service/` in this repo — FastAPI, one endpoint:

    POST /run {scenario_id, seed, plan: [{from_tick, levers}...], upto}
      -> {segment: contract-v1.1 fields for [prev_upto, upto),
          resources, manifest: run-record header}

- The plan arrives as piecewise segments, is expanded server-side to the
  (T, P) array, validated against the lever ranges — the whitelist IS the
  assumptions card, refusing anything off it.
- One warm jitted runner per scenario shape; requests differ only in array
  values. Cold start hides behind the game's loading screen by design.
- Caps: T ≤ 800, request rate limited, single scenario id in v0. No
  arbitrary config — scenario variants are named server-side.
- Deploy target Fly.io/Cloud Run (~$5–15/mo warm); CORS pinned to
  eq-network.org. Not in scope here: the deployment itself.

## 5. What stays where (the hybrid, Jonas 2026-07-31)

The tree-backed game remains the guided opening — instant, reliable,
offline-safe — and absorbs the Civilization-style pacing work: backstory
screens, initial choices, the run starting quietly behind them. The policy
tabs (economy / attention / polity, levers changing over time) are the live
sandbox behind it, reached only after the guided arc. If the server is cold
or down, the opening still works and the sandbox says so plainly.

## 6. What a critic should press on

Re-running from t=0 per request is honest but quadratic in interaction count
over a session; at T=400 it is nothing, at T=8000 it would not be — the
design holds for this model class, not universally, and says so. Upkeep
coefficients will be tuned until the sandbox plays well; they carry their
typing and the tuning is declared. The server widens the attack surface of a
until-now static site; the mitigations (range whitelist, rate caps, no
config passthrough) are listed, not proven. And a live endpoint makes it
easy to quietly generate numbers no card governs — every response therefore
carries the run-record manifest so any screenshot traces back to
`(scenario, seed, plan)`.
