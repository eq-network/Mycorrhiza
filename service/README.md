# service — the RemoteEngine (v0)

> **DEAD 2026-08-01. Do not deploy, extend, or wire a front end to this.** The
> policy-tab game it was written for was judged not worth
> continuing. It is route B of two dead
> routes: a stateless endpoint that re-runs from t=0 is submit-and-wait, not
> the real-time editing a game needs, and the levers it exposes are model
> internals a new player cannot read. See `docs/gd-game-postmortem.md` before
> touching anything here.

Live `ledger_society` runs behind one stateless endpoint, for the policy-tab
game (docs/remote-engine-design.md; decision 2026-07-31). A game in progress
is `(scenario, seed, plan)`; every request re-runs from t=0 and shared-seed
determinism keeps the past identical, so there are no sessions and nothing to
store. One jitted runner per (scenario, T) — the plan is a dynamic argument,
so requests never recompile.

    pip install -e .[service]
    uvicorn service.app:app --port 8100

    POST /run {"scenario_id": "ledger-society-default", "seed": 3, "T": 400,
               "plan": [{"from_tick": 100, "levers": {"levy_rate": 0.15}}],
               "upto": 200}

Levers and their declared ranges come from `mechanisms.PolicyLeverConfig`
(`GET /health` lists them); off-range values are refused with the range in
the error, never clamped silently. Named scenarios only — config never
crosses the wire. Not in this repo: deployment (Fly.io/Cloud Run), TLS, and
rate limiting, which belong to the platform in front of this app; the
eq-network side carries the ADR that authorizes a backend at all.
