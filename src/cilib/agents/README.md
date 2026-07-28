# Agents catalog

Pre-made, swappable decision rules. Pick one by name from `REGISTRY` (in
`__init__.py` — open it to see every entry).

**Type function:** `AgentFactory = Config -> Policy`, where a `Policy` is a callable
`(obs, key) -> action` (`cilib.core.protocols.Policy`). A *pure* agent that threads
no Python state may instead implement `PureAgent` — `round_fn() -> (state, t, key) -> state` —
so it runs inside `core.scan.run_scan` and `vmap`s over seeds.

**Entries:** `random`, `tit_for_tat`, `linear`, `ai_delegate` (principal→delegate acting
with an alignment dial — the influence-metric hook), `labor_supply` (work_pref-scaled,
mildly wage-elastic household rule — closes `compute_economy`), `spend_share`
(spend-by-preference household rule — closes `io_economy`), `broadcast`
(constant transmission-effort rule — closes `value_contagion`; the C4
strategic-persuader seam). (Also present:
`rl_components`, `profiles` — support modules, not yet cataloged.)

**Add one:**
1. Write a `Policy` (or `PureAgent`) in a module here.
2. Add one line to `REGISTRY` in `__init__.py`: `"my_agent": MyPolicy`.
3. Add a behavioral test asserting what it does.

Effectful agents (LLM/HTTP) live on the eager `core.time` tier.
