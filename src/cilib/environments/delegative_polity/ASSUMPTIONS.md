# delegative_polity — assumptions card

**What this says a polity is.** A polity is a delegation structure: everyone
holds one vote, may hand it to someone else (human or AI), and the tax rate is
whatever the power-weighted median of the ballot-holders declares — political
power is nothing over and above the ballots in your hand.

**Assumptions.**

*Power accounting (the floors — each is a dial, and each bounds the failure
away from zero at its default; measured decomposition of the captured
end-state: share 0.383 = 0.132 franchise + 0.089 handback + 0.161 residual):*

- Ballots are conserved and zero-sum: one per ballot-holder, none minted, no
  wealth -> attractiveness coupling (that arrow lives in coupled_society).
  Power redistributes; it cannot compound.
- Power is one-hop ballot weight: your kept diagonal plus votes handed
  directly to you (ballot-weighted column sums of the row-stochastic
  ``delegation`` matrix). Deliberately NOT eigenvector/transitive voice — a
  delegate casts what it holds rather than forwarding it forever, and the
  eigenvector's recycling fixed point would cap the AI bloc below a majority
  by construction. One-hop also caps a citizen's exposure at their delegated
  share: without chains, voice cannot be laundered onward.
- **The franchise floor** (``self_weight`` = 0.15, ``franchise_erosion`` = 0):
  every citizen keeps 15% of their own ballot, and by default NO transform
  writes to it — not even lock-in. The vote itself is unrevokable unless
  ``franchise_erosion`` arms lock-in's third channel (effective kept share =
  self_weight x (1 − erosion x (1 − regime))). This floor alone is 0.132 of
  total power.
- **The AI-ballot handback** (``ai_ballot`` = 1): AI delegates hold one base
  ballot each AND their frozen uniform rows spread most of it back over
  citizens every tick — a perpetual pro-citizen subsidy worth 0.089, inherited
  from influence_exchange's listening idiom, with no strong political
  interpretation of its own. ``ai_ballot = 0`` makes them pure conduits.
- **The flow floors**: attachment moves at most ``update_rate`` per tick
  toward a target that saturates (T(s) <= 1 at any finite advantage), and the
  churn re-draw is uniform over the polity — mostly citizens — so a fixed
  advantage buys a fixed interior share, never a runaway. Lock-in kills churn
  (friction -> 0), which removes this floor's citizen-to-citizen residual.
- Crash corollary (E4; ladder rung 9): remove all three — lock-in driving the
  regime to zero, ``ai_ballot = 0``, ``franchise_erosion = 1`` — and the human
  power share reaches exactly zero by t≈400. The default ~0.38 captured floor
  is an assumption, not a finding.

*Resource/type inventory (interop; declared in code as ``state.LEDGERS`` /
``state.PORTS`` per docs/ledger-design.md §2, mapped across the suite in
docs/gd-suite-v0.1.md §2):*

- Conserved ledgers: ``delegation`` (per-ROW ballot shares — pure
  reallocation, the suite's politics ledger; note the referee caveat that this
  is a re-issue rule, not an accumulating stock) and ``wealth`` (money, minted
  at sum(endowment)/tick by ``tax_and_redistribute``, never spent — a
  write-only score whose promotion to WP1's circulating money is the coupled
  rewrite's named unification seam).
- Ports (computed collective / rate variables — the only other fields a
  coupling may read): ``influence``, ``policy_target``, ``enforcement``,
  ``redelegation_friction``, ``amplification``, ``cap_scale``,
  ``attract_boost``, ``engagement``.
- Shared kernel: ``rewire_delegation`` is a conscious fork of
  ``influence_exchange.make_rewire`` (gd-suite §2.4) with exactly three
  declared differences (churn, the erodible franchise floor, engagement-gated
  rows); divergence beyond those is a suite-level bug.

*Preferences and positions:*

- Preferences are exogenous ideal points on one tax dimension (Black/Downs
  spatial voting): citizen ideal = clip(``true_rate`` + noise) — a noisy
  signal of an epistemically best rate (the Condorcet reading). Citizens never
  change their minds; only the delegation structure evolves. Structure never
  reads ideals (re-delegation is prominence-driven, not ideological), so
  influence-from-birth equals influence-now by construction.
- A human super-voter votes their OWN ideal, not their delegators' (faithful
  liquid democracy). AI delegates blend their delegators' weighted mean ideal
  with ``ai_bias`` at fidelity ``alignment_ai`` (agents/delegate.py's formula).

*Dynamics and populations:*

- Delegation drifts by preferential attachment with SUPERLINEAR prominence
  (``gamma`` = 1.3 > 1): linear attachment on a fixed population is
  share-neutral (Krapivsky–Redner), so the Michels organic-oligarchy baseline
  requires gamma > 1 — the gamma <= 1 honest region is where no organic
  concentration occurs. Churn drifts every row back toward a uniform re-draw:
  the freedom to re-delegate, Przeworski's institutionalized uncertainty as a
  rate.
- AI actors are a frozen reservoir with a FIXED population (no arrivals, no
  replication) and a STATIC advantage (``ai_advantage`` is a constant
  multiplier on reach, not a growing capability). Positions are pinned through
  the fidelity blend; content/persuasion lives in value_contagion.
- Citizens' engagement is constant (BroadcastPolicy closure): no apathy
  feedback. Disengagement in this model would freeze a citizen's row — apathy
  protects rather than surrenders, a deliberate (and disputable) boundary
  choice.
- Lock-in is one memoryless scalar: ``regime`` falls smoothly as the top
  node's power share exceeds ``entrenchment_threshold`` and gates tax
  ``enforcement``, ``redelegation_friction``, and (only if armed) the
  franchise. No hard-coded ratchet — hysteresis must emerge from the
  power -> rules -> power feedback or be reported absent.
- The economy is deliberately thin: fixed heterogeneous endowments, flat tax
  at ``rate x enforcement``, equal redistribution — a pure transfer,
  money-conserving by construction. Wealth is a readout only; it buys nothing.
- Absent: delegation chains, parties/elections/terms, strategic or learning
  agents, media beyond the network, multi-dimensional policy,
  endowment-preference correlation (Meltzer–Richard's channel — noted as the
  alternative microfoundation, not modeled).

**Classical result reproduced.** Black/Downs median voter: in the
direct-democracy limit (delegation = identity, no AI) the enacted rate equals
the citizen median ideal exactly, every tick, and the identity matrix is a
fixed point of the rewire
(`tests/test_validation_ladder.py::test_median_voter_direct_democracy_limit`).
Supporting rungs: organic super-voter concentration (Michels; Gölz et al.'s
attachment mechanism) and the wisdom readout
(``::test_wisdom_dispersed_vs_captured``).

**The dial.** ``ai_advantage`` (with ``ai_advantage_onset``): 1.0 = organic
delegation (concentration but human-held, share ≈ 0.89); 4.0 from t=50 =
capture (human share ≈ 0.39 at T=400; policy jumps to the AI blend once the
bloc crosses half the ballots, decision error ≈ 0.20 vs ≈ 0.04 dispersed).
Escalations: ``entrenchment_gain`` (default 0) — the lock-in dial — and the
floor dials ``ai_ballot`` / ``franchise_erosion`` above.

**Instrument.** Mainline indicators: ``human_power_share``,
``influence_gini`` / ``top_delegate_share`` / ``centralization``
(metrics/families/concentration.py), ``policy_median_gap``,
``decision_quality``, ``enforcement_level``, ``wealth_gini``. Defenses: the
``political`` family — ``sortition`` (adj_key="delegation", scheduled) +
``influence_cap``; calibrated 2026-07-30 so the conditions separate: organic
0.89 / captured 0.39 / defended 0.74 (cap 0.04, share 0.5, cadence 15, T=400,
6 seeds).

**Lineage.** Fork-sibling of influence_exchange (A4): reservoir, scheduled
threat, ER-init, and rewire idioms carried over; changed the adjacency
semantics (delegation of ballots, not attention), power (one-hop ballot-
weighted column sums, not eigenvector), and added the policy/tax layer, the
lock-in regime, and the explicit floor dials. Built for WP3
(papers/wp3-politics), as capital_economy was for WP1.

**Status.** Live v0.1 (2026-07-30: v0 same-day revision after the floor
audit — power floors surfaced, dialed, and swept; defaults bit-preserve v0
behavior). 11-rung ladder passing. Not in
`experiments/benchmark/scenarios.py` (WP3 stands alone, like
capital_economy); public /lab status chip untouched.
