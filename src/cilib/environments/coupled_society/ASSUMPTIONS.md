# coupled_society — assumptions card

**What this says a society is.** A society is three tensor programs sharing
one population: the same humans supply labor (compute_economy), hold culture
(value_contagion), and grant attention (influence_exchange); the same AI block
accumulates capital, broadcasts, and attracts influence. Coupling is three
small named transforms, and everything — substrates, couplings, defenses — is
ordered by `compile_pipeline` from declared reads/writes.

**Assumptions.**

- Composition, not a new model: every domain layer is the registered
  substrate's own transform factories, unchanged. Whatever each home card
  assumes (CES with σ=2; origin-only contagion with native reversion; anchored
  DeGroot with preferential attachment), this env inherits verbatim — read the
  three home cards; this card only owns the couplings.
- Five coupling transforms, one dial κ, all neutral at κ=0 (identical pipeline
  shape, so same-key κ-vs-0 twins are causal comparisons):
  `economic_power_buys_persuasion` (AI broadcast effort ∝ capital's income
  share), `persuasion_shifts_politics` (AI attractiveness ∝ converted-human
  share), `politics_rewrites_market_rules` (tax `enforcement` erodes as human
  influence falls below `fair_share` — a gradient from the initial share, not
  a 50% cliff), and the two arrows INTO the economy that close GD §5's
  compounding loop (added after review, 2026-07-27): `regulatory_capture`
  (the same influence deficit extracts rents from labor income into AI
  capital income, where reinvestment compounds it) and `converts_capitalize`
  (AI-cultured humans route income into AI services/capital). Without those
  two, the economy was an exogenous driver and the undefended coupled and
  sealed labor shares were bit-identical.
- The economy's vital sign is the human INCOME share (who income goes to),
  not the factor share wage·L/Y — factor shares are blind to rents by
  construction, and rule-capture is a rent. `labor_share` is kept as the
  classical readout.
- Coupling functional forms are linear-in-κ with hand-set gains — chosen for
  legibility, not measured. What a critic should press on: the gains'
  magnitudes ARE the result's scale; only the sign/ordering claims are robust.
- Per-domain dials default MILD (p_advantage 1.0, amplification 2.0): the
  flagship premise is per-domain recoverability, so any joint decline is
  attributable to coupling, not to stacking three separately-lethal baselines.
- One AI block spans all domains (the same actors get rich, loud, and
  followed). No within-block heterogeneity of roles.
- The composition operator is THE SCHEDULE (whitepaper §3.3): each domain's
  transforms register on the global clock with their own (cadence, phase);
  between firings the other domains read stale state. Lockstep (all cadence 1,
  the default) is the special case, reproduced bit-exactly by the unwrapped
  pipeline. First scan (2026-07-27, {1,2,4}³ at κ=0.8): no chaos at these
  gains — late-composite std stays in [0.032, 0.042] everywhere — but the
  economy clock is the one real lever (slower economy compounds less: labor
  share 0.14/0.21/0.26 at cadence 1/2/3; undefended composite 0.40 → 0.46),
  while culture and politics cadences are near-neutral (equilibrium-seeking
  dynamics). Most volatile structures: slow economy + fast culture/politics.
- Absent: any domain's own absent list, plus cross-domain prices (persuasion
  is bought with income *share*, not spent), lags (couplings act within one
  tick), and a state actor.

**Classical result reproduced.** Not one theorem — a sealing identity: at κ=0
an economy-dial change leaves culture and politics trajectories BIT-IDENTICAL
under the same key (`tests/test_composition.py::test_kappa_zero_seals_domains`),
and the transfer-gap instrument's null is exactly 0 between sealed twins.

**The dial.** `kappa`. At T=500, seeds 8 (defenses: enforced tax onset 50 +
sortition cadence 15 + influence cap): sealed defended composite 0.762 →
coupled defended 0.634; undefended human income share 0.117 sealed → 0.070
coupled (the flywheel reaching the economy); correlated decline −0.02
sealed-defended → 0.60 coupled-undefended.

**Instrument.** `metrics.defense_transfer_gap(coupled, sealed, key, ...)` —
paired same-key batches, composite differenced: **0.128 defended / 0.109
undefended** at defaults. The defended gap EXCEEDING the undefended one is
the paper's claim in its sharpest form: defenses lose more of their
protection to coupling than having none would. Sortition still holds
influence (0.73→0.64), but the rent channel taxes what enforcement erosion
leaves open, and money-buys-reach erodes culture (0.80→0.62) — the channels
no attached defense guards.

**Lineage.** Composes compute_economy + value_contagion + influence_exchange
(their cards colocate with them). `enforced_ai_tax` originates here-adjacent
(mechanisms/fiscal.py) as the enforcement-scaled flat tax.

**Status.** Live v0 (2026-07-27, this deposit): 5-rung composition ladder
passing. The docs' spectral lock-in claim (per-domain ρ<1, coupled ρ>1)
remains an UNVERIFIED research thread — nothing here tests it; do not present
it as a finding. Public /lab chip stays "in-design" until deliberately
flipped. Not yet in `experiments/benchmark/scenarios.py`.
