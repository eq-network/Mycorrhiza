# Assumptions card — `value_contagion`

*Card contract: `docs/model-register-design.md` §6. Colocated so a fork of this
directory carries its assumptions with it. Register: cultural
(`docs/cultural-register-design.md`), entry C2.*

**1. What this says culture is.** Something you catch from friends: one variant
per person, adopted by exposure, distinguished only by origin (human vs AI).

**2. Assumptions.**
- Two-sided competing contagion with independent per-source transmission
  (`1 − Π(1 − β_j)`); synchronous updates; complex-contagion gate at
  `k_threshold` exposures (default 1 = simple contagion).
- The persuasive advantage rides on the **variant**, not the carrier: a
  converted human spreads AI-origin culture at full advantage. It is a bare
  multiplier — no content, no semantics, no targeting (the register's axis-P
  sub-parameters 2–3 belong to `value_spectral`/C4, not here).
- AI nodes are a **frozen reservoir** (never convert); humans holding AI-origin
  culture revert natively at `recovery` — load-bearing: without it the reservoir
  makes "everyone AI-cultured" the only fixed point and pluralism cannot exist.
- Fixed friendship network per run (`networks.typed_homophily`, degree-corrected
  so the separation dial does not move the epidemic threshold); no payoffs
  (`last_reward` inert); the open boundary is per-agent broadcast *effort* only.
- **Nested-substrates caveat (register §3):** this contagion model is the
  constant-adoption-rate limit of the planned `value_spectral` model, and its
  competitive decay is `value_replicator`'s. Agreement across the cultural
  register's members is agreement across three separately-validated regimes of
  ONE theory of cultural transmission — a weaker robustness claim than the
  economy register's three independent substrates.
- **Absent:** variant content/semantics and dissonance (C3), strategic
  persuaders (C4, `close_multi`), message phylogeny/attribution (§6 of the
  register), population turnover, media/broadcast structure beyond the network.

**3. Classical result reproduced.** The SIS invasion threshold in the
substrate's degenerate limit (tiny `beta`, large `p_advantage`, no reservoir:
sub-critical seeds die out, super-critical go endemic) and the
complex-contagion gate (a lone carrier can never spread at k=2 — Centola 2010),
plus the register's dial rungs: `ai_homophily` moves
`fiedler_partition_alignment` monotonically, `p_advantage` moves
`human_origin_share` monotonically (`tests/test_substrate.py`,
`environments/tests/test_networks.py`).

**4. The dial.** The **(S, P) pair** — `ai_homophily` × `p_advantage` — and the
deliverable is the four-corner phase table (measured at defaults, 16 seeds ×
200 steps: pluralism 0.79/0.12, assimilation 0.23/0.12, parallel 0.98/0.93,
displacement 0.25/0.93 as share/alignment). Never report either readout alone:
a scalar "AI cultural share" cannot tell assimilation from displacement
(register §7), which is the point of the model.

**5. Instrument.** Headline readouts: `human_origin_share` (env metric) paired
with `metrics.families.spectral.fiedler_partition_alignment` (C0 slice, read
from finals). Not yet a benchmark `ScenarioSpec` in
`experiments/benchmark/scenarios.py`; candidate counterfactual design (channel =
broadcast effort / recovery, outcome = late human-origin share) noted, not
declared — no duplicate contracts.

**6. Lineage.** Originates here (2026-07-24, register C2), superseding the
`value_epidemic` sketch in `docs/abm-suite-design.md` (multi-strain SI/SIS with
fidelity decay) per the cultural register redesign. Built ahead of A4 as a
deliberate override of the register's A4-first sequencing — the cheap
C0+C2 slice only; the risky part that rule guards (C4/`close_multi`) is
untouched. Register siblings: `value_replicator` (C1, designed),
`value_spectral` (C3, designed).

**7. Status.** Live v0 (2026-07-24): all four regime corners separate with
margin at defaults; SIS-limit threshold + gate rungs passing. k≥2 is the
minimal gate rung only (full Centola threshold-shift prediction deferred);
C0 delivered `fiedler_partition_alignment` only.
