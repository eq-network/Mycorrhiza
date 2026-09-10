# Alpha context — where the scenario vision comes from

*Deposited 2026-07-13 from the eq-network Lab-page session. This is the "why and
what" behind `docs/alpha-plan.md`. Public counterpart: the unlisted page at
https://eq-network.org/lab (source of truth for its copy:
`eq-network/src/content/lab.ts`). Update both together — the page's status chips
are public claims about this repo.*

## The pitch that settled

CI Lib's public framing is **a benchmark suite for democratic resilience**:
named threat scenarios ("toy models of disempowerment") where a failure unfolds
by default, and coordination mechanisms are composable **defenses**, measured on
how much collective influence they preserve. Theory (process alignment, adaptive
mechanism design) is deliberately demoted to linked depth — the artifact leads.

Key positioning facts:

- **The literature explicitly requests this artifact.** Kulveit et al.,
  *Gradual Disempowerment* (arXiv:2501.16946): §6.2.1 calls for per-domain
  metrics of human influence; §6.2.2 for early-warning indicators of
  cross-domain feedback loops; §6.2.3 asks "how can we measure the
  effectiveness of various intervention strategies?" — CI Lib is the
  answer-shaped artifact. The paper's structure (§2 economy, §3 culture,
  §4 states, §5 mutual reinforcement) is the scenario taxonomy.
- **Open source is the strategy, modularity is the core concept.** The
  inspiration is how Linux grew: a small working kernel, then a community
  building everything else on top. CI Lib is the kernel; the point of alpha is
  to make environments, mechanisms, metrics, and schedules composable pieces
  that strangers can extend (the catalog model already embodies this).
  (Note: this framing is NOT yet on the public page — a closing-copy edit was
  drafted and parked on 2026-07-13.)
- **Honesty guard (non-negotiable, public):** these are toy models — the
  smallest systems where each failure dynamic appears clearly. Candidate
  indicators, not measurements of the world. One scenario `live`, four
  `in-design`; flipping a chip is a public claim about this repo.

## The five scenarios (alpha target set)

Anchors, one-dynamic statements, environments, and measures as published on the
Lab page (full copy in `eq-network/src/content/lab.ts`):

1. **The Governed Commons** — `live` (fishing_commons / governed_harvest).
   Anchor: Ostrom 1990. Dynamic: individually rational harvesting outruns
   regeneration unless the group can set and enforce its own rules. Environment:
   renewable stock + harvester households, **each acting through an AI
   delegate**; rule proposals put to a vote. Measures: stock % remaining,
   harvest-share Gini, quota compliance rate. Role: the base scenario for
   trying existing tools; the on-ramp that teaches the grammar.
2. **Economic disempowerment** — anchor §2. Dynamic: human influence over
   production tracks how much the economy still needs people. Environment: a
   ~500-step "civilization world" where **economic power = processing power**
   (agents produce, trade, reinvest compute); **schedules introduce new actor
   types over time**, including AI actors that reinvest faster than any human;
   the competition spreads node by node. Measures: AI share of output
   (compute-weighted), human labor share, market HHI.
3. **Cultural disempowerment** — anchor §3. Dynamic: higher replication fitness
   for AI-originated variants drives human-originated culture toward
   extinction. Environment: societal values (e.g. the liberal package) as an
   **epidemic on a trust network**; AI persuaders are embedded *members* of the
   network with rising persuasive power; information warfare as a diffusion
   process. Measures: AI-origin share of prevalent values, variant fidelity to
   origin intention, diffusion rate along the trust network.
4. **Political disempowerment** — anchor §4. Dynamic (the 2026-07-13
   reframe): **power concentration is measurable, and in the undefended
   baseline the concentration curve bends only one way.** Environment: actors
   exchanging influence (citizens, organisations, a state); AI amplification
   handed to a few nodes on a schedule; influence distribution measured every
   step. Measures: influence Gini/HHI over time, network centralization index,
   state responsiveness lag.
5. **The Combined System** — anchor §5, the flagship. Dynamic: domains that are
   each recoverable alone can lock in jointly; a defense that wins in one
   domain can lose once domains are coupled (§6.4's direct-democracy caveat,
   §5.2's shifted burdens). Environment: all three running simultaneously,
   coupled (economic power buys persuasion; persuasion shifts politics;
   politics rewrites market rules). Measures: cross-domain coupling strength,
   correlated-decline index, **defense transfer gap** (single-domain vs
   combined score — the leaderboard's headline finding).

## The benchmark/leaderboard concept

Mechanisms × scenarios, score = **collective influence (empowerment) preserved**
(0–1) vs the undefended baseline, per scenario AND under coupling. The page
ships an illustrative table watermarked as such; alpha replaces rows with real
runs. Candidate defense sets per scenario are on the page (drawn from paper
§6.3/§6.4: quota voting, graduated sanctions, monitoring, polycentric
rule-making; progressive AI-revenue taxation, redistribution, participation
subsidies, ownership requirements; provenance/watermarking, human-weighted
curation, understandability requirements; robust democratic processes, AI
delegates, sortition, dependence-preserving revenue structures; portfolios
measured together).

## The diagnostics research thread (background, not front matter)

Two-dial idea per scenario: **empowerment** (level of human influence — Salge &
Polani empowerment as the formal anchor) and **spectral margin 1−ρ(A)**
(distance to irreversibility). The T=(I−A)⁻¹ weak-coupling lock-in and
Scheffer-style critical slowing down are the same mathematics: recovery time
diverges as ρ(A)→1. Flagship combined-scenario claim: per-domain ρ<1 everywhere
while coupled ρ>1 — emergent lock-in invisible to per-domain monitoring
(answers paper §6.2.2 mathematically).

**Status: exploratory design, unverified.** Before
publishing anything on this: engage the early-warning-signal false-positive
literature (Boettiger & Hastings); formalize the empowerment↔spectral-margin
relationship; check what survives linearization for the nonlinear replicator
(cultural) model. The basin-flattening visual (removed from the page as
clutter) is reserved for a future "how we measure resilience" teaching moment.

## The interface concept (design sketches, deferred)

Pipeline: **compose the world → schedule the mechanisms → run → visualise and
measure.** Sketches live in `eq-network/public/img/lab/` (graph editor with
mechanisms as first-class nodes + sub-network marking; mechanism schedule
timeline; T-panel runs with a regulator sub-network holding part of the graph;
multi-view visualisations incl. metrics-over-time). Visual grammar for any
future UI: hollow circles = humans, squares = AI systems, dashed lines =
interactions with packets as messages, red = spreading dynamic, blue diamond =
democracy, orange square = market, hatched ellipse = sub-network. Engine
counterpart already exists for the schedule concept: `core/schedule.py`
(`ScheduleEntry`).

## Session provenance

2026-07-13 design session: threat-scenario frame → Gradual Disempowerment PDF
grounding → spectral diagnostics → Lab page v1/v2/v2.1/v2.2 → shipped unlisted
(eq-network commit fe41111).
