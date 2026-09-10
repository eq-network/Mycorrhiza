# GD game dynamics — diagnosis and candidates for the design review

> **ABANDONED 2026-08-01. Do not act on the candidates below.** The games this
> review was trying to repair were judged not worth continuing. Every candidate here — re-timing
> the world, more windows, journey scoring, the live sandbox — is a repair to
> route A or a move to route B, and both routes are dead. The measured
> diagnosis in §2 stands as a description of what was built; the treatment
> plan does not. The record and the rule against restarting either route are in
> [docs/gd-game-postmortem.md](gd-game-postmortem.md).

*2026-07-31. Material for a staged game-design review (the brief: a debate
between designers from different schools). §2 is measured; §3 is candidate
directions, all design conjectures; §4 is constraints the review must respect;
§5 is what the panel is being asked. Siblings: docs/gd-game-design.md (the
tree game, built), docs/remote-engine-design.md (the live sandbox direction,
decided 2026-07-31), the v3 UI brief in eq-network
(apps/site/docs/lab-game-v3-design.md, shipped). Probes in this doc: 4 seeds,
harbor dials unless stated, run this day; none are committed claims.*

## 1. The owner's critique being answered

The v3 UI landed; the complaint is the game underneath it: the opening moves
too fast, the choices do not feel open, and there is not enough gameplay. The
wish, in the owner's words: complexity from the combination of many simple
choices with an underlying simulation engine.

## 2. Diagnosis (measured)

**The game is over before the first decision.** In the shipped trees the
majority-flip of human attention onto AI listeners happens at tick 27 — on
every path, in all three towns, out of 73 paths total. The player's first
real window is tick 100. The rehearsal is tick 75. Attention share falls
0.78 → 0.39 by tick 75 and is flat (± 0.01) for the remaining 325 ticks.
More than half of all wealth-share loss happens before the first window.

**The flip is an arrival shock, not a contest.** Probes: stretching arrivals
(first at 40, spacing 45) moves the flip to ~44 — it tracks the first
arrival by a handful of ticks. Cutting the AI seed capital 3.0 → 0.5 moves
it to ~57; to 0.1, ~66. The strongest defense in the deck (reach cap 0.9)
enacted *before* the flip delays it by about ten ticks and no probe
prevented it: once arrived, AI income via capability growth buys the room
regardless of the war chest.

**Attention is nearly memoryless; the hysteresis lives elsewhere.** Cap at
t=60 versus cap at t=200: by t=399 both end at attention 0.51 and wealth
0.31 — the endgame forgets when you acted; only the middle of the run
differs. The model's genuine irreversibility is (a) the
enforcement/affordability ratchet — capacity spent or eroded prices cards
out of later trays — and (b) wealth compounding. The scene dramatizes the
listening arrows; the stakes live in the meters.

**Scoring erases timing.** The tree scores late-window metric levels, which
are near-equilibrium values — precisely the quantities least sensitive to
when the player acted. Composite spreads, best path minus wait: 0.057
(harbor), 0.032 (boomtown), 0.070 (commune), against a passive drift of
roughly 0.4. And no path scores below waiting — the worst path IS the wait
path in every town. There is no way to make things worse, so there is no
risk, so "choices" reduce to a checklist.

**The endgame is dead air.** Boomtown is inert from ~t=200 (enforcement
0.000, wealth share 0.002); commune is flat from ~t=250; harbor drifts
slowly. The final third of most runs asks the player to watch an
equilibrium.

## 2b. Two probes that bear directly on the candidates (8 seeds, harbor)

Run after the diagnosis, at the game's real seed count, on the wait path and
on the reach cap enacted at three different ticks. Medians across seeds;
no CIs computed, so these are ordering claims only.

**Journey scoring roughly triples the timing signal.** Comparing the cap
enacted early against the cap enacted late — the same card, the only
difference being when:

| world | endpoint score, early − late | journey score, early − late |
|---|---|---|
| shipped harbor | +0.014 | +0.045 |
| C1 re-timed | +0.019 | +0.043 |

Journey score = mean of the three human shares over the whole run; endpoint =
their mean over the last 50 ticks, which is what the shipped tree reports.
The ordering is the same in both worlds and in both scorings — acting early is
better — but under endpoint scoring the difference is a third of what the
player actually lived through. This is a metric change, not a model change,
and it is the cheapest available fix for "timing does not feel like it
matters".

**Only in the re-timed world can the player act before the flip.** Shipped
harbor flips at t≈26 and the first window is t=100, so no card can ever
precede it; the cap at t=45 and at t=250 both leave the flip at 26. In C1
the wait path flips at t≈56, and the cap enacted at t=45 — before the flip —
moves it to t≈68. Acting early visibly delays the town turning; acting late
cannot. That is the decision the shipped game structurally cannot offer.

## 2c. Decision — the game scores the journey (2026-07-31)

> "We score the entire journey. The player's choices DO matter — that is the
> point of the game."

Decision, not conjecture. Consequences, dated here:

- **C3 is adopted.** Built the same day: `ledger_society/metrics.py` gains a
  `journey_*` family (income, wealth, attention, power, composite,
  enforcement) computed over the whole run beside the existing late-window
  metrics, which are unchanged and still exported. Suite green.
- **C4 is bounded by the axiom.** The flip may remain unpreventable, but no
  reading of it that makes the player's choices decorative survives. If the
  two cannot coexist under probe, the flip's unpreventability gives, not the
  axiom.
- **The shipped solution tables are superseded** for the game's purposes:
  every claim about which path is best was computed in the late window. The
  trees must be re-scored before any of them is quoted again. This obsoletes
  the current committed harbor expectations, which is a declared revision
  under this decision, not a retune.

## 2d. Journey scoring re-scored: it does not by itself restore downside

All 73 shipped paths re-run at 8 seeds and scored under both windows, to
answer the question §2 raised — under the new scoring, can you lose?

| town | paths | wait rank, late | wait rank, journey | paths below waiting, late → journey |
|---|---|---|---|---|
| harbor | 13 | 13/13 | 12/13 | 0 → 1 |
| boomtown | 7 | 6/7 | 6/7 | 0 → 1 |
| commune | 53 | 39/53 | 21/53 | 14 → 31 |

**Where the tray stays live, journey scoring creates real downside.** Commune
affords 4/3/3 cards across its windows; under journey scoring waiting falls
from near-worst to the middle of the field, and 31 of 53 paths now do worse
than doing nothing. That is a game with a way to lose.

**Where the tray empties, it does not.** Harbor and boomtown move by one
rank, and their single sub-waiting paths beat the wait path by 0.0004 and
0.0005 — inside anybody's noise. The reason is structural: a player cannot
overspend when there is nothing left to buy. Downside needs affordable
choices that can be wrong, which needs the tray to stay live — the C1/C2
re-timing and more windows — or a continuing cost on enacted cards, which is
what Rafael's per-tick upkeep inside the tree and Ilse's "the band must be
able to go red" both name.

**A presentation warning that came with the result.** The journey window
averages over the whole run, including the long opening stretch that is
identical on every path, so it compresses absolute spreads: harbor's
best-minus-wait is 0.057 in the late window and 0.011 in the journey window.
Journey scoring amplifies *timing* differences (§2b) while shrinking the
*range* the debrief has to display. A raw journey number will read as noise.
It has to be shown as distance from the wait path — which is precisely the
shaded band between the player's curve and the ghost line that the design
review converged on independently, and under journey scoring that area is
the score rather than an illustration of it.

## 2e. The scoreboard hides the game — the largest finding here

Chasing a round-2 conjecture about a dead final third (falsified: the spread
between paths keeps widening to the last tick), the measurement turned up
something bigger. Spread across all paths of each shipped metric, per town:

| metric | harbor | boomtown | commune |
|---|---|---|---|
| human wealth share | **0.449** | **0.477** | **0.446** |
| enforcement level | 0.247 | 0.115 | 0.342 |
| human attention share | 0.082 | 0.072 | 0.107 |
| human power share | 0.057 | 0.079 | 0.082 |
| **composite (the headline)** | **0.057** | **0.032** | **0.101** |

The player's choices move the human wealth share by up to 0.45 and the
institution's enforcement by up to 0.34. The number the game reports as the
score moves by 0.03 to 0.10 — an order of magnitude less, and in boomtown a
factor of fifteen.

The cause is in the definition, not the model. `composite` is the mean of
income share, attention share and power share. Two of those three are sticky
ledgers that barely respond to anything a player does, and the average
excludes wealth — the ledger the cards actually move — and enforcement — the
resource the player actually spends. So the scoreboard is built almost
entirely out of the quantities least sensitive to play.

This is the numerical form of "the choices do not feel open". Every earlier
conclusion in this document that rests on composite spreads — including "your
entire agency is a rounding error on the drift" — is a statement about the
metric at least as much as about the model. The drift is real; the claim that
the player cannot affect it was measured through an instrument that averages
the effect away.

**Implication for the game, following directly from the 2c decision.** The
headline the debrief leads with should be the journey wealth share and the
journey enforcement level — the two quantities that carry the consequences of
play — with the composite kept as a secondary research readout. Both journey
metrics are built (2c). No model change is required, and none of the register
work that depends on `composite` is touched.

## 3. Candidate directions (design conjectures, typed tuned-for-legibility)

- **C1 — re-time the world.** Probed dial set: first_arrival 40,
  arrival_spacing 45, update_rate_w 0.04, ai_seed_capital 0.5. The drift
  then unfolds across the whole run (wealth share 0.97 at t=60, 0.50 at
  t=150, 0.29 at t=300 on the wait path) and the flip lands ~t=57.
- **C2 — put decisions where the drama is.** First window before the flip
  (~t=50 in the C1 world), rehearsal ~t=35, and more windows — five instead
  of three. Affordability pruning keeps the tree in the low hundreds of
  paths; the sizing napkin in gd-game-design.md §4 allowed for this from the
  start.
- **C3 — score the journey, not the endpoint.** Make the composite a
  whole-run mean (the area under the human-share curves) instead of a
  late-window level. Acting early then scores exactly as much as it should
  feel like it matters; the town lived through those ticks. One metric
  change, honest, and it survives the endgame-equilibrium wash-out.
- **C4 — the flip is unpreventable; say so.** No affordable mechanic stops a
  majority of the town listening to the newcomers first. Make delay the
  stake: the story names it — you will not stop them being heard; you decide
  what it costs them and how long the town keeps its own voice.
- **C5 — the live sandbox is where openness lives.** Already decided and
  half-built: continuous levers (levy, repair, reach cut, sortition) set
  over time in policy tabs, paying per-tick upkeep out of enforcement, served
  by the stateless engine endpoint (service/app.py, built; deploy pending).
  Upkeep gives the game its missing downside: over-extend and the
  institution collapses under your own program. The tree game remains the
  guided opening.
- **C6 — more openness inside the tree game, short of live.** More windows
  (C2), and possibly card intensities (small/large levy) at the cost of tree
  width. Re-enactment stays out — it is the sandbox's job.

## 4. Constraints the review must respect

- The browser computes nothing. A choice is either precomputed
  (branch tree) or answered by the live engine endpoint. No client-side
  dynamics, ever.
- No invented mechanics: every lever is an existing model quantity; a new
  dynamic added for the game must be declared where its effect is claimed.
- Parameters are typed (anchored / tuned-for-legibility / arbitrary-but-swept)
  and mismatches are reported, never silently retuned. Committed harbor
  expectations may be *revised with a dated note*, not quietly changed.
- The modal visitor stays under two minutes and leaves with one image; the
  player who stays should find a real game. Both audiences are real.
- One scenario must not overfit the model to one story: the register models
  remain the research objects; the game is a window onto them.

## 5. What the panel is asked

1. Where should this game's interest live — timing (when to act),
   composition (which motions), or intensity (how hard) — given the model's
   real hysteresis is budgets, not attention?
2. Is the guided-tree-opening → live-sandbox split the right shape? What is
   the minimum each half owes the player?
3. How does waiting become a real choice — sometimes right — without
   inventing model mechanics? Does journey scoring (C3) plus upkeep (C5)
   get there?
4. What does the first sixty seconds owe a player who knows nothing? Where
   exactly does the first interactive beat belong?
5. Endpoint versus journey scoring; the solution table's role in replay;
   what "winning" should mean in a game about managed decline.
6. The empty tray is the thesis. How is inevitability staged so it lands as
   tragedy rather than as the game ignoring the player?

## 6. What a critic should press on

The probes are 4-seed medians on harbor's dials and none is a committed
claim; C1's numbers will move under the full seed batch and the other towns'
dials. Journey scoring changes what the shipped solution tables mean and
obsoletes the current committed expectations — adopting it is a declared
revision, not a tweak. The panel itself is a simulation: its output is
design ideation, exploratory by definition, and lends no real person's or
studio's authority to any decision it inspires. And the sandbox's upkeep
coefficients will be tuned until over-extension is *possible* but not
*inevitable* — that tuning carries its type and must be probed before any
claim about difficulty is written down.
