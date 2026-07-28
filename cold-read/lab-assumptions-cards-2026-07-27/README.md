---
type: cold-read
draft: "lab modelling-assumptions cards"
draft_path: "C:\\GitHub\\eq-network\\src\\content\\lab.ts (assumptions arrays)"
audience: "lab-page visitors: Ostrom-tradition scholars, ABM reviewers, GD-paper readers, smart high schoolers (the register's own bar)"
intended_ask: "A visitor at the register's bar — a smart high schooler who just played the playground — should understand what each toy deliberately leaves out, trust the lab more because of the honesty, and be able to connect the assumptions to the sliders they played (why are these dials the meaningful ones?)"
readers: 4
dispatched_at: 2026-07-27T11:45:00+02:00
completed_at: 2026-07-27T14:55:00+02:00
status: complete
tags: [cold-read]
---

# Cold read: lab modelling-assumptions cards

**Author's worries, verbatim:** (1) "too complicated and not necessarily the core
point... missing something... not intuitive"; (2) "I should see more collapse if
it's actually the underlying numbers... they feel almost arbitrarily defined...
suspicious the numbers are not fully collapsing."

Companion evidence: the collapse audit run alongside this panel
(`experiments/assumption_audits/collapse_audit.py`) — culture's floor is entirely the constant
native `recovery` (share 0.59 → 0.008 at recovery 0); the political floor
(~0.50) is structural (frozen AI-listens-to-citizens rows; insensitive to
self_weight, update_rate, and amplification); the economy genuinely collapses
(income share still sliding at T=1500, 0.018 in the all-out run); opinions feed
back into nothing; undefended politics is a sink.

## The panel
| # | Reader | Based on | Scene | Attention | Outcome |
|---|--------|----------|-------|-----------|---------|
| 01 | [[01-janssen-like-ostrom-scholar]] | a reader like Marco Janssen (ASU, Ostrom collaborator) | fresh-careful | high | engaged — all 5 cards, card 1 twice |
| 02 | [[02-maya-high-schooler]] | composite, 17, played sliders first | distracted-typical | medium | skimmed — Commons full, half of Combined, closed |
| 03 | [[03-squazzoni-like-jasss-editor]] | a reader like Flaminio Squazzoni (JASSS) | busy-hostile | low | skimmed — 2 of 5 cards, kept the tab |
| 04 | [[04-gd-coauthor-like-ally]] | composite GD co-author (paper content only) | substantive | high | engaged — Combined+Economic full, rest skimmed |

## Intended vs. received

- **Janssen-like** — received: "a team with admirable bookkeeping honesty testing
  an AI-delegate mechanism *wearing a commons costume*." The alignment knob +
  coin-flip defection read as circular tuning ("a thermostat, not a finding");
  and the single strongest lever in the real CPR literature — communication /
  cheap talk — isn't even *named* as an omission. The "leaves out" discipline
  is supposed to catch exactly that; its absence is the tell the authors don't
  know what they don't know here.
- **Maya (the bar)** — received: "the collapse I watched was a setup, not a
  forecast — and at least they admit it." Three of five Commons bullets landed
  *because they matched sliders she had touched*. But the card never answered
  her actual question — why nothing hits zero — so she left MORE suspicious of
  a hidden floor, not less ("probably go back and try to make the pond hit
  exactly zero to test the floor theory myself"). Which is, verbatim, the
  author's own worry #2. Jargon bounce: "paired same-seed runs, differenced"
  (three re-reads, skipped). Trust hit: Combined's IN-DESIGN chip after ten
  minutes of playing its polished sliders — "what else on this site is labeled
  wrong?"
- **Squazzoni-like** — received: "candor that clears the desk-reject bar" — the
  calibrated-knob admission and sign/ordering-only bullet are "sentences that
  never appear in the papers I reject." Kept the tab. But: a card is not a
  validation section — no sensitivity sweep named, no ODD-shaped protocol
  behind it, and fresh he'd ask whether "robust to the hand-set range" was
  tested or asserted. (It HAS been tested — the audit and tuning sweeps exist —
  the cards just never say so.)
- **GD-co-author-like** — received: "a real closed loop, not a cascade — but
  the plumbing for the §5 flywheel, not yet the flywheel." Deep cuts: the one
  moment the build approaches irreversibility (spectral lock-in) is an unwired
  comment; the one true absorbing state (universal AI culture) has a knob
  installed specifically so it's never reached — un-swept, uncalibrated — "we
  spent a paper arguing you can't necessarily install a floor under the thing
  disempowering you; this model has installed exactly that floor." §2's
  consumption channel and §4's state-dependence are structurally absent, and
  Political measures *voice*, not state dependence, unflagged. Nothing in the
  build is a *basin*; everything is a rate or a level.

## What survived every read

The "leaves out" candor discipline. Every reader, at every attention level,
credited it and kept reading because of it — it is the load-bearing design
choice and must survive any rewrite. Second survivor: bullets phrased in the
language of things the reader had just touched (Maya's alignment/sanction/
coin-flip hits; the κ=0 sentence — "that's the kind of thing I wish more of
this was").

## What only the careful reader saw

The causal-twin instrument (Janssen and Squazzoni's favorite methodological
move; Maya's hardest bounce), the Friedkin–Johnsen degeneracy argument, the
"mild dials isolate the coupling" control. The cards' best *content* is locked
behind their worst *sentences*.

## Where readers diverged

- Candor-as-credit (Squazzoni, Maya partially) vs candor-as-exhausting
  ("somewhere between admirable and a legal disclaimer" — Janssen). Writing-
  fixable: candor lines should carry information (what was swept, where the
  floor is), not just disclaimers.
- Knob-tolerance: Squazzoni forgives hand-set values if swept; Janssen wants
  them *fit to CPR experiment data*; GD-ally wants the reversion knob swept
  down to find where the floor gives way. All three converge on: **say what
  was swept and what the sweep showed.** (Their-priors residue: Janssen's
  demand for experimental calibration is a research program, not a card fix.)

## If you change one thing

Answer Maya's question on every card, because it is also the author's
question and now has an empirical answer: **name the floor.** One line per
scenario in slider language: "if you crank everything against the humans, X
still won't go below Y — because of Z [constant/structure]; remove Z and it
goes to zero; we checked." The collapse audit provides Z for every scenario
(culture: constant recovery; politics: AI actors frozen listening at citizens;
economy: no floor, it keeps sliding; consensus: the FJ anchor). This one line
converts the cards' candor from disclaimer into explanation, answers "was the
demo rigged" in both directions, and is the sentence every one of the four
readers was independently looking for.
