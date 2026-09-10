# The GD game — postmortem, and the two routes we do not repeat

*Deposited 2026-08-01. This document supersedes docs/gd-game-design.md,
docs/gd-game-dynamics-review.md, docs/gd-game-three-families.md and
docs/remote-engine-design.md, all four of which are abandoned as game designs
and kept only as the record of what was tried. Read this before proposing
anything that calls itself a game.*

## 1. The decision

The games built on 2026-07-31 were judged not worth continuing, and the
premise behind them was wrong from the start. Trying to make the premise
change from the engine side did not work.

This is not a note that the games need tuning, more windows, better scoring,
or a nicer front end. Two full attempts were made, in two opposite
directions, and both produced something nobody wants to play. The failure is
upstream of every parameter either attempt exposed.

## 2. The two routes, and what each one actually produced

**Route A — the precomputed branch tree.** `docs/gd-game-design.md`,
`experiments/gd_game/`. Three intervention windows, five cards, affordability
pruning, every path enumerated as a full run from t=0 and shipped as a
finished tree the browser walks. What it produced: a game with about three
choices in a whole run. That is the ceiling of the route, not a bug in this
instance of it. Precompute makes the choice space something you must
enumerate ahead of time, so it must stay small, so the player gets a short
menu. Widening it does not help: more windows and card intensities buy a
wider tree and the same experience, because the thing that is missing is not
the number of options.

**Route B — live levers behind an endpoint.** `docs/remote-engine-design.md`,
`docs/gd-game-three-families.md`, `service/app.py`,
`mechanisms/families/{economy,culture,politics}.py`. Continuous levers in
three policy tabs, plans carried as data so one compiled program serves every
plan, a stateless server re-running the trajectory from t=0 per request. What
it produced: a form full of parameters a new player does not understand.
Again the ceiling of the route. The levers are model internals — `gamma_w`,
`repair_rate`, an allocation vector over five columns — and no amount of
labelling makes a stranger's first minute with them anything other than a
parameter form. Submit, wait for a run, read a chart, guess again.

The two routes are the two halves of the same dead space. Route A fixes the
interaction ahead of time so it can be precomputed; route B opens the
interaction and pays for it with a round trip. Neither is the thing.

## 3. The premise that was wrong

**We derived a game from the engine we had, instead of designing a game and
then asking what engine it needs.** Every design decision on record was made
inside a constraint the engine imposed rather than in service of anything a
player would feel.

The engine is a batch research simulator. A run is `lax.scan` over a fixed T,
compiled and executed as a whole. There is no live world to poke. Given that,
exactly two game shapes are reachable: precompute the choices, or take a
request and re-run. Route A and route B are not two ideas that happened to
fail — they are the complete enumeration of what that engine permits, and we
built both.

**The missing ingredient is real-time editing.** For a game to be fun you
change something and watch the world respond, now, continuously, in the same
motion as the change. That is what neither route has, and it is not
recoverable by tuning either one.

**The attempt to move the premise from the engine side did not work, and this
is the part worth remembering.** Plans became data rather than closed-over
config; six substrate fields were promoted to per-tick ports; three lever
families were built and sealed to bit-identity; journey metrics were added
because endpoint scoring erased timing; a service was written. All of that
made the engine *more configurable*. None of it made it *live*. A batch
simulator with a wider input surface is still a batch simulator, and the game
on top of it is still submit-and-wait. Configurability is not interactivity.

**The engine work was driven by game needs, not research needs.** The claim
discipline in CLAUDE.md exists for exactly this pattern — a dynamic that
exists because the game needed it must say so wherever its effect is
discussed. Anything still standing from this work carries that provenance.

## 4. The rule going forward

**Do not restart either route.** Not a fourth town, not five windows instead
of three, not better card art, not a nicer policy-tab layout, not a
deployment of the service. If a future session finds itself widening the
branch tree or adding a lever to a policy tab in the name of making the game
better, it is repeating this and should stop.

**Nothing is built until there is a design.** The next attempt, if there is
one, starts from what the player does second by second and what the world
does back, and only then asks what engine can serve that. If the answer is an
engine this repo does not have, that is the finding, and it is a cheaper
finding than another day of building. Design first, engine second — the
inverse of what was done here.

**The engine is not a game engine and there is no obligation to make it
one.** `src/cilib` is a research simulator and it is good at that. The wish
for a playable artifact is a real wish and it is not a reason to bend the
engine toward it. Any future game is a separate design question with its own
premises, and it may well conclude that it needs a different substrate
entirely.

## 5. What is on disk, and what its status is

Recorded so nobody has to guess, and so nobody reads the presence of working
code as a reason to revive the game.

| artifact | status |
|---|---|
| `docs/gd-game-design.md` | abandoned; route A's plan |
| `docs/gd-game-dynamics-review.md` | abandoned; route A's diagnosis and the journey-scoring decision |
| `docs/gd-game-three-families.md` | abandoned as a game design; §2 and §3 contain engine findings |
| `docs/remote-engine-design.md` | abandoned; route B's plan |
| `experiments/gd_game/` | dead; the tree generator and its three towns' artifacts |
| `service/` | dead; not deployed, and not to be deployed for this |
| `mechanisms/families/`, `mechanisms/interventions.py` | on disk, engine-side, provenance is the game |
| `ledger_society` per-tick ports and `journey_*` metrics | on disk, engine-side, provenance is the game |

Two things found along the way are engine findings rather than game findings,
and they survive the decision as open questions about the model:

- **`belief` is causally disconnected from every scored outcome.**
  `susceptibility` moved no ledger metric to four decimal places, and reading
  `dynamics.py` says why: `belief` is written and read by `pool_belief` and
  by the trace, and by nothing else. Either it gets a path into the ledgers or
  culture in `ledger_society` means the attention kernel's shape and nothing
  more. This has WP2 implications and is an open ruling.
- **Sealing conventions have to be pinned in the pure tier.** A neutral plan
  that was bit-identical eagerly drifted by about 1 ULP under `lax.scan`,
  because a fused tick does not round two references to the same producer
  alike. `make_policy_levers` still carries the weaker form. The per-family
  suites were green while testing the wrong tier.

Neither of these is a reason to revive either route.

## 6. What a critic should press on

The decision was one person's, formed after one day of building and without
players in front of either artifact, so "not worth continuing" is a judgment
about the work rather than a measurement of reception — which is exactly the
standard this repo applies to the design conjectures the game docs were full
of, now applied to their conclusion. The claim that route A and route B
exhaust what a batch engine permits is an argument, not a proof; a third
shape may exist and nobody has looked for it, because the instruction is to
design before looking. Calling real-time editing the missing ingredient is
itself an untested design conjecture and the next design owes it the scrutiny
this one did not get. And keeping the lever families and the journey metrics
on disk after abandoning their purpose leaves code whose only justification
is a game nobody is building, which is a debt this document names rather than
settles.
