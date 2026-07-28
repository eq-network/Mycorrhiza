---
type: cold-read-reaction
parent: "[[README]]"
draft: "lab modelling-assumptions cards"
reader: squazzoni-like-jasss-editor
based_on: "a reader like Flaminio Squazzoni (JASSS; ABM standards)"
scene: busy-hostile
attention_budget: low
read_completion: "commons card in full, combined card skimmed, three cards unopened"
outcome: skimmed
dispatched_at: 2026-07-27T11:45:00+02:00
status: complete
tags: [cold-read, reaction]
---

# Reader like Flaminio Squazzoni — cold read of "lab modelling-assumptions cards"

## The scene
18:40, lights half off, bag on the shoulder already. Fourth desk rejection of the day still open in another tab — some poor postdoc's 40-page LLM-swarm paper with zero sensitivity analysis, "emergent cooperation," no ODD in sight. Marco pinged the group chat: "AI-safety people doing ABM — actually with assumption cards!" I have maybe ninety seconds before I close the laptop and this becomes tomorrow's problem, or it becomes nobody's problem.

## Reading it (stream of thought)
0:00 — Cards. Dashed borders, "MODELLING ASSUMPTIONS," chips that say LIVE and IN-DESIGN. So this is a web thing, not a submission, good, different rules apply, I relax by maybe four percent.

0:10 — I go straight for Commons, it's mine, I've refereed forty papers that get Ostrom wrong. "One aggregate, non-spatial resource pool with logistic regrowth — the closed-form Ostrom anchor." Fine. Naming what you left out — space, heterogeneous access, prices — in the same breath is the single thing that separates a serious modeler from a demo. Half a point, immediately.

0:25 — "Households act only through fixed-behavior AI delegates... The blend (alignment) is a knob calibrated so the undefended baseline collapses — not a measured quantity." I actually stop scrolling here. This is the sentence that never appears in the papers I reject. They tune until the model does the thing and then write the method section as if the parameter fell from the sky. This one says it out loud, in public, on the marketing page. Va bene. Noted, genuinely.

0:40 — "Delegates never learn, so sanctions can confiscate but not deter." — leaves out deterrence, "tracked in the engine backlog." Okay, that's an honest scope statement, not a claim they've solved commons governance. Good instinct. Still — no learning delegate means whatever "collapse under greedy AI" story this produces is a foregone conclusion, not a discovery. I'd want to know if they know that too. The card sort of tells me they know.

0:55 — Governance line, "median vote... fixed coin-flip," v0-simple. Fine, disclosed, moving on.

1:05 — Last bullet: "influence is measured causally here: paired same-seed runs with shifted preferences, differenced." Now that's a real methodological move — a counterfactual pair under a shared seed is at least a defensible identification strategy, more than most of what crosses my desk. I'd want the actual estimator and whether it's differenced in outcome-space or in policy-space, but the instinct to say "measured causally" and mean an actual paired design, not just "we varied a parameter and eyeballed it," is correct.

1:15 — I jump to Combined because that's the headline, that's what Marco actually wants me to react to. "Composition, not a new model... inherits verbatim." Okay, architecturally sane — no silent extra assumptions bolted on at the seam, in principle.

1:30 — "The gains' magnitudes ARE the scale of the result; only sign and ordering claims are robust." — leaves out measured coupling strengths. Stop. Read again. This is — unusual. This is a team telling me, unprompted, exactly where their result stops being trustworthy. Most people bury this in appendix table 14, if it exists at all. I believe this sentence more than I believe most abstracts.

1:40 — "Per-domain dials default mild, so any joint decline is attributable to the coupling rather than to stacking three separately lethal baselines" — that's a real control, that's thinking like someone who's been burned by a reviewer asking "how do you know it's not just additive."

1:50 — Last line, spectral lock-in "unverified research thread... not presented as a finding." Good — that's the sentence that keeps this off my desk-reject pile if it ever were a submission, because it isn't dressing a conjecture as a result.

1:55 — Laptop's still open. I look for the word "sensitivity" anywhere on the page. Don't see it. Look for ODD, or anything ODD-shaped — purpose, entities, process order, submodels laid out as a protocol. Not here, this is cards, not a protocol, I knew that going in.

2:00 — Decision point.

## After (the collapse)
- **How far I actually read:** Commons card end to end, Combined card in the same pass, mostly the calibration/coupling bullets. Never opened economic, cultural, or political.
- **What I think they want from me / my takeaway:** They want credit for candor — for saying "this is tuned, here is exactly how, here is what you should NOT conclude from it" instead of hiding it. On that narrow ask, they get it from me.
- **Did I believe it? What lost me:** I believed the disclosures more than I believe most methods sections I referee this week — that's not high praise, it's a low bar, but they clear it. What I don't have, and what would decide whether this is science or a nicely-worded demo, is: is there a sensitivity sweep on those linear gains anywhere, is there an ODD-equivalent write-up behind the cards, and does "the closed-form Ostrom anchor" mean they reproduce a known qualitative regime (tragedy under no governance, stability under quota) or just that the equation looks like Ostrom's. The cards assert the second thing implicitly; they don't show me the first. A card is not a validation section.
- **What I'd actually do next:** Close the laptop. It goes in a tab, not the trash — that's rarer than it sounds from me at 18:40 on rejection number four. If Marco or anyone asks me tomorrow whether this is worth an hour, I'd say: worth thirty minutes to find out if the sensitivity analysis exists behind the site, because if it does, this is the first "AI-safety ABM" thing all year I wouldn't reflexively bin.
- **Would it land differently if I were fresh / less busy?:** Fresh, I open all five cards, I go looking for the repo, and I check whether "median vote becomes next tick's quota" actually reproduces anything from the commons literature or is just plausible-sounding. Fresh, the honesty about tuning stops being enough on its own — I start asking why the knob has to be hand-set instead of fit, and whether "sign and ordering claims are robust" has ever actually been stress-tested across the hand-set range or just asserted. Tired, the fact that they said it at all is most of what I needed to not close the tab.

**Outcome: skimmed — read two of five cards closely, kept it in a maybe-tomorrow tab instead of closing.**
