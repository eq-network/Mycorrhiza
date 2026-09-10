# Simulation standards

What a claim made with this library has to show before it travels, and the tools the
library gives you for showing it. Companion to [CONTRIBUTING.md](CONTRIBUTING.md) (how
code enters) and [EXTENDING.md](EXTENDING.md) (how a building block is made). This
document is the norm. The paper *Evidence Standards for Computational Mechanism Design*
(Hallgren, in preparation) is the argument for it.

**Status: v0, 2026-08-24. Open for comment. Versioned by pull request; the changelog
at the bottom says what moved.** If you think a rule here is wrong, open an issue that
names the claim it would have blocked or let through. This is a standard we are
proposing, not one we are enforcing on anyone else.

## The one idea

A simulation can be made to say almost anything. Tune the agents, pick the seed,
choose the frame, and cooperation appears or vanishes on demand. So a result from one
model, one agent type, one seed, one reference frame is not evidence yet. It is an
artifact until it survives variation of the things you are unsure about.

Elinor Ostrom's rule for institutions was that no single evidence stream suffices:
she used formal analysis, laboratory experiments, and field cases, and trusted what
held across all three. We extend that rule to computational settings. **Vary what you
are uncertain about; trust what survives.**

## What a claim carries

A claim that ships from this library carries the following. None of it is a gate.
It is what a reader needs to re-run your reasoning and disagree with it precisely.

1. **The claim, as a counterfactual ordering, never a prediction.** "Under mechanism
   M the desirable outcome holds from a larger set of starting conditions than under
   M-prime" is a claim. "This will happen" is not.

2. **The frame.** What was intervened on, over which distribution of starting
   conditions, measured by which observable. Reference-frame choice moves numbers,
   sometimes by a lot. State it every time.

3. **What was varied, what was held fixed, and why.** The library supports two axes
   directly:
   - **Agents.** Vary the decision procedure: fixed rules, learning agents, language
     models, people. Each kind relaxes a different assumption. Rule-based agents test
     the incentive structure; learners test whether the property survives when agents
     adapt instead of optimise; language models test whether agents arriving with
     priors behave as the simpler models predicted; people test everything else.
     There is no canonical ordering. Pick the agents that stress the assumption your
     claim depends on, and say which that is.
   - **Substrates.** Vary the model of the world on the other side of the game
     boundary. A register holds structurally different models of the same domain
     (three economies, three cultural dynamics); a claim about a mechanism is
     strongest when its effect holds across the register, not inside one member. See
     `docs/model-register-design.md`.

4. **Resilience, not point outcomes.** Report how much the system can be pushed
   before it changes regime, not only where it ends up. The library's shape for this
   is a basin fraction as a function of the mechanism dials, computed by sweeping
   perturbed starting conditions in one batched run. Phase boundaries are where that
   fraction crosses one half. Headline claims live on swept planes with regime
   boundaries; single trajectories are illustrations.

5. **Statistics as table stakes.** Comparative claims use paired seeds and a
   bootstrap confidence interval. A claim is resolved when the interval excludes
   zero, otherwise it is reported as unresolved. No magic seed counts, no point
   estimates travelling without error bars.

6. **Predictions on record before the sweep runs.** A mismatch between the committed
   prediction and the result is a finding and is reported as one. It is never tuned
   away.

7. **Parameters typed.** Every number is one of: anchored (to a cited source or a
   reproduced classical result), tuned for legibility (and said so), or arbitrary but
   swept. A number that is none of these does not appear.

8. **An assumptions card next to the model.** What the model says the world is, what
   it assumes, what it leaves out, which classical result it reproduces, and which
   dial the claim turns. Colocated so a fork carries its assumptions with it. The
   contract is in `docs/model-register-design.md`; an example is
   `src/cilib/environments/capital_economy/ASSUMPTIONS.md`.

9. **Figures regenerate from committed artifacts.** Nothing hand-drawn, no numbers
   typed into figure code. A figure you cannot regenerate does not ship.

10. **What you do not understand, written down in place.** An open question stated
    plainly beats a proxy metric or a confident sentence. Unknowns are first-class.

## Validation levels: how far a model has been validated

Every model in the library is placed on the same five-level scale. The level is declared, in the
model documentation and its tests, so a reader knows how far to trust it.

- **Level 0 (L0), runs.** Exists, compiles, produces trajectories.
- **Level 1 (L1), replicates a known result.** Reproduces at least one known result from the literature
  the model claims to belong to.
- **Level 2 (L2), calibrated.** With controlled agents, the environment
  demonstrably produces the target phenomenon and the observables demonstrably
  detect it. The instrument is calibrated before it is pointed at anything new.
- **Level 3 (L3), validated across agent models.** Physics frozen, a structurally different agent kind swapped in,
  and the level-2 signals measured for survival. Reproducibility splits here: the physics
  is bit-exact; agent behaviour is pinned to a provider and version, with churn
  disclosed.
- **Level 4 (L4), independently replicated or used.** Reviewed, replicated, or built on by someone outside the group
  that built it.

The scale ranks evidence, not agents. It says how thoroughly a model has been validated,
not how realistic its agents are. Agent kinds are a design space, chosen by which
assumption needs stressing; the level says whether that test has been run.

## Acceptance: robust across what you varied

The acceptance criterion the paper proposes, in words: a mechanism is robustly good
for an outcome if, under every agent model and substrate you tested, the outcome sits
in a basin whose resilience clears a threshold. Three things this does and does not
say:

- It is a robustness claim across the models you tested, not a guarantee about models
  you did not.
- It does not say the mechanism is optimal. Another mechanism may do better on
  equity, simplicity, or cost.
- The threshold is comparative, not absolute: set relative to the alternatives in the
  same domain. There is no universal number, and this document will not invent one.

Divergence across agent kinds is as informative as convergence. If rule-based
analysis predicts cooperation and learners defect, the mechanism's property depends
on a rationality assumption that does not hold under learning. That is a result.

## When the tools apply, and where they break

The resilience tools are mathematics. They apply with full force when three things
hold: there is an order parameter (a low-dimensional observable that stands for the
system's state), the dynamics are approximately stationary during measurement (agents
have converged), and the basin boundary is stated (what counts as a regime shift).
State all three when you use them.

They break, in ways that are fundamental rather than practical, in four places:

- **During learning.** The landscape is moving while agents update. Measure at
  checkpoints and report a resilience trajectory, or wait for convergence.
- **Agency.** Organisms do not lobby to change the rules of their basin. Participants
  in an institution do. Measured exit times are upper bounds when actors can choose
  to leave.
- **Hard constraints.** A mechanism enforced by construction (a contract that cannot
  be broken) has a trivially infinite basin. The question there is verification of
  the constraint, not resilience of the dynamics.
- **No normative content.** Resilience says how stable a state is, not whether it is
  good. A resilient bad equilibrium is lock-in, not a virtue. Pair every resilience
  claim with a welfare claim, and keep them separate.

## What this standard cannot do

It cannot prove optimality. It cannot predict what a real population will do; the gap
between computational and human evaluation is the largest and least understood. It
cannot yet handle institutions that rewrite their own rules while being evaluated. It
cannot settle whether an outcome that appears under every agent kind was genuinely
produced by the mechanism or quietly built into every agent; it makes that question
askable, not answered. It cannot replace domain knowledge about which mechanisms to
compare, which outcomes matter, and where the boundary sits. And it says nothing
about whether a stable state is desirable.

## Applying it to your own model

You do not need this library to use the standard. Any model can carry a frame, a
varied-and-fixed list, a typed parameter table, an assumptions card, committed
predictions, paired-seed statistics, and a declared validation level. If you apply it to a model
built elsewhere and something here does not fit, that is exactly the comment we want.

## Provenance and changelog

- **v0, 2026-08-24.** Distilled from the paper drafts (thesis v2 of 2026-02-27;
  sections 3 to 5) and from practice already in the repository: the validation
  validation ladder, the assumptions-card contract in `docs/model-register-design.md`, the
  experiment conventions in `experiments/CLAUDE.md`, and the writing rules in
  `docs/paper-style.md`. One reconciliation made explicit: the paper dropped a rigid
  agent hierarchy; the repository keeps a scale of validation levels (R0 to R4 in the internal documents and test names, L0 to L4 here). Both hold, because they
  rank different things.
