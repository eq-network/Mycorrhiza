# Prose rules — design docs, deposits, cards

These bind all repo prose: docs, `ASSUMPTIONS.md` cards, study READMEs, journal
entries. Agents writing project prose systematically overclaim; these rules are the
counterweight. Direct prose, smart-high-schooler register, no fluff.

- **A number travels with its context or it doesn't travel.** Quoting a result
  outside its home card requires substrate + seed count + error bar (or an explicit
  "no CI computed"). Without a CI it is a sign/ordering claim — magnitude language
  stays in the home card next to its caveats.
- **Model choices carry their provenance at the point of use.** A dynamic that
  exists because a channel was added or tuned until the effect appeared says so
  wherever the effect is claimed (e.g. the coupled flywheel arrows, added
  2026-07-27 precisely because sealed and coupled were identical without them).
- **Unknowns are stated in place.** "We don't understand this," written down, beats
  a proxy metric or a confident sentence.
- **Red-team before commit.** Substantive designs, deposits, and headline results
  get an independent adversarial pass (`/red-team`). Surviving attacks are recorded
  together with their answers — never silently patched.
- **Simulated-expert output (forest walks, round tables) is Stage 1–2** by
  definition and never lends a real person's authority to a claim in another doc.
- **History stays as record.** Dated deposits are not rewritten to match new
  direction; supersede with a status line, don't edit the past.
- **No invented vocabulary, no parenthetical asides, no test-count citations** —
  the paper rules (`docs/paper-style.md`) apply to internal prose too.
