"""Suites — named readings of the environments catalog.

A suite is not a new mechanism or a wrapper: it is a plain dict naming which
registered environments together make one research program's claim, so code,
docs, and experiments can reference the set by one name (the same
"register is a reading of the catalog" convention as docs/model-register-design.md).

GRADUAL_DISEMPOWERMENT v0.1 — the three working-paper models (papers live in
the vault, ``…/CI Library/papers/``; the repo stays code-only) plus the v0
coupled composition they are meant to replace:

- ``economy``  -> ``capital_economy``   (WP1 "Where Does the Money Go?" — SFC
  circular flow, money conserved at every r, conservation probe on every run)
- ``culture``  -> ``influence_exchange`` (WP2 "Who Fills Your Head?" — used
  UNCHANGED as the belief/attention substrate; the same env is also alpha
  scenario A4's political substrate. One env, two readings — deliberate,
  stated on WP2's vault README.)
- ``politics`` -> ``delegative_polity`` (WP3 — delegation capture, the
  power-weighted median, taxation, and the entrenchment lock-in dial)
- ``coupled``  -> ``coupled_society``   (v0, κ-gated couplings — demoted by
  docs/ledger-design.md; kept as the modulation-implementation baseline the
  ledger rewrite must be compared against)

The resource map that grounds the coupled rewrite is docs/gd-suite-v0.1.md.
"""
from __future__ import annotations

GRADUAL_DISEMPOWERMENT = {
    "version": "0.1",
    "members": {
        "economy": "capital_economy",
        "culture": "influence_exchange",
        "politics": "delegative_polity",
        "coupled": "coupled_society",
    },
}

# name -> suite dict; a suite's members must all resolve in environments.REGISTRY
SUITES = {
    "gradual_disempowerment": GRADUAL_DISEMPOWERMENT,
}
