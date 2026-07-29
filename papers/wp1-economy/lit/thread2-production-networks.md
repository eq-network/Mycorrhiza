# Thread 2 — Leontief IO / production networks (positioning note)

Status: sweep done (6 results, 2026-07-29); canon anchors pending per-title
verification. Sweep artifacts:
`PaperPipeline/lit_reviews/2026-07-29_wp1_economy/t2_networks.{bib,json}`.

## The canon we position against

**Miller & Blair (2009)** — the IO reference: technical coefficients,
Leontief inverse, demand attribution, **hypothetical extraction** — the last
already serving as `io_economy`'s analytic validation anchor (its
counterfactual instrument recovers `1ᵀ(I−A)⁻¹Δd_H` in closed form). The
paper inherits this anchor unchanged. `refs.bib: miller2009io` (VERIFY).

**Carvalho — "Production Networks: A Primer"** (with Tahbaz-Salehi, Annual
Review of Economics 2019; the sweep found it directly, twice) — micro shocks
propagating through IO linkages to macro outcomes. Our sequential
rebalancing (`x ← Ax + d`, one Neumann step per tick, dynamics not
equilibrium) is the Farmer-school reading of the same object. Note: the
canon citation should be updated from Carvalho 2014 (JEP) to Carvalho &
Tahbaz-Salehi 2019 (primer) or both — the sweep's DOI gives the verified
entry for 2019.

**Our delta:** the network literature studies shocks to *given* structures;
our fork grows a new node-type (owned automation capital) whose accumulation
endogenously shifts the automation share and, through upkeep/investment
routed to the machine sector, feeds demand back into the structure. The
knee and eradication band have no analogue because capital in this
literature does not pay to exist.

## What the sweep adds

- **Afrouzi et al. 2023** (sufficient statistics for dynamics in production
  networks) — cite only if we need a modern dynamics-in-networks reference
  beyond the primer; likely appendix.
- **Arata 2024 (Japan), Liao 2025 (China)** — empirical demand-shock
  propagation through IO linkages; useful one-line support that demand-side
  propagation (our closure channel) is empirically real, not a toy artifact.

## Parameter anchoring verdict for Table 2

`A_ij` stays tuned-for-legibility (hub+chain); the empirical IO literature
could in principle anchor coefficient *magnitudes* (column sums of real IO
tables ≈ 0.4–0.6 for intermediate shares), which would upgrade `A_ij` from
tuned to anchored-in-range. Worth one sentence + citation in the final
draft; note it for the Farmer-referee.
