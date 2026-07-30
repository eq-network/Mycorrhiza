# Gradual Disempowerment suite v0.1 — membership and the resource map

*Deposited 2026-07-30. Direction set by Jonas: name the three WP models one suite,
group them in the library (`environments/suites.py`), and map the resources each model
uses so the coupled rewrite (`docs/ledger-design.md`) starts from what actually exists.
The inventory below is code-derived (declared reads/writes and dynamics read directly
from each substrate); the rewrite implications in §4 are design conjecture. Siblings:
`docs/ledger-design.md` (the grammar this map instantiates), `docs/model-register-design.md`
(register conventions), the three vault papers (`…/CI Library/papers/wp{1,2,3}-*`).*

## 1. Membership (code: `environments/suites.py`)

| Role | Environment | Paper | One line |
|---|---|---|---|
| economy | `capital_economy` | WP1 "Where Does the Money Go?" | SFC circular flow on an IO backbone; automation share splits value added; owners reinvest, hoard, or stall (r-closure) |
| culture | `influence_exchange` | WP2 "Who Fills Your Head?" | anchored DeGroot belief + preferential-attachment attention drift; power read as percept supply |
| politics | `delegative_polity` | WP3 (delegation & takeover threshold) | liquid-democracy ballots, power-weighted median sets the tax, entrenchment gates enforcement and re-delegation |
| coupled | `coupled_society` | — (v0 baseline) | κ-gated modulation couplings; demoted by ledger-design; kept as the implementation-invariance baseline |

Honest wrinkle, stated everywhere it matters: **`influence_exchange` serves double
duty** — it is alpha scenario A4's political substrate *and* WP2's cultural substrate,
used unchanged with offline instruments. One env, two readings. The suite's "culture"
role reflects WP2's actual engine choice, and the rewrite must decide whether attention
and delegation stay one shared object or two (see §4).

## 2. The resource map

Read each column as: what the stock is, who holds it, what conserves it, and what
mints or destroys it.

### 2.1 Conserved resources — one per model, already there

| | Money (WP1) | Attention (WP2) | Ballots (WP3) |
|---|---|---|---|
| carrier | `last_reward`, `wealth`, `capital_income`, `demand_h/k` (`capital_economy`) | rows of `listening` (row-stochastic, `influence_exchange`) | base votes through `delegation` rows; power = `influence` = normalized ballots held (`delegative_polity`) |
| held by | households, owners, the public fund | each citizen: 1 unit of listening per row | each citizen: 1 vote (AI delegates: `ai_ballot`) |
| conservation | **explicit SFC discipline**: upkeep settled `min(rev, mK)`; unspent surplus parks in the tracked hoard; money/title separation "exactly once"; **a conservation probe checks every run** | **by construction**: `rewire` reallocates off-diagonal mass and renormalizes; `self_weight` is a fixed floor; AI rows frozen | **by construction**: `tally_power` re-issues base votes each tick and counts who holds them; `tax_and_redistribute` is money-conserving (flat tax, equal payout) |
| sources / sinks | value added mints income each tick (production); consumption/demand recirculates; no leak (that is the probe's claim) | none — pure reallocation | none — pure reallocation; ballots are re-issued, never accumulated |
| who moves it | households' `spend_weights` **(an allocation action — the ledger grammar's §3 primitive already exists here)**; owners' fixed reinvest/consume/hoard split | citizens' drift toward attractiveness `a_j`; churn absent | citizens' drift toward `a_j` **plus churn toward uniform** (the freedom to re-delegate), gated by own `engagement` |

### 2.2 Non-conserved stocks (real stocks, honestly minted or decaying)

| Stock | Model | Dynamics | Note |
|---|---|---|---|
| physical capital `capital`, `pub_cap` | WP1 | formed by investment *spending*, decays by depreciation, charged-to-stock on losses | title, not money — the separation is the point |
| capability `efficiency` | WP1 | grows `e·(1+g+γe)` once AI capital exists, capped | the model's only pure source; drives the automation share |
| wealth (WP3's copy) | WP3 | accumulates net income forever; observed by agents, **never spent** | a score, not circulating money — the unification target in §4 |
| belief `opinion` | WP2 | FJ-anchored pooling, AI reservoir pinned | bounded field, not a resource |

### 2.3 Institutional / rate layer (computed collective variables — ledger-design §4's "residue")

| Variable | Model | Written by | Gates |
|---|---|---|---|
| `enforcement` + `redelegation_friction` (one `regime` scalar) | WP3 | top power share past `entrenchment_threshold`, gain-scaled | effective tax rate; churn; optionally the franchise floor itself (`franchise_erosion`) |
| `amplification` | WP2, WP3 | the clock (onset schedules) | attractiveness — the threat is a schedule, not a decision |
| `alignment_ai` | WP3 | static config | AI delegates' fidelity blend |
| `cap_scale`, `attract_boost` | WP2, WP3 | defense mechanisms / coupling ports | attractiveness |

### 2.4 The shared kernel nobody named

`influence_exchange.rewire` and `delegative_polity.rewire_delegation` are the **same
transform** up to three deliberate differences (churn toward uniform, the franchise
floor `s_w`, engagement-gated rows). Both: attractiveness `a_j ∝ (v_j+ε)^γ ·
amplification · cap_scale · attract_boost · engagement`, row-stochastic drift, frozen
AI rows, fixed self-weight. Attention and delegation are one mathematical object —
a conserved per-citizen share being reallocated by preferential attachment — carrying
two civic interpretations. `engagement` is the common action channel in both, and in
`coupled_society` all three domains already act through one (N, 3) action vector.

## 3. What v0's couplings did to these resources (the diagnosis, now precise)

`coupled_society` couples the *predecessors* of these models by writing the **rate
layer** directly: income share → broadcast effort, converted share → attract boost,
influence share → enforcement. None of the three conserved resources crosses a domain
boundary; nothing is spent. The two channels that do move money
(`regulatory_capture`, `converts_capitalize`) conserve it — they are the template.
The v0.1 suite makes the cheat unnecessary: every domain now natively owns a conserved
resource that the others could transact in.

## 4. Implications for the coupled rewrite (design conjecture)

1. **The three ledgers exist; the rewrite's job is unification, not invention.**
   Money: WP1's circular flow becomes the shared money ledger — WP3's exogenous
   `endowment` becomes WP1 income, and WP3's write-only `wealth` becomes WP1's real
   `wealth` (spendable, hoardable). Attention: WP2's `listening` rows. Ballots:
   WP3's `delegation` rows.
2. **Cross-domain influence = spending one ledger to move another**, per
   ledger-design §3: money buys attention (paid reach entering `a_j` as bought
   listening share, displacing organic weight); attention converts belief (WP2's
   pooling already is this); belief and attention move ballots (delegation drifts
   toward attention winners — the shared kernel makes this one term, not a new
   transform); money lobbies the regime (funded pressure on `enforcement`, replacing
   the free influence-share read). Every channel is an allocation out of a conserved
   stock, so it has an opportunity cost.
3. **The allocation action already exists in WP1** (`spend_weights` over sectors);
   the rewrite extends its columns: sectors + broadcast + lobbying. Owners' fixed
   reinvest/consume split becomes the same vector. "Stupid thing first": fixed
   fractions, per ledger-design §3.
4. **The institutional residue is exactly WP3's regime variable.** `enforcement` /
   `redelegation_friction` stays a computed slow variable — but moved by funded
   lobbying flows (value-agnostic: citizens' spending defends it exactly as capital's
   erodes it), not by a free share readout. `entrenchment_gain=0` remains the honest
   region and the sealing-analog for the political ledger.
5. **The shared kernel is the integration seam.** One attachment transform, two
   instances (attention, delegation) with declared differences — factor it or fork it
   consciously; accidental divergence between the two copies is now a suite-level bug.
6. **Open question carried from ledger-design §9:** whether WP1's economy leg enters
   whole (32-node IO backbone) or reduced. The conserved-flow discipline is the
   non-negotiable part; the sector count is not.

## 5. Status

- Suite named and grouped: `environments/suites.py` (`GRADUAL_DISEMPOWERMENT`,
  version 0.1), behavioral test `tests/test_suites.py`.
- This map: code-derived inventory, deposited 2026-07-30. §4 was design direction
  when written; **built the same day** as `environments/ledger_society` (shared
  kernel in `environments/attachment.py`, ledger checks in `environments/ledger.py`,
  8-rung ladder). §4.5's factoring landed for the new model only — the two older
  envs keep their inlined kernels pending a bit-identity-gated refactor. Suite
  membership unchanged: `ledger_society` is the v0.2 `coupled`-slot candidate,
  gated on the invariance run against `coupled_society`.
- Referee paragraph (what a critic should press on): the "three conserved ledgers"
  symmetry is partly aesthetic — WP3's ballot conservation is a *re-issue* rule (votes
  reset each tick), not an accumulating stock like money, and attention conservation
  is per-row rather than global, so "spending attention" and "spending money" are not
  the same operation and the rewrite should not pretend they are. WP2-as-culture rests
  on reading belief as culture; a critic who rejects that reading collapses the suite
  to two domains plus a duplicate. And the shared-kernel observation cuts both ways:
  if attention and delegation are one object, the culture and politics domains are not
  structurally independent evidence in the register's sense.
