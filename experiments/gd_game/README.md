# gd_game — the GD game's branch trees

> **DEAD 2026-08-01. Do not extend, re-run, or re-score this.** The game this
> generates trees for was judged really bad and essentially worthless by Jonas,
> and the work was not worth it. It is route A of two dead routes: precompute
> forces a small enumerable choice space, so the player gets about three
> choices in a whole run, and widening the tree does not fix that. See
> `docs/gd-game-postmortem.md` before touching anything here.

The game's data source (docs/gd-game-design.md): three starting towns over
`ledger_society` where humans are losing, three intervention windows, five
defense cards priced against the state the run has reached — economic cards
floored on the human wealth share, political cards debiting the enforcement
stock in-model (`mechanisms/interventions.py`). The tree generator enumerates
every affordable branch as a full run from t=0, once per town; the browser
walks the finished tree and computes nothing (CLAUDE.md ⟦BOUNDARY⟧).

One role per file, `_template` style: `config.py` freezes deck, windows, the
three towns (`VARIANTS`), committed expectations and design gates;
`branches.py` is the single "choices -> environment" source shared by run and
export; `run.py` builds each town's affordable tree (8 seeds per path,
bootstrap CIs, affordability from the median across seeds at the tick before
each window) -> `results.json` keyed by tree id (expectation failures exit
nonzero — report and revise, never retune; design gates may license a declared
revision of that town's dials); `export.py` re-runs each path single-seed and
writes one `tree.json` + `runs/p-*.json` directory per town, validated on
write against `schema/tree.schema.json` (eq-network CI re-validates it too).

    pip install -e .[export]
    python -m experiments.gd_game.run [--smoke] [--variant ID]
    python -m experiments.gd_game.export [--smoke] [--variant ID] [--out DIR]

`--variant` merges over the other towns' trees in `results.json` rather than
clobbering them; with no flag all three run.

The towns share one deck and one price list and differ only in model dials
(`config.VARIANTS`, all tuned-for-legibility, probed 2026-07-31). Wait-path
arcs at 8 seeds, enforcement across the three windows and the political cards
affordable there:

| town | dials vs default | enforcement | political tray |
|---|---|---|---|
| harbor | `regime_rate` 0.02 | 0.727 / 0.516 / 0.402 | 3 / 1 / 0 |
| boomtown | + `init_wealth` 2.0, `repair_rate` 0.01 | 0.663 / 0.095 / 0.000 | 3 / 0 / 0 |
| commune | + `init_wealth` 0.6, `repair_rate` 0.025, `churn` 0.08 | 0.763 / 0.638 / 0.617 | 3 / 2 / 2 |

Committed expectations bind harbor only; every town is held to the
`tray_shrinks` design gate. Boomtown clears that gate but leaves one live
decision — its window-2 and window-3 trays are empty. Regenerate whenever
`ledger_society` or the deck changes; never hand-edit an artifact (tree.json
checksums make edits detectable on both sides).
