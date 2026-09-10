"""Frozen spec for the /library/prototype explainer fixtures (see README.md).

One substrate threads the whole explainer page: governed_commons — the same
model appears as a transform, a compiled pipeline, a system graph, and a
trajectory. Phase 0 scope: pipeline-subsets, system-graphs, condition runs,
manifest. Snippets / schedule-golden / scorecard land with the page's later
phases.
"""
from cilib.core.schedule import ScheduleSpec
from cilib.mechanisms import QuotaVoteConfig, SanctionConfig

ENV = "governed_commons"
T = 200
SEED0 = 0
ROUND_DECIMALS = 4
MAX_FILE_BYTES = 100_000     # per emitted file
MAX_TOTAL_BYTES = 400_000    # the whole fixture set

# Same (mechanism, config, schedule) triple convention as examples/05 and
# experiments/benchmark — kept in sync by hand for governed_commons.
CONDITIONS = {
    "baseline": [],
    "quota_voting": [("quota_vote", QuotaVoteConfig(), ScheduleSpec(cadence=5))],
    "graduated_sanctions": [
        ("quota_vote", QuotaVoteConfig(), ScheduleSpec(cadence=5)),
        ("graduated_sanction", SanctionConfig(), None),
    ],
}

# Runs ship (T,) globals only. The schema additionally forbids a nonempty
# "node" section, so the size budget holds structurally, not by care.
RUN_WHITELIST = ("resource_level", "policy_target")

# The BatchBoard lattice: the graduated_sanctions pipeline in program order
# minus the trailing step_counter (bookkeeping). 4 transforms -> 2^4 rows.
SUBSET_MECHANISMS = CONDITIONS["graduated_sanctions"]

# scheduled() golden windows: (cadence, phase_offset, onset) probed through the
# engine's own wrapper over SCHEDULE_TICKS ticks — pins the page widget.
SCHEDULE_TICKS = 60
SCHEDULE_COMBOS = (
    (1, 0, 0), (5, 0, 0), (5, 2, 0), (5, 0, 12), (7, 3, 20), (3, 1, 6),
)

# Scorecard: the benchmark's scenario-1 instrument at the benchmark's scale.
# The caveat class travels INSIDE the fixture — the page renders it as a badge.
SCORECARD_SEEDS = 32
SCORECARD_DELTA = -0.5
SCORECARD_DESCRIPTIVE = ("stock_pct", "compliance_rate", "influence_fidelity")
SCORECARD_CAVEAT = ("exploratory — instrument calibration on one substrate, "
                    "not a finding about institutions")

# Influence-from-birth vs influence-now: responsiveness of a post-shift harvest
# window to a one-shot collective ask-shift scheduled at t0, per condition.
CURVE_T0S = (0, 30, 60, 90, 120, 150)
CURVE_SEEDS = 16
CURVE_DELTA = -0.5
CURVE_WINDOW = 10
CURVE_CONDITIONS = ("baseline", "graduated_sanctions")
