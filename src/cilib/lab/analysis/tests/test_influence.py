"""Behavioral tests for the influence_preserved v0 score."""
import math

import numpy as np

from cilib.lab.analysis.influence import influence_preserved


def test_baseline_vs_itself_is_zero():
    baseline = np.array([0.1, 0.15, 0.05, 0.2])
    assert influence_preserved(baseline, baseline) == 0.0


def test_full_restoration_is_one():
    baseline = np.array([0.1, 0.2])
    treatment = np.array([1.0, 1.0])
    assert abs(influence_preserved(treatment, baseline) - 1.0) < 1e-9


def test_worse_than_baseline_is_negative():
    baseline = np.array([0.4, 0.5])
    treatment = np.array([0.1, 0.2])
    assert influence_preserved(treatment, baseline) < 0.0


def test_nothing_lost_is_nan():
    baseline = np.ones(4)
    assert math.isnan(influence_preserved(np.ones(4), baseline))
