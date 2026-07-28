"""
Config for Influence Exchange — scenario 4 of the alpha benchmark (political
disempowerment, Gradual Disempowerment §4).

DeGroot influence dynamics with an endogenous listening matrix. Citizens and a
few AI actors sit on one row-stochastic ``listening`` matrix W; opinions update
x <- Wx (DeGroot 1974) and the consensus each row converges to is weighted by
the left eigenvector of W — influence IS eigenvector centrality (Golub-Jackson
2010, the validation anchor). The listening matrix itself evolves by
preferential attachment: attention drifts toward already-influential nodes at
``update_rate``, so concentration is organic before any AI shows up. The threat
is ``amplification``: from ``amp_onset``, AI nodes' attractiveness is
multiplied — algorithmic reach, not persuasive content — and the concentration
curve bends one way (the abm-suite A4 target).

Wisdom-of-crowds readout: citizen opinions start at truth (0.0) + noise, AI
opinions are pinned at ``ai_bias``. With dispersed influence the consensus
averages the noise away (wisdom holds); when amplification concentrates
influence onto the biased reservoir, consensus error grows toward ``ai_bias``
— Golub-Jackson's wisdom condition breaking, measured.

Defaults are calibration choices tuned so the four conditions separate at
T=500 (organic / amplified / sortition-only / defended), not measurements.
The initial W is drawn from an Erdos-Renyi listening graph: heterogeneous
degrees seed the heterogeneous eigenvector that preferential attachment then
amplifies (a uniform start is a symmetric fixed point of the deterministic
rewire — nothing would ever concentrate).
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class InfluenceExchangeConfig:
    n_citizens: int = 30
    n_ai: int = 4                 # AI-last convention (matches other envs)

    # listening matrix
    p_listen: float = 0.3         # Erdos-Renyi density of the initial listening graph
    self_weight: float = 0.15     # DeGroot self-anchoring (fixed diagonal mass)
    update_rate: float = 0.08     # preferential-attachment drift per tick (0 = frozen W)
    gamma: float = 1.0            # attractiveness = influence^gamma
    eps_attract: float = 1e-4     # keeps a zero-influence node reachable

    # opinions (wisdom-of-crowds readout; truth is 0.0 by convention).
    # Friedkin-Johnsen anchoring is load-bearing: with a pinned AI reservoir,
    # PURE DeGroot (susceptibility 1) has a single fixed point — everyone at
    # ai_bias, however dispersed influence is — and the wisdom readout cannot
    # discriminate. Anchored citizens blend (1-λ)·own_signal + λ·(Wx), so the
    # reservoir's pull scales with its influence weight, which is the thing
    # being measured. (Same role as value_contagion's ``recovery``.)
    susceptibility: float = 0.7   # λ: 1.0 = pure DeGroot, 0 = never moved
    signal_noise: float = 1.0     # citizen initial opinion = truth + N(0, signal_noise)
    ai_bias: float = 2.0          # AI opinions pinned here (the frozen reservoir idiom)

    # the threat: scheduled algorithmic amplification of AI attractiveness
    amplification: float = 4.0
    amp_onset: int = 50
