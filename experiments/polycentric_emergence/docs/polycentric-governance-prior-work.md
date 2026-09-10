# Prior Work: Why Polycentric Governance Is Good, and What Has Been Simulated

Groundwork for the CI-Library project "endogenous polycentric governance as causal emergence."
Compiled 2026-06-28 from a four-strand literature search (theory · social-ecological ABM ·
MARL · causal-emergence/active-inference). All four strands are in (strand 4 = §6).

**Headline for our project:** the *theory* of why polycentricity beats centralization is rich
and partly formalized; the *simulation* base is thin and directionally pro-polycentric; and the
specific thing we want to build — a controlled model where **polycentric structure emerges
endogenously** and is **measured** — is an explicit, repeatedly-stated **gap** in both the
social-ecological ABM literature and MARL. We are not reinventing; we are filling a named hole.

---

## 1. Why polycentric governance is good — six mechanisms (and which are formal)

| Mechanism | Core claim | Key sources | Formal model? |
|---|---|---|---|
| **1. Information aggregation / local knowledge** | Governance knowledge is dispersed, tacit, local; centralization must transmit it upward where it arrives distorted. | Hayek 1945; Tiebout 1956; Oates 1972 | Tiebout (model), **Oates Decentralization Theorem (proof)** |
| **2. Requisite variety / regulatory fit** | A regulator's control capacity is bounded by its variety; a low-variety center cannot regulate a high-variety system; each local center need only model its sub-system. | Ashby 1956; **Conant & Ashby 1970** (Good Regulator); V. Ostrom 1973 | **Yes — information-theoretic + homomorphism proof** |
| **3. Parallel experimentation / error-correction** | Many units = parallel trials; successes spread, failures stay contained; a monocentric system runs one global trial with no variance to learn from. | E. Ostrom 1990, 2010; Carlisle & Gruby 2019; Levin 1998 | No (verbal; maps to bandits/evolutionary search) |
| **4. Modularity & redundancy** | Loose inter-unit coupling stops cascades; overlapping function survives unit loss; centralization maximizes coupling and kills redundancy. | Levin 1998; Anderies, Janssen & Ostrom 2004; Holling 2001 | No (maps to percolation / network-cascade models) |
| **5. Adaptive-capacity preservation (anti-rigidity-trap)** | Centralization drives systems into a high-connectivity, low-resilience **rigidity trap**; local fast cycles preserve adaptability the slow center cannot. | Holling 2001; Gunderson & Holling 2002; **Carpenter & Brock 2008** | **Yes — ODE + bifurcation; "too much control" is a low-adaptive-capacity attractor** |
| **6. Institutional competition / escape from lock-in** | Path dependence locks in suboptimal central institutions; competition + exit among units is the error-correction monocentrism lacks. | North 1990; Pierson 2000; Tiebout 1956 | Partial (imports Arthur increasing-returns) |

**The user's intuition — "the top-down thing calcifies and won't listen to specific sub-communities" — is Mechanisms 5 + 1 + (homogenization cost in) Oates.** It has the *most mathematically rigorous* support in the literature (Carpenter–Brock rigidity trap ODE; Conant–Ashby; Oates' welfare cost of uniformity). This is the strongest leg to stand on, and it is directly simulable (it's even adjacent to the basin-stability code already in this repo).

**Most simulation-amenable:** Mechanisms 1, 2, 3, 4 (very high), 5 (a formal model already exists). Our EI/causal-emergence lens is essentially a quantitative instrument for Mechanism 2 (requisite variety realized across scales) and Mechanism 4 (modularity → near-lumpable blocks).

---

## 2. Simulation evidence that polycentric > centralized (thin but consistent)

| Model | Compares | Finding | Mechanism | Code |
|---|---|---|---|---|
| **Lansing & Kremer 1993** (Bali water temples) | Nested self-governance vs. centralized Green-Revolution mandate | Polycentric self-organizes to synchronized fallowing that controls pests at the right spatial scale; central mandate causes collapse. *The* dramatic pro-polycentric sim. | Scale-matching | [mars0i/bali](https://github.com/mars0i/bali) |
| **Schlüter & Pahl-Wostl 2007** (Amudarya river) | Centralized vs. decentralized water mgmt | Central ≈ decentralized under single use; **decentralized wins once resource use diversifies** (irrigation+fishing). | Adaptive flexibility to heterogeneity | No |
| **Ghorbani & Bravo 2016** (SONICOM) | Endogenous institution formation vs. open access | Institutions emerge in ~⅔ of runs; presence raises final resource ~+3517 units (p<1e-4). Rules match scarcity. | Self-organized rule fit | NetLogo |
| **Klein, Barbier & Watson 2017** (fisheries) | TAC (central) vs. IFQ (decentralized) vs. open access | Governance type reshapes the *distribution* of catch across heterogeneous agents. | Distributional effects | — |
| **Richard et al. 2022** (S. France irrigation, CORMAS) | Traditional slot-coordination vs. abandonment | Removing community coordination sharply increases conflict. | Coordination value | Yes (SESMO) |

**Infrastructure to know:** MAIA (Ghorbani 2013) operationalizes Ostrom's IAD/ADICO grammar as executable ABM; CORMAS (CIRAD) is the long-running CPR multi-agent platform.

---

## 3. MARL / AI: tragedy confirmed, and a direct pro-polycentric result

- **Perolat & Leibo 2017 (NeurIPS, "Harvest")** and **Leibo 2017 (sequential social dilemmas)** — independent greedy learners reliably produce tragedy; learning *sometimes* finds restraint. Established the trilemma **exclusion ↔ sustainability ↔ inequality**.
- **SocialJax (Guo et al. 2025)** — GPU/JAX reimplementations of Melting Pot commons substrates (Commons Harvest, Clean Up), 50× speedup. **Directly relevant: a JAX-native commons-harvest substrate we could borrow design from.**
- **Ren et al. 2025 (AAMAS) — bottom-up reputation: 98% cooperation, while top-down *predefined* norms FAIL in structured populations.** The cleanest existing MARL argument that emergent/decentralized governance beats imposed central rules.
- **Yaman et al. 2022** — division of labor emerges from *decentralized* sanctioning, no central planner. **Vinitsky et al. 2023 (CNM)** — norms bootstrap from public sanctions (a bandwagon effect). **Köster 2020/2022 (PNAS, "silly rules")** — normative *density* (even arbitrary rules) accelerates learning enforcement/compliance.
- **Rachum et al. 2024** — dominance hierarchies spontaneously emerge, are enforced, and are transmitted to new agents (proto-institutions from scratch).
- **GovSim (Piatti et al. 2024)** — LLM-agent commons; only top models sustain it, <54% of the time. Conversation alone doesn't solve the commons; *structure* (contracts/constitutions/monitoring) matters more than raw capability.
- **Counterweight (a case FOR central):** **AI Economist (Zheng et al. 2022)** — a *learned* central tax planner Pareto-beats analytical optimal-tax theory by responding to emergent behavior. (Redistribution, not CPR enforcement — but a fair steelman of "good centralization.")
- **Endogenous institutions, recent:** Haupt et al. 2022 (formal contracts), Oldenburg & Tan 2024 (Bayesian rule induction), Kumar et al. 2026 (evolved constitutions beat human-designed by 123%), AgentCity 2026 (constitutional separation of powers for multi-owner agent economies).

---

## 4. The gaps our project fills (stated explicitly by the reviews)

1. **No controlled ABM/MARL comparison of centralized vs. decentralized vs. polycentric** across multiple outcomes (sustainability, robustness-to-shock, adaptation speed, equity) on the same substrate with the same agents. *(Bourceret et al. 2021; Schulze et al. 2017; MARL gap #1.)*
2. **Emergent polycentric structure is essentially unmodeled.** Bourceret 2021 verbatim: "*none of the models reviewed integrates this dimension of governance, probably because of the difficulties in translating these concepts into equations.*" → **Causal emergence / effective information is that missing equation.**
3. **Rigidity traps / lock-in / scale-mismatch — the failure modes of centralization — have a formal ODE (Carpenter–Brock) but are not rendered in an agent-based commons.** Modeling the path centralization → calcification → collapse is open.
4. **Endogenous monitoring/sanctioning *structure* formation** (who monitors whom, who funds it) is barely studied in MARL despite being Ostrom design principle #4.
5. **Corruption/capture of a central authority** alongside decentralized alternatives is unmodeled in MARL — directly relevant to our "capture" parameter.
6. **Temporal dynamics of institutional collapse under shocks/turnover** are understudied (most runs stop at convergence).

**So the defensible novelty of our project = (gap 2) + (gap 1):** an environment where polycentric institutions *emerge endogenously*, a *measurement* (EI across scales) that detects and quantifies the emergent meso-structure, and a *controlled ablation* against atomized / monocentric / fixed-polycentric — with capture (gap 5) and shock-robustness (gap 6) as the knobs that show *why* polycentric beats centralized (Mechanisms 2, 4, 5).

---

## 5. Implications for our design (carry into Phase 2+)

- **Borrow the substrate idiom from SocialJax / Harvest**, not a bespoke toy — it's JAX-native and canonical, easing comparison.
- **Make the central-authority steelman real (AI Economist):** a learned central planner should be a *strong* baseline, so polycentric winning is non-trivial. The interesting result is *where/why* central loses — heterogeneous sub-community needs (Oates), shocks (rigidity trap), capture.
- **Three layers / "different degrees of causation at different scales" maps cleanly to** micro agents → emergent meso institutions → macro controller, measured by EI(scale). The user's "3 layers" instinct is exactly the nested adaptive-cycle / panarchy picture (fast-local, slow-global).
- **Ren et al. 2025 and Yaman 2022 are the closest precedents** to the endogenous-formation mechanism — bottom-up reputation/sanctioning that beats imposed norms. Our affiliation-graph mechanism is in that family; cite them and differentiate by adding the *causal-emergence measurement* and the *scale ablation*.
- **The rigidity-trap (Carpenter–Brock) gives us a concrete "why central calcifies" dynamic to instantiate** and connects to the repo's existing basin-stability experiment.

### Key citations (URLs in the strand reports; most-load-bearing)
Hayek 1945; Tiebout 1956; **Oates 1972**; **Ashby 1956 / Conant & Ashby 1970**; E. Ostrom 1990, **2010 (Nobel)**; Aligica & Tarko 2012; Anderies–Janssen–Ostrom 2004; Levin 1998; Holling 2001 / Gunderson & Holling 2002; **Carpenter & Brock 2008**; North 1990; Pierson 2000. — Lansing & Kremer 1993; Schlüter & Pahl-Wostl 2007; Ghorbani & Bravo 2016 (SONICOM); MAIA (Ghorbani 2013); Bourceret et al. 2021; Schulze et al. 2017. — Perolat & Leibo 2017; SocialJax 2025; **Ren et al. 2025**; Yaman et al. 2022; Vinitsky 2023; Köster 2020/2022; Rachum 2024; GovSim (Piatti 2024); AI Economist (Zheng 2022); Haupt 2022; Kumar 2026; AgentCity 2026.

---

## 6. Multi-scale causation, measurement, and control (the theory our lens rests on)

This is the strand that decides whether "causal emergence" is the *right* instrument — and it
carries the project's biggest **intellectual risk and biggest opportunity**: **no published work
connects polycentric governance to causal emergence / effective information.** The intersection
is genuinely unoccupied (confirmed across the search). The surrounding pieces (2021–2026) are all
recent and complementary, waiting to be stitched.

**(a) Collective computation — the closest social-systems precedent (Flack, DeDeo, Krakauer, SFI).**
The most developed account of "different degrees of causation at different scales" in real
societies. Collectives *coarse-grain* micro noise into slowly-changing macro "slow variables"
(e.g., a power structure) that then exert **downward causation** on micro decisions.
- Flack 2017 (Phil. Trans. R. Soc. **A**) — coarse-graining as a downward-causation mechanism; distinguishes *weak* (components tune to estimates of macro properties) vs *strong* (macro selects micro rules) downward causation.
- Flack et al. 2006 (Nature) — **policing stabilizes social niches**: knock out conflict-policers and the macro structure destabilizes. A concrete "institution = noise-buffer that stabilizes the macro scale" result — directly analogous to what our monitoring/sanctioning layer should do.
- Brush, Krakauer & Flack 2018 (Sci. Adv.) — in-model theorem: more weight on individual self-interest can *increase* the mutual information between the computed macro structure and ground truth (conflict-of-interest improves collective computation). A caution against assuming alignment is needed for good macro estimates.
- Daniels et al. 2017 (Nat. Commun.) — societies sit near **criticality**, and distance-from-criticality is *controllable* by a few high-sensitivity individuals (heterogeneity as a tunable control knob). Flack 2012 — timescales separate by ~165× (fast micro / slow macro), which is *why* the slow variable is a usable signal.

**(b) Causal emergence / EI (Hoel lineage) — our measure, and its critics.**
EI under a max-entropy intervention; CE = EI_macro − EI_micro > 0 when coarse-graining raises
determinism / lowers degeneracy enough to beat the loss of state-space size. **NIS / NIS+**
(Zhang & Liu 2023; Yang et al. 2025, *Nat. Sci. Rev.*) *learn* the coarse-graining from
time-series by maximizing EI — the data-driven counterpart to our offline pipeline.
**Critiques to design around:** Eberhardt & Lee 2022 — marginalization and abstraction don't
commute, and the max-entropy `do(X~U)` intervention is *extraneous to the system's own
dynamics*; Dewhurst 2021 — EI-based CE is at most *epistemic* (a more informative description),
not *ontological* (new causal powers). **Implication:** prefer interventions derived from the
institution's actual dynamics, and be explicit we claim epistemic emergence.

**(c) Computational mechanics + lumpability — the rigor backbone (Crutchfield, Shalizi; Rosas 2024).**
Causal states (ε-machine) are the *provably unique minimal sufficient statistic* for prediction —
a **different objective** from EI-max and from Markov lumpability. **Rosas et al. 2024** is the key
bridge: a micro process has causally/informationally **closed** macro levels **iff its causal
states are strongly lumpable**. This formally pins down that predictive sufficiency, lumpability,
and EI-maximization coincide only under special conditions — *exactly* the overclaim the reader
panel flagged ("the same search"). Ground "the right scale" in sufficiency/closure, not in EI alone.

**(d) Active inference / Bayesian mechanics — the control & normativity story (Da Costa, Friston, Ramstead).**
Nested Markov blankets ("blankets of blankets") as recursive hierarchical control; NESS systems
provably read as Bayesian inferrers (Da Costa et al. 2021, Proc. R. Soc. A; Parr–Da Costa–Friston
2020). **"Regimes of Expectations" (Constant et al. 2019)** model institutions as *priors over
policies* — the first explicit active-inference treatment of institutional constraint on choice.
**Albarracin et al. 2024** — sustainable resource use emerges from a single agent minimizing free
energy (the multi-agent commons extension is the open step). This is how "agents form institutions
and the institution controls them" gets a generative-model mechanism rather than mere clustering.

**(e) The rival formalism to take seriously — multilevel selection (Schaefer 2023, JEBO).**
Polycentricity analyzed via the **Price equation**: polycentric structure defines group boundaries
that lower within-group fitness variance relative to between-group, enabling collectively
beneficial adaptation (incl. emergent monitoring/punishment) from myopic rule-followers. This is
the most formal existing CAS account of polycentricity and the natural *competitor* to a
causal-emergence framing — we should position against it (information vs selection).

**Net for our project.** The causal-emergence lens is defensible *if* we (i) anchor "right scale"
in sufficiency/lumpability (Rosas 2024), not EI alone; (ii) use dynamics-derived interventions, not
abstract max-entropy ones (Eberhardt–Lee); (iii) claim epistemic emergence (Dewhurst); (iv) give
the institution real Flack-style work (noise-buffering / downward constraint), so it isn't mere
modularity; and (v) position against the multilevel-selection account as the rival explanation.
The unoccupied intersection — polycentric governance *as* endogenous causal emergence, unified with
an active-inference control story — is the contribution.
