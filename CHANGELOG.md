# Changelog

All notable changes to Omega-S will be documented in this file.

## [0.3.0] : 2026-07-31

### The contrast-preserving reformulation was tested. It is worse.

The v0.2.0 release, and the appendix of the preprint, identified the
saturating logistic in the construction of `A` as the reason the clustering
channel is inert, and offered a contrast-preserving reformulation as an open
direction for the community. We ran it ourselves. The result is negative and
is now reported in the paper (Section 4, *Testing the contrast-preserving
reformulation*) rather than left open.

- **New arm `cos_composite`.** A replica of `StochasticOmegaS` changing one
  thing: `A_ij = |cos(w_i, w_j)|` with zero diagonal, in place of
  `sigmoid(|W W^T|)`. Same four factors, same formulas, same modules, same ten
  evaluation seeds.
- **The construction does what it was designed to do.** The excess statistic
  `C/D` moves from 1.0000 (dead, four decimal places, every module) to a median
  of 1.36; the library's own clustering term moves off its ceiling from 0.9999
  to 0.80 on `q_proj`; the sign of the penalty controls the direction of
  movement in 16 of 16 measurements, with a monotone dose response.
- **And it loses on all ten seeds.** Mean retention 0.537 against 0.766;
  Wilcoxon signed-rank `W=0`, two-sided `p=0.002`. The maximum of the cosine arm
  falls below the minimum of the original arm, so the verdict does not depend
  on the pairing. Against weight decay and EWC it is a tie with the point
  estimate below (4/10 each).
- **The ordering is monotone in the wrong direction:** clustering-only under
  cosine 0.477, full composite under cosine 0.537, original construction 0.766.
  The saturated channel was not a defect holding the method back.
- **Target recalibration.** The ratio calibration normalises gradient magnitude
  but not direction, so the optimum does not transfer: 0.003 for the cosine
  form against 0.03 for the original. The new optimum is interior and bracketed
  on both sides, which the original sweep never achieved.
- Results: `results/cosine_construction_20260731.json`.

### Two findings about the existing implementation

- **`StochasticOmegaS` branches on matrix shape.** It forms
  `A = sigmoid(|W W^T|)` only when `W` is non-square; for square `W` it applies
  `sigmoid(|W|)` elementwise, without forming the Gram matrix. In Llama-3-8B
  `q_proj` is square and `v_proj` is not, so the published runs used one module
  of each kind, while the paper described only the Gram form. Measured impact:
  the degree-variance term differs by a factor of roughly 400 between branches
  on `q_proj`. **The reported numbers are unaffected**, having been produced by
  the code as written; the description of the method is what was incomplete,
  and both the paper and this repository now state it.
- **What the degree sequence encodes.** Regressing degrees on row norms over
  the base weights: the square branch is 96% magnitude (layers 8 to 31 of
  `q_proj`), the Gram branch is 71% alignment (`v_proj`). The regulariser
  applies two different criteria depending on module shape. Results:
  `results/mechanism_decomposition_20260731.json`.

### Sign of the clustering term: no effect

Six seeds held out of the evaluation set, both signs at matched strength: 3 and
3, median paired difference -0.045, Wilcoxon `W=8` at `n=6`. Whether the
penalty raises or lowers clustering does not determine retention in this
formulation. Recorded because the question was open and is now closed.

### Row-norm control, off-optimum confirmation

The control's own sweep (v0.2.0) selected target 0.001 and rejected 0.003. We
evaluated 0.003 over the ten seeds and obtained 0.437, consistent with that
rejection. **This does not supersede the control**, which remains the 0.001 run
at retention 0.661 reported in v0.2.0 and in the paper; it is an independent
confirmation that the sweep chose correctly. Results:
`results/rownorm_t0.003_20260731.json`.

### Merging line: closed

Screened whether the index carries information about interference between two
LoRA adapters. Three findings, all negative for the line and one useful on its
own: the two reasonable symmetrisations of the interference matrix rank modules
at Spearman 0.42, so the triadic excess is not a well-defined quantity there;
the interference itself is rank one, with the leading principal angle 3 to 5
times above chance and every subsequent one indistinguishable from it, so a
single SVD captures it exactly; and the excess does not correlate with that
angle in either module type. What survives is a by-layer map of where two
tasks interfere, which needs no index at all. Scripts in
`experiments/cosine/merging_*.py`.

### Method section rewritten: the paper defined the wrong objective

The Method section defined `A = W W^T` and `Omega = Tr(A^3)/numel(W)`, which is
the **raw** form, the weakest arm in the ten-seed sweep. The log-ratio
composite that produces the headline result was named throughout but its
formula appeared nowhere, nor did the logistic in the construction of `A`. A
reader implementing the Method as written would have reproduced 0.539, not
0.766. The section now defines both forms, gives the algebraic reason the raw
one fails (homogeneous of degree six in `W`, so its gradient decays as the
fifth power of weight magnitude and switches off as weights shrink), and
states what the composite does when minimised. It also names the cost of the
fix: the bounded map that removes the scale sensitivity is the same map that
saturates the clustering term.

- **`M` is the algebraic connectivity, not modularity.** The mean-centring in
  the power iteration deflates the trivial eigenvector, so what is estimated is
  the Fiedler value. The symbol is kept for continuity and the quantity is now
  named.
- **Which form produced which result is now explicit.** Retention uses the
  composite; the MLP pruning and FLOPs results use the raw penalty. These are
  different objectives, and the raw one is the arm that loses on retention. The
  algebra also predicts a small marginal contribution in the pruning setting,
  since group lasso shrinks the weights that the raw gradient depends on, which
  is consistent with the connected GPT-2 re-run finding none.
- **Construction screening table added** to the paper: the six candidate
  constructions with their excess `C/D` and the reason each was rejected.

### New: FLOPs by penalty form (`experiments/exp6b_flops_by_form.py`)

`exp6` measures 54.46% FLOPs reduction for "Omega-S + group lasso" using the
**raw** penalty, and has no group-lasso-alone arm, so its number is a property
of the combined pipeline rather than of Omega. This script adds the missing
control and the missing forms: `gl` alone, `gl+raw`, `gl+lib` (log-ratio
composite) and `gl+cos` (cosine construction), plus the weight-decay sweep.

Two design points make the comparison meaningful. Lambda is **calibrated per
arm** so that the penalty gradient is a fixed fraction of the task gradient;
comparing forms at a fixed lambda is meaningless, since the raw form reaches
1e14 and the composite is a logarithm of order 10. And a **gate** screens,
on the trained MLP weights, whether the cosine construction decompresses the
clustering channel at all; if it does not, that arm is skipped rather than run
blind.

The script exists to test a falsifiable prediction. The raw penalty is
homogeneous of degree six in `W`, so its gradient scales as `c^5`, verified
numerically: at a quarter of the weight scale it retains 0.1% of its force.
Group lasso shrinks weights by design. The raw form should therefore switch
itself off in exactly this setting while the composite should not, which would
predict a larger marginal contribution for `gl+lib` than for `gl+raw`.

### Pruning re-measured over ten seeds: the number is larger, the attribution narrower

`exp6` reported 54.46% FLOPs reduction from one seed, a fixed coefficient and no
group-lasso-only arm. Re-run with the protocol used elsewhere in this paper
(strengths swept on two held-out seeds, then evaluated on ten, with a validity
criterion requiring pruning to cost under one point of accuracy):

| arm | reliable | FLOPs | acc pruned |
|---|---|---|---|
| weight decay | **10/10** | 24.12% | 97.08% |
| **GL + raw** | **9/10** | **94.56%** | 95.96% |
| GL + cosine | 6/10 | 94.77% | 96.10% |
| GL + composite | 6/10 | 94.69% | 96.02% |
| group lasso alone | 6/10 | 94.60% | 96.18% |
| GL + row-norm | 3/10 | 94.33% | 95.90% |
| none | 10/10 | 0.00% | 99.17% |

- **The achievable reduction is 94.6%, not 54.46%.** The published coefficient
  was far from optimal. Weight decay reproduces its published value (24.12%
  against 25.27%), so the mis-calibrated side was ours. The ratio improves from
  2.16x to **3.92x**.
- **The compression advantage over weight decay belongs to group lasso.** Group
  lasso alone reaches 94.60%; the penalty contributes −0.04 pp, that is nothing.
  The arm that would have shown this was missing from the original experiment.
- **What the penalty contributes is reliability**, 6/10 → 9/10 seeds in which
  the network is prunable at all. With ten seeds this cannot reach significance
  (McNemar, best possible discordant pattern, p=0.25), and we say so.
- **The two forms rank oppositely across the two settings**: the composite wins
  retention and adds nothing to pruning; the raw form is the weakest on
  retention and is the only one that helps here. This is why they are now
  treated as two methods.
- Accuracies are not exactly matched: weight decay ends at 97.08% against
  95.96%, so it has both higher accuracy and less compression. The honest
  framing is a trade, not dominance.
- **The reliability gap survives post-pruning recovery.** A criterion forbidding
  any accuracy loss at the moment of pruning is stricter than practice, so we
  repeated the comparison with two epochs of task-only fine-tuning on every
  pruned model. Group lasso alone recovers from 6/10 to 8/10; the raw form goes
  from 9/10 to **10/10**. The gap narrows from +3 to +2 seeds and holds, and two
  group-lasso seeds remain broken after recovery, so those were genuine failures.
  Expected training runs per prunable model fall from 1.25 to 1.00, a 20% saving
  against the 33% implied by the strict criterion. Significance is weaker here,
  not stronger: McNemar's best case gives p=0.50.
- Results: `results/flops_by_form_20260731.json`. Scripts:
  `experiments/exp6b_flops_by_form.py`, `experiments/flops_por_forma.ipynb`
  (self-contained Colab notebook) and `experiments/cosine/anadir_wd.py`.

### Mechanism: which channel carries the retention effect

Ablation by module type, penalty restricted to one while the adapter stays on
both, so what varies is the channel and not where the model can learn. Twenty
cells over the ten evaluation seeds.

| | mean retention | wins |
|---|---|---|
| alignment only (`v_proj`) | **0.648** | **9/10** |
| magnitude only (`q_proj`) | 0.533 | 1/10 |
| full objective | 0.766 | reference |

- **The alignment channel contributes more**, exact sign test one-sided p=0.011.
  Wilcoxon gives W=9, p=0.065, which does not clear the threshold because the
  single loss is also the largest difference.
- **Neither channel reproduces the full objective.** The alignment arm exceeds
  it on 3/10 seeds, the magnitude arm on 2/10, and the better of the two barely
  clears no regularisation (0.648 against 0.631). The effect requires both
  module types.
- **An observation on the bimodal pattern.** The one seed where magnitude wins
  (456) is where it wins largest and where it *exceeds* the full objective
  (1.148 against 1.023); on seed 5055 the two tie and both exceed it. If this
  held with more seeds it would suggest per-seed regimes, one magnitude-driven
  and one alignment-driven, which would be a candidate explanation for the
  bimodal per-seed advantage our earlier structural measurement failed to find.
  On one seed of ten it is an observation, not a finding.
- Caveat: the calibration fixes the total penalty gradient, so restricting to
  half the modules concentrates that force. This does not affect the comparison
  between the two ablated arms, which carry the bias equally, but it does affect
  any comparison against the full objective.
- Results: `results/module_ablation_20260731.json`.

### How much of the objective is live: all four factors measured

The previous entries report that the clustering term is inert. We measured the
other three. Elasticity with respect to the weights, that is the relative change
in a factor produced by a relative change in `W`, median over ten attention
projections of Llama-3-8B spanning layers 0 to 31, each with the branch the
implementation applies to it:

| | `C` | `D` | `M` | `Coex` |
|---|---|---|---|---|
| median elasticity | 0.0000 | 0.0000 | 0.0001 | **0.0091** |

**Three of the four factors are numerically inert and one carries everything.**
Degree variance responds about ninety times more strongly than the modularity
term. As implemented and in these conditions, the composite objective reduces
in practice to a penalty on the variance of node degrees.

This settles a discrepancy that would otherwise be found by comparing the code
against the companion theoretical paper: the library estimates `M` as λ₂ and
drives it *down*, while the index as defined there requires λ₂ *up*. At an
elasticity of 0.0001 the term does not move, so the orientation has had no
effect in either direction. What needs correcting is the description of the
objective, not its behaviour. A construction under which `M` is live would make
the orientation matter, and testing whether correcting it improves retention is
an experiment nobody has run.

Script: `experiments/cosine/check_M.py`. No GPU, minutes on a laptop.

### Corrected orientation of the modularity factor, and every comparison re-run

The index is defined with a modularity factor in the denominator; the reference
implementation estimated algebraic connectivity, the inverse quantity. Found by
checking the applied manuscript against the theoretical one. Corrected, and all
five comparisons re-measured in a single session on the same hardware:

| comparison | wins | Wilcoxon |
|---|---|---|
| vs no regulariser | 9/10 | p=0.006 |
| vs weight decay | 10/10 | p=0.002 |
| vs EWC | 8/10 | p=0.014 |
| vs row-norm control | 10/10 | p=0.002 |
| vs cosine composite | 10/10 | p=0.002 |

Retention 62.9% -> 84.1%; absolute capability 0.173 -> 0.238 (+37.7%).

**Why it helps is not why one would assume.** The modularity term stays inert:
elasticity 0.0001 before training, 0.0001 after, +0.06% change during a run. The
correction changes the *scale* of the objective (-1.06 to -10.73), which through
the gradient-ratio calibration changes the operating point of the one live
factor. Control: the same inversion on the cosine construction, where the scale
does not jump, changes nothing (4/10, p=0.83).

**And the bound**: the difference between orientations (+0.075) is below the
run-to-run standard deviation (0.104), so the two are not distinguishable from
each other at ten seeds. Each is distinguishable from the baselines.

### Run-to-run variation quantified

Repeating an identical configuration gives a standard deviation of 0.104 in
retention ratio. The seed does not identify the run in this setting, which
bounds every seed-paired comparison including ours. Cited alongside the vision
and RL literature where GPU non-determinism is already documented as the
dominant source of variance.

### Adapter placement, measured before training

Gradient-subspace overlap per module identifies where the two tasks share least.
Placing the adapter there, against the conventional placement with the same
number of modules, keeps 36.8% more capability (9/10, Wilcoxon p=0.004), with
plain LoRA in both arms and no Omega-S. The head of the ranking (o_proj first,
q_proj second) transfers across Llama-3-8B, Mistral-7B and Qwen2.5-7B and three
task pairs; the tail does not.

### Documentation


- `experiments/cosine/README.md`: the scripts of this round in the order they
  should be used, and two design points (diagnose exactly, train
  stochastically; read differentially) that this round established the hard way.
- Patent application serial numbers removed from all published materials on
  counsel's advice; the pending status is still stated.

## [0.2.0] : 2026-07-25

### Corrected and re-measured

**Retention results (LLM family)**
- Fixed a bug that disconnected the penalty from the autograd graph in the
  LoRA/LLM experiments (`.data` on the penalized tensors). All LLM retention
  numbers were re-measured with the penalty verified connected.
- Row-norm control re-run with its OWN hyperparameter grid (`run_rownorm.py`).
  In the original overnight run this arm was the only one never swept: it was
  launched with the calibration target selected for Omega-S and collapsed
  (retention 20.7%), so beating it established nothing. Properly swept over
  {0.001, 0.003, 0.01, 0.03} it does not reliably beat no regularisation
  (6/10 seeds, p=0.38; absolute HumanEval 0.174 vs 0.168) and carries the
  highest variance of any arm, while Omega-S exceeds it on 8/10 (p=0.055).
  Evidence that the effect is not simply row-norm equalisation, short of
  conventional significance at ten seeds. Caveat recorded: the control's
  strength was selected on the two seeds that proved to be its own worst.
  Data in `results/rownorm_control_20260726.json`.

- Re-ran retention on Llama-3-8B with 10 seeds. Primary metric is now absolute
  code capability kept (HumanEval pass@1 after the prose task): Omega-S keeps
  more than no regularisation on 9/10 seeds, 0.168 -> 0.223, +33% relative,
  sign test p=0.011. As a retention ratio the same comparison is 8/10
  (63.1% -> 76.6%); weight decay 9/10 and EWC 8/10 are reported as secondary.
  The column previously labelled "plasticity" was the same quantity as the
  retention numerator and is no longer described as plasticity: nothing in
  this experiment measures the second task. Hyperparameter selection was
  single-seed and weaker than earlier notes claimed; see paper section 4.5.
- The raw Tr((WW^T)^3) form is the worst arm (53.9%); the library and scripts
  are unified on the log-ratio `StochasticOmegaS`.

**Mechanism (measured, new)**
- Direct measurement (`experiments/measure_structure.py`): the penalty operates
  by reducing node-degree variance, not clustering. The clustering term is
  saturated by the sigmoid used to build the adjacency (dC ~= 0 across seeds).
  Omega-S is topological by definition but operates via degree-variance control
  as formulated. No two-regime/double mechanism.
- Explored reformulations of C to restore the clustering channel
  (`experiments/reform_c.py`): only rank-normalisation decompresses C;
  temperature and degree-normalisation do not. Not pursued to training; see
  `HALLAZGOS_canal_topologico.md`.

**Superseded**
- Earlier headline numbers (83.03/81.07, +2.23pp, 2-3 seeds) came from the
  disconnected penalty and are superseded. GPT-2 FLOPs gain over group-lasso
  did not survive a connected re-run.

## [0.1.0] : 2026-07-18

### Initial release

**Core library**
- `StochasticOmegaS`: single-node topological regularizer using Hutchinson
  trace estimation of Tr(A³) on WW⊤. O(N²) complexity per layer.
- `DistributedOmegaS`: multi-GPU FSDP-compatible variant with synchronous
  all-reduce of topological metrics across nodes.
- HuggingFace + PEFT integration utilities for LoRA adapter injection.

**Validated results (from preprint)**
- 8× greater degree variance reduction vs. weight decay (0.136 vs. 1.12)
  with only 0.54pp accuracy cost vs. 2.90pp for weight decay.
- 54.46% real FLOPs reduction via topology-guided structural pruning
  (vs. 25.27% under equivalent weight decay), with zero accuracy degradation.
- +3.7% latency overhead on single RTX 4090 at K=10.
- +1.5% latency overhead on 2× RTX 4090 FSDP at K=10, with zero
  all-gather communication cost on LoRA adapters.
- +13 MB VRAM overhead (+0.06%) on Llama-3-8B.

**Experiments included**
- exp1: Structural control metrics (Omega-S vs. Baseline vs. Weight Decay)
- exp2: Sparsity structure analysis
- exp3: Omega-S + Group-Lasso combination
- exp4: Group-lasso λ sweep
- exp5: Structural pruning validation (FLOPs reduction)
- exp6: Weight Decay vs. Omega-S pruning comparison
- exp7: Chatbot orchestration topology study

**Fase 3 results**
- Segunda semilla (SEED=123): retención 91.94% Omega-S vs 86.21% baseline (+5.73pp)
- Media 2 semillas: +4.26pp retención, resultado CONSISTENTE y robusto
- GPT-2 full fine-tuning: Omega-S genera 7-8x más sparsity estructurada que WD
  (1.47% vs 0.21% FLOPs), señal en dirección correcta, escala pendiente
- Lambda sweep: plasticidad controlable via lambda, retención necesita
  más resolución estadística para barrido completo

**Fase 2 results (Llama-3-8B, sequential fine-tuning)**
- 20% relative reduction in catastrophic forgetting (6.10% → 4.88%)
- Knowledge retention: 86.67% (Omega-S) vs 83.87% (baseline LoRA)
- Validated on CodeAlpaca-20k → Wikitext-2 transfer, HumanEval metric
- Plasticity cost confirmed: -1.22pp on new-domain learning (controllable via λ)

**Patent**
- USPTO Provisional Patent Application Serial No.  filed.

**Fase 4 (pruning guiado por mapa topológico : cerrado)**
- Pruning por zeroing guiado por Omega-S sobre LoRA no predice redundancia
  en modelo base : resultado negativo informativo
- 4C mostró ventaja aparente (+3.66pp) pero artefacto de 0% sparsity real

**Fase 5 (Omega-S + Wanda : trabajo en curso)**
- v1 (score multiplicativo): sin ventaja en 30%, 50%, 70% sparsity
- v2 (sparsity no uniforme, norm. min-max): sparsity real 15.2% vs objetivo 30%
- v3 (norm. por percentil p10-p90): sin ventaja (-3.05pp coste)
- Conclusión: integración con Wanda requiere recalibración post-pruning
  o señal topológica calculada con activaciones reales : trabajo futuro

**Cambios paper (v2 → v3)**
- Título: "Large Language Models" → "Neural Networks" (más honesto con los datos)
- Sección 4.4: +24.4% preliminar → tabla real 3 semillas (+2.23pp media, 2/3 positivas)
- Sección 2.5 nueva: Non-Uniform Structured Pruning (OWL, AlphaPruning, SV-NUP, DarwinLM)
- Referencias 13-19 añadidas
- Quick Start añadido al final del paper
