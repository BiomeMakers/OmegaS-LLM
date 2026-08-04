# Omega-S: Stop Your LLM From Forgetting

> *A drop-in penalty that reduces catastrophic forgetting when you fine-tune,
> and that needs nothing from the previous task.*

**Alberto Acedo** · Biome Makers Inc.
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)
[![Patent Pending](https://img.shields.io/badge/USPTO-Patent%20Pending%2064%2F121%2C656-blue)](https://www.uspto.gov)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-red)](https://arxiv.org)

---

## Three lines, and the number

```python
from omega_s import StochasticOmegaS

omega = StochasticOmegaS(num_samples=16)
loss = task_loss + lam * sum(omega(m.weight) for m in target_modules)
```

Applied every ten steps it adds **under 0.4%** to the cost of a training step.

Llama-3-8B, LoRA, fine-tuned from code to prose, HumanEval retention, ten
seeds, every arm tuned:

| | absolute pass@1 after the new task | retention ratio |
|---|---|---|
| no regulariser | 0.173 | 62.9% |
| weight decay (tuned) | 0.174 | 61.9% |
| EWC (tuned) | 0.190 | 69.1% |
| **Omega-S** | **0.238** | **84.1%** |

Omega-S wins on 9 of 10 seeds against no regulariser (sign test p=0.011,
Wilcoxon p=0.006), 10 of 10 against weight decay and 8 of 10 against EWC, all
on absolute capability. **Every arm was measured in the same session on the same
hardware**, because repeating an identical run in this setting moves the result
by 0.104 in standard deviation (see below) and comparing across sessions would
mix the effect with that variation.

Better on **9 of 10 seeds** against no regularisation (exact sign test,
one-sided *p* = 0.011). No previous-task data, no Fisher matrix, no stored
copy of the old weights, which is what EWC needs.

**What we would find most useful:** someone running this on a different model
or task pair. If it does not reproduce for you, that is the most valuable thing
you could tell us. See [Quick Start](#quick-start).

---

## Run the mechanism check on your model

The claim that three of the four factors are numerically inert is a claim about
one construction on one model's weights. The check runs on base weights, needs
no training and no GPU:

```bash
python experiments/check_M.py
```

**[Report what you get →](../../issues/new?template=replication.yml)**

A report that the factors are live on some other model is as useful to us as one
that confirms they are not: it would mean the reduction is specific rather than
general, and we would rather know.

## An implementation error we found and corrected

The index is defined with a **modularity** factor in the denominator. The
reference implementation estimated **algebraic connectivity**, which is the
inverse quantity, so the objective being optimised differed from the one
defined. We found it by checking the applied manuscript against the theoretical
one, measured first whether it was material, and then corrected it and re-ran
every comparison.

The framework's orientation is the better of the two (84.1% against 76.6%,
winning 8 of 10 seeds). But **not for the reason one would assume**: the
modularity term stays numerically inert either way. Its elasticity with respect
to the weights is 0.0001 before training, 0.0001 after training, and it changes
by +0.06% during a run, against -5.87% for degree variance. What inverting it
changes is the *scale* of the objective, from -1.06 to -10.73; since the penalty
coefficient is calibrated by gradient ratio, a different scale gives a different
operating point for the factor that does act.

The control: applying the same inversion to the cosine construction, where the
objective does not jump in scale, changes nothing (4/10 seeds, p=0.83).

**And the honest bound**: the paired difference between the two orientations is
+0.075, below the run-to-run standard deviation of 0.104. The two orientations
are *not distinguishable from each other* at ten seeds. What is distinguishable
is each of them against the baselines.

Data: `results/minv_10seeds.json`, `results/mecanismo_minv.json`.

## How much run-to-run variation this setting carries

Repeating an identical configuration, same seed and same hardware, gives a
standard deviation of **0.104** in retention ratio, with individual pairs as far
apart as 0.596 and 0.793. Re-measuring the four baseline arms months later
reproduced their *means* to within 0.04 but not their per-seed values.

Two consequences. The seed does not identify the run, so a seed-paired test is
comparing runs that differ by more than the seed. And any mean difference below
roughly 0.066 is not distinguishable here.

That GPU non-determinism dominates seed variance is established in vision
([Morin & Willetts 2020](https://arxiv.org/abs/2001.11396): 74-87% of the
observed standard deviation) and in RL
([Nagarajan et al. 2018](https://arxiv.org/abs/1809.05676)). We have not found
it quantified for low-rank fine-tuning of language models, and we report it
because it bounds every seed-paired comparison in this literature, ours
included.

## If you run it, tell us

The placement diagnostic (`experiments/adapter_placement.py`) takes minutes, does
not train anything, and needs no data beyond a few batches of each of your two
tasks. It is validated with training on one model and one task pair, and the head
of its ranking reproduces on three model families, which is not the same as
knowing that it transfers to yours.

**[Open a replication report](../../issues/new?template=replication.yml)** with
what came out. A negative result is as useful to us as a positive one and we
would rather have it: if the diagnostic does not hold up outside the cases we
tested, that is the single most valuable thing anyone can tell us right now.

If you also measured run-to-run variation in your setup, please include it. Ours
is 0.104 in retention ratio and we have not found the quantity reported for this
regime, so any second data point is worth having.

## What we do not claim

We measured the mechanism rather than asserting it, and it is not what the name
suggests. The objective is topological, built from `Tr(A³)`, but the clustering
term is inert in this formulation: the effect is carried by the variance of node
degrees. We then built the reformulation that would make the topological channel
live, and it made retention **worse** on all ten seeds. That is in the paper, in
[`CHANGELOG.md`](CHANGELOG.md) and in [`results/`](results/), with the per-run
JSON for every number.

Ten seeds establish direction, not certainty. Several supporting comparisons sit
near *p* = 0.05 without clearing it and we say so at each one. Full audit trail
in [`AUDIT.md`](AUDIT.md).

---

## The Problem

Fine-tuning large language models on new data degrades previously learned capabilities. Update a general-purpose model on customer support data, and it loses coding ability. Fine-tune on legal documents, and general reasoning degrades.

Standard regularisers don't solve this. Weight decay and SAM operate on loss landscape geometry : they can't see the structural root cause: **weight monopolies**.

## The Solution

Omega-S is a **drop-in regularizer** that targets weight monopolies directly. It adds **1.5% training overhead**, requires **no architectural changes**, and needs **no access to previous data**.

```python
from omega_s import StochasticOmegaS

omega = StochasticOmegaS(lambda_omega=0.05, k=10, n_probes=3)
# note: these are the five-seed-run defaults. The ten-seed results in the table
# below used n_probes=16 and a lambda calibrated at run time (paper §4.1).

# Add to your existing training loop : that's it
loss = model(inputs, labels=inputs).loss + omega(model)
loss.backward()
```

## Results

| Metric | Value | vs. Baseline |
|--------|-------|-------------|
| **Code capability kept (Llama-3-8B)** | **0.238** HumanEval pass@1 after the prose task, **9/10 seeds** over no regulariser (p=0.011, Wilcoxon p=0.006) | vs. 0.173 none, **+37.7% relative** |
| Same, as retention ratio | **84.1%** mean, 10 seeds | vs. 62.9% none, 61.9% WD, 69.1% EWC, all re-measured in one session |
| **Mechanism** | alignment channel leads the magnitude channel **9/10 seeds** (sign test *p*=0.011) | but neither reproduces the full objective alone: it needs both module types |
| **Method note** | log-ratio `StochasticOmegaS` (raw Tr penalty = worst arm, 53.9%, below no regulariser) | see AUDIT.md |
| **Cosine reformulation** | **0.537** mean retention, **loses 0/10** vs. the original form (Wilcoxon *W*=0, *p*=0.002) | activating the clustering channel makes retention *worse* |
| ↳ *what the raw form adds* | 6/10 → **9/10** seeds prunable | **reliability**, and only the raw form: the composite and the cosine form both sit at 6/10, exactly where group lasso alone sits |
| ↳ *with post-pruning recovery* | 8/10 → **10/10** | the gap narrows from +3 to +2 seeds and survives; two group-lasso seeds stay broken even after retraining |
| ⚠️ *which form* | retention uses the **log-ratio composite**; pruning uses the **raw trace penalty** | different objectives, see paper §3 |
| ⚠️ *and they rank oppositely* | composite wins retention and adds nothing to pruning; raw is worst on retention and is the only one that helps pruning | two methods, not one |
| Comparison vs EWC | open: mixed under unswept EWC, matched sweep in progress | : |
| Latency overhead (K=10, 1GPU) | **+3.7%** | negligible |
| Latency overhead (K=10, 2GPU FSDP) | **+1.5%** | zero network cost |
| VRAM overhead | **+13 MB** | +0.06% |

### The cosine round: a negative result we ran on ourselves

The preprint's appendix identified the saturating map in the construction of
`A` as the reason the clustering channel is inert, and offered a
contrast-preserving reformulation as an open direction. We built it and tested
it. It does what it was designed to do and it makes retention worse.

| arm | mean retention (10 seeds) | what its clustering term does |
|---|---|---|
| `omega_lib`, original construction | **0.841** | inert, saturated by the logistic map |
| `cos_composite`, cosine construction | 0.537 | live and steerable |
| `cos_clustonly`, clustering term alone | 0.477 | live, and the only thing acting |

The ordering is monotone in how active the clustering term is, in the wrong
direction. Along the way the construction was verified to do its job: the
excess statistic `C/D` moves from 1.0000 (dead) to a median of 1.36, the
library's own clustering term moves off its ceiling from 0.9999 to 0.80, and
the sign of the penalty controls the direction of movement in 16 of 16
measurements with a monotone dose response. It is not that the reformulation
failed to engage. It engaged, and the effect was negative.

Two further findings from the same round are recorded in
[`results/mechanism_decomposition_20260731.json`](results/mechanism_decomposition_20260731.json):

- **A construction detail we had not documented.** `StochasticOmegaS` builds
  `A = sigmoid(|W W^T|)` only when `W` is non-square; for square `W` it applies
  `sigmoid(|W|)` elementwise. In Llama-3-8B `q_proj` is square and `v_proj` is
  not, so the published runs used one of each. The degree-variance term differs
  by a factor of roughly 400 between the two branches. The numbers are
  unaffected, having been produced by the code as written; the description of
  the method is what needs correcting.
- **What the degree sequence encodes.** Regressing degrees on row norms, the
  square branch is 96% magnitude and the Gram branch is 71% alignment. The
  regulariser applies two different criteria depending on module shape.
- **Three of the four factors are numerically inert.** Elasticity with respect
  to the weights, median over ten attention projections: `C` 0.0000, `D` 0.0000,
  `M` 0.0001, **`Coex` 0.0091**. As implemented, the composite objective reduces
  in practice to a penalty on degree variance, and the orientation of the other
  three terms has no measurable effect. Script: `experiments/cosine/check_M.py`,
  runs on a laptop in minutes.

Scripts, in the order they should be used, are in
[`experiments/cosine/`](experiments/cosine/).

## Quick Start

### LoRA + HuggingFace (recommended)

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from omega_s import StochasticOmegaS

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
                                          r=8, lora_alpha=16,
                                          target_modules=["q_proj", "v_proj"]))
omega = StochasticOmegaS(lambda_omega=0.05, k=10)

for step, batch in enumerate(dataloader):
    loss = model(**batch).loss + omega(model)
    loss.backward()
    optimizer.step()
```

See [`examples/quickstart_gpt2.py`](examples/quickstart_gpt2.py) for a complete working example.

### FSDP (multi-GPU)

```python
from omega_s import DistributedOmegaS
omega = DistributedOmegaS(lambda_omega=0.05, k=10)
# Zero inter-GPU communication cost on LoRA adapters
```

## How It Works

```
A = W · Wᵀ                          # weight adjacency matrix
Tr(A³) ≈ (1/n) Σ zᵢᵀ A³ zᵢ        # Hutchinson stochastic estimator
A³z = W(Wᵀ(W(Wᵀ(W(Wᵀz)))))        # O(N²) : no materialization of A
Ω = λ · Σₗ Tr(Aₗ³) / |numel(Wₗ)|  # per-layer penalty
```

The method reduces complexity from O(N³) to **O(N²)** by avoiding materializing A explicitly.

## Installation

```bash
git clone https://github.com/BiomeMakers/OmegaS-LLM.git
cd omega-s
pip install -r requirements.txt
# or: pip install -e .
```

## Reproducing Paper Results

```bash
# Structural control (Table 1)
python experiments/exp1_structural_control.py

# FLOPs reduction (Tables 2 & 3)
python experiments/exp4_lambda_sweep.py
python experiments/exp5_structural_pruning.py

# Catastrophic forgetting : 3 seeds (Table 4)
# Requires 2x A100 80GB, ~$30 on RunPod
CUDA_VISIBLE_DEVICES=0 python experiments/fase2_baseline.py > log_baseline.txt &
CUDA_VISIBLE_DEVICES=1 python experiments/fase2_omega.py > log_omega.txt &
wait && python experiments/fase2_analisis.py

# Additional seeds
CUDA_VISIBLE_DEVICES=0 python experiments/fase3_semilla2.py  # SEED=123
CUDA_VISIBLE_DEVICES=0 python experiments/fase3_semilla3.py  # SEED=456

# Infrastructure benchmarks
python benchmarks/benchmark_1gpu_lora.py
torchrun --nproc_per_node=2 benchmarks/benchmark_2gpu_fsdp.py

# GPT-2 FLOPs in transformer (exploratory)
CUDA_VISIBLE_DEVICES=0 python experiments/fase3_gpt2_flops.py

# Wanda+Omega-S pruning (exploratory : work in progress)
CUDA_VISIBLE_DEVICES=0 python experiments/fase5v3_wanda_nonuniform.py
```

## Call for Distributed Validation

The results above are validated in the paper. We now invite the community to
validate **new applications** of the regularizer on GPU : starting with
**continual learning** (the direct extension of the forgetting-resistance
result) and **model merging**. Because single-run results can mislead, the
protocol enforces multi-seed reporting, a properly tuned weight-decay baseline,
pre-registered metrics, and honest reporting of null results.

See **[docs/VALIDATION_PROTOCOL.md](docs/VALIDATION_PROTOCOL.md)** for the full
protocol, hyperparameter grid, and reporting template. Contributions : including
null results : are aggregated across contributors for the record.

## Repository Structure

```
omega-s/
├── omega_s/                    # Core library
│   ├── omega_s.py              # StochasticOmegaS (single-node)
│   ├── omega_s_distributed.py  # DistributedOmegaS (FSDP)
│   └── integration_huggingface.py
├── benchmarks/                 # Infrastructure profiling
├── experiments/                # Reproducible paper experiments
│   ├── rerun_retention.py      # Ten-seed retention harness (all arms)
│   └── cosine/                 # July 2026 cosine round: patches + static screens
├── results/                    # Per-run JSON, one file per experiment
│   ├── merged_10seeds.json                     # main table
│   ├── rownorm_control_20260726.json           # row-norm control, own grid
│   ├── cosine_construction_20260731.json       # cosine round, full protocol
│   ├── flops_by_form_20260731.json            # pruning by penalty form, 10 seeds
│   ├── rownorm_t0.003_20260731.json            # off-optimum confirmation
│   ├── mechanism_decomposition_20260731.json   # what Coex is made of
│   └── module_ablation_20260731.json           # which channel carries the effect
├── examples/                   # Quickstart examples
└── omega_s_preprint.pdf        # Research preprint
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_omega` | 0.05 | Regularization strength |
| `k` | 10 | Apply every K steps |
| `n_probes` | 3 | Hutchinson probe vectors |

**Starting point:** `lambda_omega=0.05`, `k=10`. Reduce `lambda_omega` if accuracy drops >1pp. Increase `k` to reduce overhead.

## Citation

```bibtex
@article{acedo2026omegas,
  title   = {Omega-S: A Functional Resilience Index for LLM Fine-Tuning},
  author  = {Acedo, Alberto},
  year    = {2026},
  note    = {USPTO Patent Pending No. 64/121,656},
  url     = {https://github.com/BiomeMakers/OmegaS-LLM}
}
```

## License & Commercial Use

**Research:** AGPL-3.0 : free for academic research and non-profit use. See [`LICENSE`](LICENSE).

**Commercial:** Production deployment, integration into commercial pipelines, or corporate R&D requires a separate license. See [`COMMERCIAL-LICENSE.md`](COMMERCIAL-LICENSE.md).

**Contact:** acedo@biomemakers.com

---

## Conceptual Origin

Omega-S was motivated by empirical observations of microbial soil network topology ([Ortiz-Álvarez et al., 2021](https://doi.org/10.1128/mSystems.00344-21); [Saati-Santamaría et al., 2026](https://doi.org/10.1111/gcb.70984)). Those studies report that agricultural disturbance reorganises the association structure of soil communities in directions their authors link to lower resilience. Neither reports degree distributions or power-law fits, so no scale-free claim is made here; earlier versions of this README did, and the sources do not support it.

*USPTO Patent Pending No. 64/121,656 · © 2024-2026 Alberto Acedo*

**Alberto Acedo** · Biome Makers Inc.  
[![Patent Pending](https://img.shields.io/badge/USPTO-Patent%20Pending%2064%2F121%2C656-blue)](https://www.uspto.gov)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-red)](https://arxiv.org)

---

## What is Omega-S?

Omega-S is a regularizer for large-scale neural networks whose objective is built on the graph structure of weight matrices. It targets *weight monopolies*: the tendency of a small subset of neurons to concentrate disproportionate connectivity during training. By construction its objective is topological (built on Tr(A^3)); as measured, it operates by controlling the variance of node degrees (see AUDIT.md). Unlike curvature-based regularizers (SAM, weight decay), it acts on the weight graph rather than the loss landscape.

The core insight originates from empirical observations of microbial soil networks: how connectivity is *distributed* over a network, and not only how much of it there is, carries information about robustness that node counts and diversity indices do not. Omega-S imports that intuition into artificial neural networks via a penalty built on Tr(A³) of the weight adjacency matrix A = WW⊤, estimated efficiently with the Hutchinson stochastic trace estimator. A direct measurement (paper §4.6) shows that in this formulation the effect is carried by degree-variance reduction, not by the clustering term.

### Key Results

| Metric | Value | vs. Baseline |
|--------|-------|-------------|
| Degree variance reduction | 0.136 (vs. 41.94) | **308× lower** |
| Code capability kept (Llama-3-8B) | **0.238** absolute HumanEval, 9/10 seeds over no regulariser, p=0.011 | vs. 0.173 none, +37.7% relative |
| Same, as retention ratio | **84.1%** mean (9/10 over no regulariser, 10/10 over WD, 8/10 over EWC on absolute capability) | vs. 62.9% none, 61.9% WD, 69.1% EWC |
| Baseline comparisons | secondary: selection was two-seed and both baselines land below no regularisation | see paper §4.5 |
| Row-norm control (own sweep) | **8/10** for Omega-S, p=0.055; the control does not reliably beat no regularisation (6/10) | `results/rownorm_control_20260726.json` |
| Row-norm control | omega beats it 10/10 | not equivalent to row-norm balancing |
| FLOPs in transformer (GPT-2) | superseded: Omega+GL does not beat GL alone (−0.72% vs −0.77%) | connected re-run |
| Latency overhead (K=10, 1GPU) | **+3.7%** | negligible |
| Latency overhead (K=10, 2GPU FSDP) | **+1.5%** | zero network cost |
| VRAM overhead | **+13 MB** | +0.06% |

---

## How It Works

```
A = W · Wᵀ                          # pseudo-topological adjacency matrix
Tr(A³) ≈ (1/n) Σ zᵢᵀ A³ zᵢ        # Hutchinson stochastic estimator
A³z = W(Wᵀ(W(Wᵀ(W(Wᵀz)))))        # O(N²) via sequential matrix-vector products
Ω = λ · Σₗ Tr(Aₗ³) / |numel(Wₗ)|  # per-layer penalty, added to training loss
```

The method reduces complexity from O(N³) (explicit A³) to **O(N²)** by avoiding materializing A explicitly. When applied to LoRA adapters in FSDP environments, **zero inter-GPU communication** is required.

---

## Installation

```bash
git clone https://github.com/BiomeMakers/OmegaS-LLM.git
cd omega-s
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

---

## Quick Start

### Single-node (any model with nn.Linear layers)

```python
from omega_s import StochasticOmegaS

omega = StochasticOmegaS(num_samples=3)

# In your training loop:
task_loss = model(inputs, labels=inputs).loss

# Apply Omega-S every K steps
if step % 10 == 0:
    for name, param in model.named_parameters():
        if param.requires_grad:
            omega_loss += omega(param)
    task_loss = task_loss + 0.05 * omega_loss

task_loss.backward()
optimizer.step()
```

### LoRA + HuggingFace (recommended for LLMs)

See [`examples/quickstart_gpt2.py`](examples/quickstart_gpt2.py) for a complete working example with GPT-2 and LoRA adapters.

### Distributed (FSDP, multi-GPU)

```python
from omega_s import DistributedOmegaS
omega = DistributedOmegaS(num_samples=3)
# Usage identical to single-node; synchronization handled internally
```

See [`benchmarks/benchmark_2gpu_fsdp.py`](benchmarks/benchmark_2gpu_fsdp.py) for the full FSDP profiling setup.

---

## Repository Structure

```
omega-s/
│
├── omega_s/                        # Core library
│   ├── __init__.py
│   ├── omega_s.py                  # Single-node regularizer (StochasticOmegaS)
│   ├── omega_s_distributed.py      # Distributed regularizer (DistributedOmegaS, FSDP)
│   └── integration_huggingface.py  # HuggingFace + PEFT integration utilities
│
├── benchmarks/                     # Infrastructure profiling scripts
│   ├── benchmark_1gpu_lora.py      # Single RTX 4090 : latency & VRAM profiling
│   └── benchmark_2gpu_fsdp.py      # 2× GPU FSDP : distributed overhead profiling
│
├── experiments/                    # Reproducible experiments from the paper
│   ├── exp1_structural_control.py  # Table 1: Omega-S vs. Baseline vs. Weight Decay
│   ├── exp2_sparsity_analysis.py   # Sparsity structure analysis (structured vs. dispersed)
│   ├── exp3_group_sparsity.py      # Omega-S + Group-Lasso combination
│   ├── exp4_lambda_sweep.py        # Group-lasso λ sweep (Table 2)
│   ├── exp5_structural_pruning.py  # FLOPs reduction validation (Table 3)
│   ├── exp6_wd_vs_omega_pruning.py # Weight Decay vs. Omega-S pruning comparison
│   └── exp7_orchestration_topology.py  # Chatbot orchestration topology test
│
├── examples/
│   └── quickstart_gpt2.py          # End-to-end example with GPT-2 + LoRA
│
├── omega_s_preprint.pdf            # Research preprint
├── requirements.txt
├── setup.py
├── LICENSE                         # AGPL-3.0 (research use)
└── COMMERCIAL-LICENSE.md           # Commercial licensing terms
```

---

## Reproducing Paper Results

All experiments use seed=42. Run from the repo root:

```bash
# Table 1: Structural control (Omega-S vs. Weight Decay vs. Baseline)
python experiments/exp1_structural_control.py

# Tables 2 & 3: Group-lasso sweep + structural pruning
python experiments/exp4_lambda_sweep.py
python experiments/exp5_structural_pruning.py

# Table 3 comparison: Omega-S+GL vs. Weight Decay (pruning pipeline)
python experiments/exp6_wd_vs_omega_pruning.py

# Infrastructure benchmarks (requires GPU)
python benchmarks/benchmark_1gpu_lora.py

# FSDP benchmark (requires 2 GPUs)
torchrun --nproc_per_node=2 benchmarks/benchmark_2gpu_fsdp.py

# Fase 2: Catastrophic forgetting (requires 2x A100, ~$30 on RunPod)
# GPU 0: Baseline LoRA
CUDA_VISIBLE_DEVICES=0 python experiments/exp8_fase2_baseline.py > log_baseline.txt &
# GPU 1: Omega-S LoRA
CUDA_VISIBLE_DEVICES=1 python experiments/exp9_fase2_omega_s.py > log_omega.txt &
# Analysis (after both complete)
python experiments/exp10_fase2_analisis.py
```

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_samples` | 3 | Hutchinson probe vectors per layer |
| `λ_omega` | 0.05 | Regularization strength |
| `K` | 10 | Apply every K training steps |
| `GL_lambda` | 1e-2 | Group-lasso strength (for pruning) |

**Recommended starting point:** K=10, λ=0.05. Reduce λ if accuracy degrades more than 1pp. Increase K if compute overhead is a concern.

---

## Citation

If you use Omega-S in your research, please cite:

```bibtex
@article{acedo2026omegas,
  title   = {Omega-S: A Functional Resilience Index for LLM Fine-Tuning},
  author  = {Acedo, Alberto},
  year    = {2026},
  note    = {USPTO Patent Pending No. 64/121,656},
  url     = {https://github.com/BiomeMakers/OmegaS-LLM}
}
```

---

## License

**Research use:** AGPL-3.0 : free for academic research, education, and non-profit experimentation. See [`LICENSE`](LICENSE).

**Commercial use:** Requires a separate commercial license covering both software and patent rights. See [`COMMERCIAL-LICENSE.md`](COMMERCIAL-LICENSE.md) for terms and contact information.

---

## Conceptual Origin

The Omega-S regularizer was conceptually motivated by empirical observations of microbial soil network topology in vineyard and agricultural ecosystems ([Ortiz-Álvarez et al., 2021](https://doi.org/10.1128/mSystems.00344-21); [Saati-Santamaría et al., 2026](https://doi.org/10.1111/gcb.70984)). Those studies report that agricultural disturbance reorganises soil network structure in directions linked to lower resilience; they do not report degree distributions, and no scale-free claim is made here.

---

*USPTO Patent Pending No. 64/121,656*  
*© 2024-2026 Alberto Acedo. All rights reserved.*
