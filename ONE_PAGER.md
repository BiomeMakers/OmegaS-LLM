# Omega-S, in one page

**A drop-in penalty that reduces catastrophic forgetting when you fine-tune,
and that needs nothing from the previous task.**

Repository: https://github.com/BiomeMakers/OmegaS-LLM (AGPL-3.0 for research;
commercial licence available)

---

## The problem

Fine-tuning a model on new data degrades what it previously learned. The standard
remedy, EWC, requires keeping the previous task's data to compute a Fisher
matrix, plus a copy of the old weights. In production that is often the one
thing you cannot do: the data is gone, restricted, or too large to keep around
for every model you maintain.

## What this is

A penalty computed from the weight matrix alone. No previous-task data, no
Fisher matrix, no stored weights. Three lines in an existing training loop:

```python
from omega_s import StochasticOmegaS
omega = StochasticOmegaS(num_samples=16)
loss = task_loss + lam * sum(omega(m.weight) for m in target_modules)
```

Applied every ten steps it adds **under 0.4%** to the cost of a training step.

## What it does

**Retention.** Llama-3-8B, LoRA, fine-tuned from code to prose, HumanEval
retention over ten seeds, every arm tuned:

| | absolute pass@1 after the new task | retention ratio |
|---|---|---|
| no regulariser | 0.173 | 62.9% |
| weight decay (tuned) | 0.174 | 61.9% |
| EWC (tuned) | 0.190 | 69.1% |
| **Omega-S** | **0.238** | **84.1%** |

Better on **9 of 10 seeds** against no regularisation (exact sign test,
one-sided *p* = 0.011; Wilcoxon *p* = 0.006), 10 of 10 against tuned weight
decay (*p* = 0.002) and 8 of 10 against tuned EWC (*p* = 0.014), all on
absolute capability. Every arm was measured in the same session on the same
hardware, because repeating an identical run in this setting moves the result
by 0.104 in standard deviation.

## What we do not claim

The mechanism is not what the name suggests, and we measured it rather than
asserting it. The objective is topological, built from Tr(A³), but the
clustering term is inert in this formulation and the effect is carried by the
variance of node degrees. We built the reformulation that would make the
topological channel live, and it made retention **worse** on all ten seeds. We
report that in the paper.

Ten seeds establish direction, not certainty: several supporting comparisons
sit near *p* = 0.05 without clearing it, and we say so at each one. The
retention result is on one model and one task pair. On GPT-2 the composite
with group lasso does not beat group lasso alone, which bounds what can be
claimed: whatever the penalty contributes, it is not compression.

## What would be useful to us

Someone running it on a different model, a different task pair, or a real
compression pipeline. The repository has a quick start, the per-run JSON for
every number above, and the scripts for every experiment including the ones
that failed. If it does not reproduce for you, that is the most useful thing
you could tell us.

**Contact:** Alberto Acedo, Biome Makers Inc., via the repository.
