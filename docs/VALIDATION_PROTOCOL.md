# Omega-S : Distributed Validation Protocol (Call for Reproduction)

This document specifies experiments the community can run on GPU to validate
**new applications** of the Omega-S regularizer beyond the results in the paper.
The goal is **third-party replicated validation**: single-run results do not
count. Everything is reported with the template at the end, and aggregated
across contributors for the record.

> **Use the reference implementation** `omega_s/omega_s.py`
> (`OmegaSRegularizer`). **Do not reimplement the objective.** It is a composite
> log-ratio combining a Hutchinson estimate of normalized clustering
> (`Tr(A³)/‖A‖³`), a power-iteration modularity estimate, and the degree
> variance of the pseudo-adjacency : *not* a plain `Tr(A³)` penalty. Naive
> reimplementations (e.g. a bare third-moment term) do **not** reproduce the
> validated behaviour.

---

## Ground rules (anti-mirage discipline)

These are mandatory. They exist because promising single-run results have a
habit of vanishing on replication.

1. **Minimum 5 seeds** per condition. Report **mean ± std**, never a single run.
2. **Strong baseline, not a straw man.** The class-appropriate rival is **EWC**
   (Elastic Weight Consolidation), *not* weight decay: weight decay is not a
   continual-learning method and a comparison against it alone is insufficient.
   Include `wd` and `none` as controls, and tune λ for *each* method equally.
   Where resources allow, add an incumbent from the **projection** family
   (O-LoRA or OSFT): on multi-task benchmarks projection methods substantially
   outperform penalty methods (~76-79 vs ~50-52 for EWC/LwF), so they are the
   honest upper reference even though Omega-S belongs to the penalty family.
3. **Pre-register the primary metric** before running. Do not pick the metric
   that looks best afterward.
4. **Paired significance test** (Wilcoxon signed-rank or paired t) of Omega-S vs
   **EWC** (and vs weight decay) across seeds. Report p-value and confidence
   interval.
5. **Report nulls.** A flat or negative result is valid and necessary. Do not
   drop it.
6. **Fix the hyperparameter grid** below so runs are comparable across
   contributors.
7. **Record the commit hash** of the implementation and the hardware used.

---

## Batch 1 : Priority experiments

### Experiment A : Continual learning (PRIORITY 1)

The direct extension of the validated property (forgetting resistance). Extends
the published result (Llama-3-8B + LoRA, CodeAlpaca→Wikitext-2) to a **sequence
of tasks**.

- **Model:** Llama-3-8B + LoRA (preferred). Lower-GPU alternative: an open 1-3B
  model (report which).
- **Task sequence:** ≥3 sequential fine-tuning tasks (e.g. instruction → code →
  technical domain). Document the exact sequence and order.
- **Conditions:** `omega` (real Omega-S) · **`ewc`** · `wd` (weight decay) · `none`.
  λ tuned separately for each. **EWC is the class-appropriate baseline** : weight
  decay is not a continual-learning method, and a comparison against it alone is
  not sufficient. If resources allow, add a projection-family incumbent
  (O-LoRA or OSFT); on multi-task benchmarks those substantially outperform
  penalty methods, so they are the honest upper reference even though Omega-S
  belongs to the penalty family.
- **Primary metric (pre-registered):** **mean forgetting** = average drop in
  per-task accuracy between "just after learning it" and "end of the sequence"
  (lower is better).
- **Secondary:** final average accuracy, backward transfer.
- **Procedure:** train task by task without resetting; after each task, evaluate
  all tasks seen so far. 5+ seeds (fixed task order; vary only the
  initialization/optimization seed).

### Experiment B : Model merging (PRIORITY 2)

The only application with a (weak) positive signal in preliminary CPU probes;
deserves real validation.

- **Setup:** train 2 adapters/models on 2 distinct tasks or domains, with each
  condition (`omega`/`wd`/`none`).
- **Merge:** weight averaging (model soup) and/or task arithmetic. Document the
  method.
- **Primary metric:** mean accuracy of the merged model across the 2 tasks
  (interference).
- **Hypothesis to falsify:** Omega-S-regularized weights (lower degree variance)
  → less interference on merge. 5+ seeds.

---

## Out of scope for Batch 1

- **Training-dynamics monitor** (Omega-S_t to anticipate phase transitions /
  grokking): preliminary signal was modest and a nestedness variant did **not**
  replicate across seeds. Excluded from Batch 1 to avoid burning credibility;
  revisited with evidence.
- **Quantization / edge:** in preliminary probes Omega-S reduced degree variance
  but did **not** improve quantization (weight decay was better). Low priority
  unless a different protocol is proposed.
- **Federated / RL plasticity:** require dedicated setups (multi-client / RL
  environment). Welcome as Batch 2 with their own protocol.

---

## Hyperparameter grid (fixed, for comparability)

- LoRA rank / alpha: as in the published baseline.
- λ (Omega-S): sweep `{1e-5, 3e-5, 1e-4, 3e-4}` (optimal λ scales with model
  size; report all).
- λ (weight decay): sweep `{0.0, 0.01, 0.05, 0.1}`.
- Seeds: `{0, 1, 2, 3, 4}` minimum.
- Optimizer / LR / steps: as in the published baseline (document them).

---

## Reporting template (fill in and attach as JSON or table)

```json
{
  "experiment": "A_continual | B_merging",
  "model": "Llama-3-8B+LoRA",
  "task_sequence": ["task1", "task2", "task3"],
  "condition": "omega | ewc | wd | none",
  "lambda": 0.0,
  "seeds": [0,1,2,3,4],
  "primary_metric": "mean_forgetting",
  "per_seed_values": [],
  "mean": 0.0,
  "std": 0.0,
  "test_vs_ewc": {"method": "wilcoxon", "p": 0.0, "ci95": [0.0, 0.0]},
  "test_vs_wd":  {"method": "wilcoxon", "p": 0.0, "ci95": [0.0, 0.0]},
  "commit_hash": "",
  "hardware": "",
  "notes": "include null results or anomalies here"
}
```

Contributions (**including null results**) via Pull Request or Issue. The
aggregate across all runs : mean across contributors, with their spread : is what
goes into the paper: replicated, multi-seed, defensible.

---

*License: AGPL-3.0 (research) + commercial (production). Patent USPTO .*
