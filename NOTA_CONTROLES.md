# Controls for a degree-variance regulariser in LoRA fine-tuning

Companion note to arXiv:2608.03887. Ten paired seeds on Llama-3-8B.
All figures in this note are computed from the per-cell JSON files of a single session and are reproduced by `generar_nota.py`; none is transcribed by hand.

## Summary

1. **The regulariser is not a proxy for shrinking the dominant singular direction.** A control that penalises only `sigma_max(W)^6` loses on all 10 seeds (sign test p = 0.00098, Wilcoxon p = 0.00098) and also loses to doing nothing at all. It buys retention by suppressing learning, which the regulariser does not.

2. **The published configuration replicates under a stricter tuning protocol.** With the target validated on a seed disjoint from the ten evaluation seeds, absolute retained capability is 0.2402 against 0.1750 for no regularisation, 8 wins to 1, sign test p = 0.01953.

3. **A data-free anchor baseline matches or exceeds it.** L2-SP, which was absent from the published baselines, reaches 0.2787. Head to head the two are not separated at ten seeds (6-3, sign test p = 0.254; Wilcoxon p = 0.049). Under identical treatment, where both arms receive their own strength sweep, L2-SP is ahead. See Annex A.

4. **The two methods have opposite profiles.** The regulariser increases task-A learning while L2-SP is neutral on it; L2-SP achieves a higher retention ratio. This is the substantive difference, not the headline number.

## Protocol

Llama-3-8B, LoRA rank 8 on `q_proj` and `v_proj`, `code_search_net` followed by `openwebtext`, HumanEval pass@1 measured after each task. Penalty applied every 10 steps to 8 sampled modules, strength calibrated so that the penalty gradient is a fixed fraction of the task gradient.

Two protocol changes with respect to the published run:

- **The tuning seed is disjoint from the ten evaluation seeds.** The published run tuned on seeds 42 and 123, both inside the evaluation set, which inflates the selected value. Here tuning uses seed 7.
- **Two additional columns are recorded in every cell**: HumanEval after task A (plasticity) and wikitext-2 perplexity after each task (general quality). Without the first, an arm that retains by not learning is indistinguishable from one that retains by protecting.

The primary metric is **absolute retained capability**, HumanEval pass@1 after task B, not the retention ratio. The ratio rewards an arm that learns less in task A, because it shrinks the denominator.

## Results

| Arm | Absolute | s.d. | Ratio | Task-A (plasticity) | Perplexity after B |
|---|---|---|---|---|---|
| **L2-SP (anchor to post-task-A weights)** | 0.2787 | 0.0341 | 1.0550 | 0.2671 | 8.615 |
| **Omega-S, M = 1/lambda_2 (published configuration)** | 0.2402 | 0.0463 | 0.7850 | 0.3049 | 8.521 |
| Omega-S, M = 1/lambda_2, tuned target | 0.1994 | 0.0303 | 0.6816 | 0.2939 | 8.518 |
| Omega-S, M = lambda_2 (pre-correction) | 0.1927 | 0.0298 | 0.6954 | 0.2811 | 8.527 |
| No regularisation | 0.1750 | 0.0335 | 0.6111 | 0.2848 | 8.503 |
| Tr(A^3) alone | 0.1409 | 0.0269 | 0.5406 | 0.2634 | 8.595 |
| Spectral-norm control, sigma_max^6 | 0.0945 | 0.0206 | 0.6847 | 0.1524 | 8.907 |

Ten seeds per arm. The two Omega-S rows with `1/lambda_2` differ only in penalty strength and are discussed in Annex A.

### Paired comparisons, absolute retained capability

Every comparison is paired seed by seed. Sign test and Wilcoxon signed-rank are both exact, one-sided, computed by enumeration.

| Comparison | Wins-losses | Sign test | Wilcoxon | Mean difference |
|---|---|---|---|---|
| Omega-S 0.03 vs none | 8-1 (1 tie) | 0.01953 * | 0.00586 ** | +0.0652 |
| Omega-S 0.03 vs lib | 8-2 | 0.05469 (+) | 0.00488 ** | +0.0476 |
| Omega-S 0.03 vs raw | 9-1 | 0.01074 * | 0.00195 ** | +0.0994 |
| Omega-S 0.03 vs spectral | 10-0 | 0.00098 ** | 0.00098 ** | +0.1457 |
| l2sp vs none | 10-0 | 0.00098 ** | 0.00098 ** | +0.1037 |
| l2sp vs lib | 9-0 (1 tie) | 0.00195 ** | 0.00195 ** | +0.0860 |
| l2sp vs spectral | 10-0 | 0.00098 ** | 0.00098 ** | +0.1841 |
| l2sp vs Omega-S 0.03 | 6-3 (1 tie) | 0.25391 n.s. | 0.04883 * | +0.0384 |
| lib vs none | 7-2 (1 tie) | 0.08984 (+) | 0.10156 n.s. | +0.0177 |
| none vs raw | 9-1 | 0.01074 * | 0.03223 * | +0.0341 |
| none vs spectral | 10-0 | 0.00098 ** | 0.00098 ** | +0.0805 |

Significance marks: `**` p < 0.01, `*` p < 0.05, `(+)` p < 0.10, `n.s.` otherwise. Ties are discarded from the sign test rather than awarded to the leading arm; one exact tie occurs in the main comparison, and counting it as a win would move the reported p value from 0.020 to 0.011.

### Plasticity and general quality

An arm that retains because it learned less has not solved the problem. The column below is HumanEval after task A, before any interference has occurred.

| Arm | Task-A capability | vs no regularisation | Perplexity after B |
|---|---|---|---|
| No regularisation | 0.2848 | reference | 8.503 |
| **L2-SP (anchor to post-task-A weights)** | 0.2671 | 4-5 wins, p = 0.7461 | 8.615 |
| **Omega-S, M = 1/lambda_2 (published configuration)** | 0.3049 | 7-3 wins, p = 0.1719 | 8.521 |
| Omega-S, M = 1/lambda_2, tuned target | 0.2939 | 6-4 wins, p = 0.3770 | 8.518 |
| Omega-S, M = lambda_2 (pre-correction) | 0.2811 | 5-5 wins, p = 0.6230 | 8.527 |
| Tr(A^3) alone | 0.2634 | 3-7 wins, p = 0.9453 | 8.595 |
| Spectral-norm control, sigma_max^6 | 0.1524 | 0-10 wins, p = 1.0000 | 8.907 |

The spectral control loses task-A capability on all 10 seeds (p = 0.0010), a mean drop of 46 per cent, and its perplexity is the worst of every arm. It is not a weaker version of the regulariser; it operates by suppressing capacity.

The published configuration goes the other way: task-A capability is 0.3049 against 0.2848, winning on 7/10 seeds, losing 3. It does not trade plasticity for stability. L2-SP is neutral on this axis (0.2671), which is what an anchor penalty is expected to do: it constrains displacement, and constraining displacement cannot increase learning.

## What the controls establish about the mechanism

Four candidate explanations have now been tested against the same protocol. Three are ruled out and one is not.

**Not triadic structure.** The clustering factor is numerically inert: its elasticity is at or below 1e-4 on base and trained weights. This was measured on Llama-3-8B and reproduced on two further checkpoints, `Qwen3-0.6B` and a 50 per cent compressed derivative of it, where the clustering and density factors are again inert and the degree-variance factor is 30 to 70 times larger. When the channel is made live by replacing the sigmoid construction with a cosine one, performance gets worse, not better (0/10 in an earlier run). The topological motivation does not describe what the objective does.

**Not row-norm equalisation.** A control that only equalises row norms, recalibrated until it learns task A normally, loses on all ten seeds in an earlier run and retains less than half the absolute capability.

**Not dominant-direction shrinkage.** This is the new control. It shares the homogeneity degree of the published penalty, so the same calibration procedure gives both arms the same effective strength, and the comparison is of content rather than of scale. It loses on all ten seeds against every other arm including no regularisation.

**Not the modularity factor, even after its orientation was corrected.** The published configuration inverts that factor. The gain is real, but the factor stays inert either way; what changes is the scale of the objective, which moves the operating point of the one live channel once the strength is recalibrated. Applying the same inversion to a construction where the objective does not change scale produces no effect. The correction is therefore a change of operating point, not a change of mechanism, and that is how it should be described.

What remains is the degree sequence of the weight graph. That sequence measures row magnitude in square modules and directional alignment in non-square ones, and an earlier module ablation found that the alignment channel contributes more (9/10 seeds) while neither channel alone reproduces the composite.

## Comparison of methods

| Method | What it constrains | Previous-task data | Stored reference | Task boundary | Restricts displacement | Result here |
|---|---|---|---|---|---|---|
| Weight decay | Parameter magnitude toward zero | no | no | no | yes | beaten in the published run |
| EWC | Distance to previous weights, Fisher-weighted | yes | yes | yes | yes | beaten in the published run |
| L2-SP | Distance to previous weights, uniform | no | yes | yes | yes | 0.2787, best measured |
| Row-norm control | Row norms toward equality | no | no | no | yes | loses 10/10 in an earlier run |
| Spectral-norm control | Largest singular value | no | no | no | yes | 0.0945, worst of all arms |
| Adapter placement (TIM) | Which modules may move at all | yes | no | yes | yes, as a hard binary mask | separate line of work |
| **Omega-S** | Shape of the weight graph degree sequence | no | no | no | no | 0.2402 |

The three columns in the middle are the operational difference. L2-SP and the adapter-placement method both require a moment at which one can say what is to be preserved: L2-SP needs a snapshot, placement needs both tasks in order to measure their overlap. The regulariser needs neither, which is what makes it applicable to a stream without task boundaries. Whether that regime is where it earns its place has not been measured.

One argument that does **not** survive scrutiny and should not be made: the memory cost of L2-SP's anchor. Under LoRA only the adapter parameters require gradients, so the anchor is 3,407,872 parameters, about 13.6 MB in fp32, or 0.04 per cent of an 8B model. The memory argument applies to full fine-tuning, not to the regime studied here.

## Limitations

- One model, one task pair, one benchmark, ten seeds. Run-to-run standard deviation on the retention ratio has been measured at 0.104 in this harness, so mean differences below roughly 0.066 are not distinguishable at this sample size.
- Weight decay and EWC were not re-run in this session. Their published figures come from a different session, without the perplexity and plasticity columns, and with tuning seeds inside the evaluation set.
- The strength sweep uses a single tuning seed and is shown in Annex A to be unreliable. This affects every arm, including the published one.
- The alignment and magnitude decomposition of the degree sequence was measured on base weights. It says what the quantity is made of, not which part moves during training.

## Annex A: single-seed tuning does not discriminate

Each regularised arm received a strength sweep on seed 7, disjoint from the ten evaluation seeds, and the value with the highest absolute capability was carried forward. The rule was fixed before any sweep was run.

**Spectral control** (seed 7)

| Strength | Absolute | Ratio | Task A | Perplexity |
|---|---|---|---|---|
| 0.03 **(selected)** | 0.1463 | 0.9231 | 0.1585 | 8.978 |
| 0.1 | 0.1280 | 0.7778 | 0.1646 | 9.019 |
| 0.3 | 0.0732 | 0.4615 | 0.1585 | 9.138 |
| 0.5 | 0.0976 | 0.6667 | 0.1463 | 9.166 |

**Tr(A^3) alone** (seed 7)

| Strength | Absolute | Ratio | Task A | Perplexity |
|---|---|---|---|---|
| 0.03 **(selected)** | 0.1280 | 0.4468 | 0.2866 | 8.484 |
| 0.1 | 0.0915 | 0.3061 | 0.2988 | 8.508 |
| 0.3 | 0.0915 | 0.3488 | 0.2622 | 8.539 |
| 0.5 | 0.1098 | 0.4091 | 0.2683 | 8.573 |

**Omega-S, lambda_2** (seed 7)

| Strength | Absolute | Ratio | Task A | Perplexity |
|---|---|---|---|---|
| 0.03 | 0.1768 | 0.6444 | 0.2744 | 8.525 |
| 0.1 | 0.1159 | 0.4419 | 0.2622 | 8.527 |
| 0.3 **(selected)** | 0.2988 | 0.9608 | 0.3110 | 8.616 |
| 0.5 | 0.1280 | 0.4200 | 0.3049 | 8.495 |

**Omega-S, 1/lambda_2** (seed 7)

| Strength | Absolute | Ratio | Task A | Perplexity |
|---|---|---|---|---|
| 0.003 | 0.1646 | 0.5000 | 0.3293 | 8.547 |
| 0.01 | 0.1890 | 0.8158 | 0.2317 | 8.551 |
| 0.03 | 0.1646 | 0.6585 | 0.2500 | 8.535 |
| 0.1 **(selected)** | 0.2134 | 0.7778 | 0.2744 | 8.515 |

**L2-SP** (seed 7)

| Strength | Absolute | Ratio | Task A | Perplexity |
|---|---|---|---|---|
| 0.1 | 0.2195 | 0.8780 | 0.2500 | 8.541 |
| 1.0 **(selected)** | 0.3049 | 1.0638 | 0.2866 | 8.587 |
| 10.0 | 0.2805 | 1.0698 | 0.2622 | 8.604 |
| 100.0 | 0.2561 | 0.9767 | 0.2622 | 8.628 |

### The sweep selected a worse value than the inherited one

For the published configuration the sweep is flat and non-monotone, and it selected a strength of 0.1. Evaluated on the ten seeds that strength gives 0.1994, against 0.2402 for the 0.03 value inherited from the published protocol, winning on only 2/10 seeds. Both figures are reported; neither is chosen after the fact.

The selection value itself makes the point. On the tuning seed, strength 0.1 scored 0.2134, which is **below** the 0.2402 that the 0.03 configuration actually achieves across ten seeds. A procedure whose selection statistic is worse than the realised performance of an alternative is measuring seed noise, not response to strength.

Two consequences. First, the comparison against L2-SP inherits this noise in both directions: L2-SP happened to draw a strength that held up, this arm happened to draw one that did not, under the same procedure. Second, tuning on a single seed should be replaced by tuning on two or three disjoint seeds and averaging, at three times the cost. Until that is done, no strength comparison in this family should be read as a statement about the method.

A point of logic worth stating, since the sweeps have repeatedly favoured weaker settings: weaker is not monotonically better. The limit of zero strength is no regularisation, which scores 0.1750 and is beaten. An interior optimum therefore exists; the sweeps simply lack the power to locate it.

## Annex B: no effect under full fine-tuning

A separate probe on `MultiverseComputingCAI/littlelamb`, a 0.3B model, with **full fine-tuning rather than LoRA**: 400 steps on code followed by 400 steps on prose, learning rate 1e-05, penalty applied at every step with the corrected orientation. Retention is measured as the relative rise in held-out code loss after the second task.

| Strength | Rise in code loss | Prose loss |
|---|---|---|
| none (reference) | +5.0% | 3.3331 |
| 0.03 | +5.08% | 3.3322 |
| 0.3 | +5.46% | 3.3298 |

Run-to-run noise on this quantity was measured at about 0.3 points by repeating an identical configuration, so neither value differs from the unregularised reference. The prose loss is unchanged, so the penalty is not braking the second task either. An earlier version of this probe, run before the orientation correction, gave the same null across four strengths spanning a factor of 33 in the multiplier. The effect does not extend to this regime.

This is consistent with the mechanism section. If the benefit of the orientation correction comes from a change in the scale of the objective that shifts the operating point of the degree-variance channel, there is no reason for that shift to land in a useful place when the module set changes from 64 attention projections to 196 matrices of mixed shape.

## Data

70 evaluation cells across 7 configurations, plus 21 tuning cells, one full fine-tuning probe and one unregularised anchor cell on the tuning seed. Per-cell JSON files record seed, arm, strength, the calibrated multiplier, HumanEval after each task, perplexity after each task and wall-clock seconds.

### Per-seed data, absolute retained capability

| Seed | l2sp | Omega 0.03 | Omega 0.1 | lib | none | raw | spectral |
|---|---|---|---|---|---|---|---|
| 42 | 0.2500 | 0.1951 | 0.1646 | 0.1463 | 0.1951 | 0.1646 | 0.0610 |
| 123 | 0.3171 | 0.2195 | 0.2073 | 0.1646 | 0.2256 | 0.0976 | 0.0732 |
| 456 | 0.2988 | 0.2378 | 0.1524 | 0.1829 | 0.1829 | 0.1280 | 0.1220 |
| 789 | 0.2622 | 0.3110 | 0.2500 | 0.2073 | 0.1646 | 0.1280 | 0.0915 |
| 1011 | 0.3415 | 0.1646 | 0.1890 | 0.1768 | 0.0976 | 0.1768 | 0.0732 |
| 2022 | 0.2866 | 0.2073 | 0.2256 | 0.2134 | 0.2012 | 0.1768 | 0.1037 |
| 3033 | 0.2866 | 0.2500 | 0.1768 | 0.2317 | 0.1707 | 0.1402 | 0.1159 |
| 4044 | 0.2378 | 0.2866 | 0.1890 | 0.2378 | 0.1768 | 0.1341 | 0.1159 |
| 5055 | 0.2378 | 0.2378 | 0.2195 | 0.1707 | 0.1585 | 0.1524 | 0.0915 |
| 6066 | 0.2683 | 0.2927 | 0.2195 | 0.1951 | 0.1768 | 0.1098 | 0.0976 |

