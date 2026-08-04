# Cosine-construction round (July 2026)

This directory contains the scripts behind Section 4 of the preprint, *Testing
the contrast-preserving reformulation: a negative result*. They are kept
together because they only make sense as a sequence: each one exists to make
the next one affordable or interpretable.

The short version of the outcome: the reformulation works as designed (it
removes the saturation and makes the clustering channel steerable), and it
makes retention worse on all ten seeds. The open direction stated in the
appendix of the preprint is closed in the negative.

## Order of use

**1. Static screening, no GPU.**

| script | question | answer obtained |
|---|---|---|
| `verify_cosine.py` | does the cosine construction decompress the clustering channel on real Llama-3-8B weights? | yes: excess `C/D` of 1.05 to 1.84 by module, against 1.0000 for the original construction |
| `compare_branches.py` | the reference implementation branches on matrix shape; does that branch matter? | yes: the degree-variance term differs by a factor of roughly 400 between branches on `q_proj` |
| `decompose_coex.py` | what does the degree sequence actually encode? | magnitude in the square branch (96% of variance in layers 8-31), alignment in the Gram branch (71%) |

Run these first. They cost minutes on a laptop and each of them can falsify
the next step before any GPU time is spent. `verify_cosine.py` in particular
is what confirms that the object being trained is the object that was
measured, which is worth checking every time the construction changes.

**2. Patches to the training harness.** Each is idempotent, anchors by content
rather than line number, keeps a backup, validates with `ast.parse` before
writing, and aborts without writing if an anchor does not match. Apply in
order from the directory containing `rerun_retention.py`:

| script | adds |
|---|---|
| `apply_patch_1a.py` | the cosine construction, the exact `Tr(A^3)` diagnostic, and `--phase1a` (vitality screen) |
| `apply_patch_1b.py` | `--phase1b` (which sign helps retention, with HumanEval) and a fix to the lambda calibration that would otherwise raise a `TypeError` |
| `apply_patch_2.py` | the `cos_composite` arm, a replica of `StochasticOmegaS` changing only the construction of `A`, plus `--phase2` |
| `apply_patch_ablacion.py` | `OMEGA_ONLY`, to restrict the penalty to one module type while leaving the adapter unchanged |

**3. Merging screen, separate line.** `merging_screen.py` and
`merging_corr.py` test whether the index carries information about
interference between two LoRA adapters. They are here because they share the
construction, not because they belong to the retention experiment. The
outcome was also negative and is documented in the results files.

## Two design points worth carrying forward

**Diagnose exactly, train stochastically.** The Hutchinson estimator is
unbiased and cheap, which is what you want in the gradient, but at practical
sample counts its noise exceeds the signal you are trying to read. All the
diagnostics here compute `Tr(A^3)` exactly under `no_grad` and use Hutchinson
only inside the penalty.

**Read differentially.** An unregularised arm moves the same statistics during
normal training, by an amount comparable to the effect being measured. Every
screening statistic here is reported against a `none` reference or as a
contrast between two arms that consume the random number stream identically.
An absolute threshold on a within-arm change measures training drift, not the
penalty.
