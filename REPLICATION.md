# Run it on your model

Section 3.5 of the paper closes with one thing we would like others to run, and
this is it. It takes minutes, needs no training, no previous-task data and no
GPU.

```bash
MODEL=your/model python experiments/check_M.py
```

The model has to be in your local Hugging Face cache already, or given as a
path. Nothing is downloaded.

## What it measures

The objective of Omega-S has four factors: clustering C, density D, a modularity
term M and the degree variance Coex. The script measures the **elasticity** of
each one, that is the relative change in the factor produced by a relative
change in the weights. It is dimensionless, so it is comparable across factors,
modules and models, and a value of zero means the factor cannot contribute
gradient at all.

On the base weights of Llama-3-8B we measure C, D and M at or below 1e-4 against
around 9e-3 for Coex. That is what the paper means when it says three of the four
factors are inert and the whole effect is carried by degree variance. It is a
claim about that construction on those weights.

## Why we are asking

We have run it on one model. Whether the reduction to degree variance is general
or specific to what we measured is the single thing that would most change what
the paper can say, and we cannot settle it alone.

**A report that the factors are LIVE on your model is more useful to us than one
that confirms they are not.** It would mean the reduction is specific rather
than general, and we would rather know. The script says so in its own output
rather than steering you towards our result.

**[Report what you get →](../../issues/new?template=replication.yml)**

Attach the `check_M.json` the script writes. That is the whole report; the form
asks for little else.

## Three things worth knowing before you read a number

**The construction branches on matrix shape.** The reference implementation
builds the pseudo-adjacency as `sigmoid(|WW'|)` for non-square W and as
`sigmoid(|W|)` for square W, so square and non-square modules are not measured
under the same construction. The script prints which branch it used per module
and you should keep the two apart when comparing.

**It subsamples.** The full spectrum of a 4096x4096 matrix costs minutes, so the
script takes 1024 rows. The saturation of the logistic is a per-entry effect
rather than one of scale, so this does not change the reading, but the number is
printed so you know what it is.

**Elasticity is measured along one random direction**, with a fixed seed. It is
a local quantity. A factor with elasticity 1e-4 is not proved dead in every
direction; it is dead in the direction tested, on those weights, which is what
matters for whether it can contribute gradient during a run.

## If it does not run

If the script cannot find the attention projections in your checkpoint, it will
print the tensor names it did find. Open a
[question issue](../../issues/new?template=question.yml) with those names and we
will add the naming pattern. That failure is a bug on our side, not on yours.
