# Run it on your model

The placement diagnostic takes minutes, needs no training and no previous-task
data. It passes a few batches of each task through the model, takes the
principal angles between the two gradient subspaces per module, and tells you
where the two tasks overlap least.

```bash
MODEL=your/model TAREA_A=code TAREA_B=prose python experiments/adapter_placement.py
```

We have run it on Llama-3-8B, Mistral-7B and Qwen2.5-7B, and on three task
pairs. `o_proj` comes first and `q_proj` second in all five combinations; the
last two positions swap in two of them. That is not enough to know where the
ordering stops holding, and it is the one thing we would most like others to
check.

**[Report what you get →](../../issues/new?template=replication.yml)**

Negative results are as useful to us as positive ones, and we would rather hear
about them. If the ordering comes out differently on your model, that bounds the
claim and we will say so.

Three things worth knowing before you read a difference:

- The statistic is the **dimension** of overlap, the sum of squared cosines, not
  the leading cosine. The leading cosine saturates at one as soon as two
  subspaces share a single direction, so it cannot tell sharing one from sharing
  eight.
- The reference is a **same-task ceiling**, the overlap between two halves of the
  first task, not a random-subspace null. Two gradients from the same model share
  directions dictated by the model's structure regardless of task, so what
  informs is the drop from that ceiling.
- If you go on to train, **both arms need the same number of adapted modules**,
  or you are comparing capacity and not placement; and both need to end the
  first task at a similar capability, or the retention ratio is inflated by a
  ceiling effect.

And on how many seeds: repeating an identical run in our setting, same seed and
same hardware, moves retention by 0.104 in standard deviation. One seed does not
separate much.
