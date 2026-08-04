# Figures

## Generated here

`make_figures.py` produces `figure1_retention`, which renders the per-seed
retention results of Section 4.4. It plots all three seeds, including the one on
which Omega-S underperforms weight decay, rather than the mean alone.

```bash
python figures/make_figures.py
```

## Requires a local run

`degree_distribution.py` produces the figure that makes the paper's central
concept visible: the distribution of row norms under baseline, weight decay and
Omega-S. It is not generated automatically because it needs the trained weight
matrices, which are not redistributed here.

```bash
python figures/degree_distribution.py \
    --baseline runs/baseline/adapter_model.safetensors \
    --wd       runs/weight_decay/adapter_model.safetensors \
    --omega    runs/omega_s/adapter_model.safetensors \
    --layer    model.layers.16.self_attn.q_proj
```

Use the same layer for all three runs; a mid-network attention projection is
where the effect reported in Section 4.2 is largest. The script prints the
degree variance of each run, which should reproduce the values in that section.
