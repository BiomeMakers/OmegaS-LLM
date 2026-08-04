#!/usr/bin/env bash
# Mide estructura (C,D,M,Coex) tras la tarea A en los brazos none y omega_lib,
# para las 10 semillas, repartido entre las GPUs. ~20 celdas de SOLO fase A.
set -e
ROOT=/workspace/omega-s
OUT=${OUTDIR:-results_structure}
mkdir -p "$OUT"; cd "$ROOT"; export PYTHONPATH="$ROOT"
NG=$(nvidia-smi --list-gpus | wc -l)
echo "GPUs: $NG. Salida: $OUT/"
SEEDS="42 123 456 789 1011 2022 3033 4044 5055 6066"
i=0
for seed in $SEEDS; do
  gpu=$(( i % NG ))
  CUDA_VISIBLE_DEVICES=$gpu OMEGA_ROOT=$ROOT PYTHONPATH=$ROOT \
    nohup python experiments/measure_structure.py --seed $seed --arm both \
      --out $OUT/struct_s${seed}.json > $OUT/struct_s${seed}.log 2>&1 &
  i=$(( i + 1 ))
  sleep 8
  if (( i % NG == 0 )); then wait; fi
done
wait
echo "MEDICION COMPLETA. Ahora:"
echo "  python experiments/analyze_structure.py $OUT results/merged.json"
