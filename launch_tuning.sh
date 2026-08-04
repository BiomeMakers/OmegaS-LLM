#!/usr/bin/env bash
# =============================================================================
# Fase de AJUSTE: barre el hiperparametro de los cuatro brazos regularizados
# en las 2 semillas de ajuste (42, 123), repartido entre las GPUs.
#
# Por que: sin ajustar todos los brazos con la misma vara, la comparacion
# favorece al que se ajusta. wd ya se barria; ewc estaba fijo en 1000 (rendia
# como no-regularizar); omega solo se calibraba, no se ajustaba. Esto corrige
# los tres. Cada configuracion corre en 42 y 123; la evaluacion final (semillas
# reservadas) es una fase aparte, DESPUES, con los optimos ya fijados.
#
# Uso: bash launch_tuning.sh
# =============================================================================
set -e
ROOT=/workspace/omega-s
OUT=${OUTDIR:-results_tuning}
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT"

NGPU=$(nvidia-smi --list-gpus | wc -l)
echo "GPUs: $NGPU. Salida: $OUT/"

# Lista de trabajos: cada uno es "brazo:valor:semilla". 4 brazos, 2 semillas.
JOBS=()
for SEED in 42 123; do
  for WD in 0.0 0.01 0.05 0.1;        do JOBS+=("wd:$WD:$SEED"); done
  for L  in 100 500 1000 5000 10000;  do JOBS+=("ewc:$L:$SEED"); done
  for T  in 0.03 0.1 0.3 0.5;         do JOBS+=("omega_raw:$T:$SEED"); done
  for T  in 0.03 0.1 0.3 0.5;         do JOBS+=("omega_lib:$T:$SEED"); done
done
echo "Trabajos de ajuste: ${#JOBS[@]}"

run_job () {
  local spec=$1 gpu=$2
  IFS=':' read -r arm val seed <<< "$spec"
  local tag="${arm}_${val}_s${seed}"
  local env="CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$ROOT"
  case $arm in
    wd)  env="$env" ;;
    ewc) env="$env EWC_LAMBDA=$val" ;;
    omega_raw|omega_lib) env="$env OMEGA_TARGET=$val" ;;
  esac
  # una sola celda de esa semilla/brazo; para wd el valor es el propio wd,
  # para el resto el valor va por variable de entorno y wd se fija a 0.05
  local bestwd=0.05
  [ "$arm" = "wd" ] && bestwd=$val
  eval "$env nohup python experiments/rerun_retention.py \
      --cell --seed $seed --arm $arm --best-wd $bestwd \
      --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1 &"
}

# Reparto round-robin sobre las GPUs, en oleadas de NGPU
i=0
for spec in "${JOBS[@]}"; do
  gpu=$(( i % NGPU ))
  run_job "$spec" "$gpu"
  i=$(( i + 1 ))
  # cada vez que llenamos una oleada, esperamos a que acabe antes de la siguiente
  if (( i % NGPU == 0 )); then
    echo "  oleada hasta trabajo $i lanzada, esperando..."
    wait
  fi
done
wait
echo "AJUSTE COMPLETO. Ahora: python experiments/merge_tuning.py $OUT"
