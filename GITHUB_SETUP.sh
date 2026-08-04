#!/bin/bash
# =============================================================================
# INSTRUCCIONES PARA SUBIR EL REPOSITORIO OMEGA-S A GITHUB
# Ejecutar estos comandos en orden desde la carpeta raíz del repositorio
# =============================================================================

# PASO 0: Asegúrate de tener Git instalado y configurado
# git config --global user.name "Alberto Acedo"
# git config --global user.email "tu@email.com"

# PASO 1: Inicializar el repositorio local
git init
git branch -M main

# PASO 2: Añadir todos los archivos
git add .

# PASO 3: Primer commit
git commit -m "feat: initial public release of Omega-S

Omega-S is a drop-in penalty for fine-tuning, computed from the weight matrix
alone: no previous-task data, no Fisher matrix, no stored copy of the old
weights.

Headline result (Llama-3-8B + LoRA, code to prose, HumanEval, ten seeds, every
arm tuned and measured in the same session):
- retains more capability than no regularisation on 9 of 10 seeds
  (0.173 -> 0.238 absolute pass@1; sign test p=0.011, Wilcoxon p=0.006)
- as a retention ratio, 62.9% -> 84.1%
- beats tuned weight decay 10/10 (p=0.002) and tuned EWC 8/10 (p=0.014)
- under 0.4% added cost per training step; +13 MB VRAM on Llama-3-8B

Mechanism reported as measured, not asserted: three of the four factors are
numerically inert and the effect is carried by degree variance. The
contrast-preserving reconstruction was built and made retention worse on all
ten seeds. Negative results are included.

Contents:
- omega_s/: core library (single-node, distributed FSDP, HuggingFace)
- experiments/: every experiment in the paper, including those that failed
- benchmarks/, examples/: profiling and a GPT-2 quickstart
- omega_s_preprint_corto.pdf: the preprint
- LICENSE (AGPL-3.0) and COMMERCIAL-LICENSE.md

Patent: USPTO Pending No. 64/121,656
Author: Alberto Acedo, Biome Makers Inc."

# PASO 4: Conectar con GitHub
# (Primero crea el repositorio en github.com/BiomeMakers/OmegaS-LLM
#  sin inicializarlo : sin README, sin .gitignore, sin licencia)
git remote add origin https://github.com/BiomeMakers/OmegaS-LLM.git

# PASO 5: Subir
git push -u origin main

# =============================================================================
# DESPUÉS DE SUBIR: configurar en GitHub.com
# =============================================================================
# 1. Ve a: github.com/BiomeMakers/OmegaS-LLM → Settings → General
#    - Description: "A drop-in penalty that reduces catastrophic forgetting in
#      LLM fine-tuning, with no previous-task data. USPTO Pending."
#    - Website: (dejar vacío hasta tener arXiv URL)
#    - Topics: deep-learning, regularization, llm, lora, fsdp, topology,
#              hutchinson, catastrophic-forgetting, continual-learning, pytorch,
#              transformers
#
# 2. Ve a: Settings → About → Edit → marca "Releases", "Packages"
#
# 3. Crea el primer Release:
#    - Tag: v0.1.0
#    - Title: "Omega-S v0.1.0 : Initial Release"
#    - Description: pegar el cuerpo del commit de arriba
#    - Adjuntar: omega_s_preprint_corto.pdf
