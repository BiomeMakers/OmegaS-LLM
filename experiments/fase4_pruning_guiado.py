# =============================================================================
# Omega-S : Reproducibility Script
# Author:  Alberto Acedo (acedo@biomemakers.com)
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

"""
FASE 4: Pruning Estructurado Guiado por Omega-S en Llama-3-8B
==============================================================
Pipeline:
  1. Entrena Llama-3-8B + LoRA + Omega-S en código (CodeAlpaca)
  2. Calcula mapa topológico por capa (Tr(A³) normalizado)
  3. Mergea LoRA al modelo base
  4. Poda las N cabezas de atención con mayor concentración topológica
  5. Evalúa FLOPs y HumanEval antes y después
  6. Compara contra pruning ALEATORIO (mismo N cabezas)

El argumento: Omega-S no solo regulariza : identifica QUÉ podar.
Pruning guiado por Omega-S > Pruning aleatorio.

Ejecutar: CUDA_VISIBLE_DEVICES=0 python fase4_pruning_guiado.py
"""

import json, torch, numpy as np, signal, copy
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED          = 42
MODEL_ID      = "NousResearch/Meta-Llama-3-8B"
LORA_R        = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.1
LORA_TARGETS  = ["q_proj", "v_proj"]
LR            = 2e-4
BATCH_SIZE    = 2
GRAD_ACCUM    = 4
MAX_SEQ_LEN   = 512
EPOCHS        = 1
MAX_SAMPLES   = 5000
HUMANEVAL_N   = 164
MAX_NEW_TOK   = 256
OMEGA_LAMBDA  = 0.05
OMEGA_EVERY_K = 10
OMEGA_PROBES  = 3

# Pruning: cuántas cabezas eliminar (de 32 totales en Llama-3-8B)
N_HEADS_TO_PRUNE = 4   # ~12.5% de las cabezas
HEAD_DIM         = 128  # dimensión por cabeza en Llama-3-8B
N_HEADS_TOTAL    = 32

OUT_FILE = "results_fase4_pruning.json"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. MODELO Y DATOS
# ---------------------------------------------------------------------------
def load_model_lora():
    print(f"Cargando {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(DEVICE)
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                     r=LORA_R, lora_alpha=LORA_ALPHA,
                     lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS)
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model, tok

def get_loader(tokenizer, max_s=MAX_SAMPLES):
    print("Cargando CodeAlpaca...")
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    ds = ds.select(range(min(max_s, len(ds))))
    def tok_fn(b):
        texts = [f"### Instruction:\n{i}\n### Output:\n{o}"
                 for i, o in zip(b["instruction"], b["output"])]
        enc = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN,
                        padding="max_length", return_tensors="pt")
        enc["labels"] = enc["input_ids"].clone()
        return enc
    ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch")
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------------------------------------------------------
# 2. OMEGA-S
# ---------------------------------------------------------------------------
def hutchinson_tr_a3(W, n=OMEGA_PROBES):
    W_norm = W / (W.norm() + 1e-8)
    total = 0.0
    for _ in range(n):
        v = torch.randint(0, 2, (W_norm.shape[0],),
                          device=W.device, dtype=torch.float32) * 2 - 1
        Wf = W_norm.float()
        z = Wf @ (Wf.t() @ v)
        z = Wf @ (Wf.t() @ z)
        z = Wf @ (Wf.t() @ z)
        total += (v @ z)
    return (total / n).item()

def omega_pen(model):
    total, count = 0.0, 0
    for name, p in model.named_parameters():
        if ("lora_A" in name or "lora_B" in name) and p.requires_grad:
            total += hutchinson_tr_a3(p.data)
            count += p.numel()
    return torch.tensor(OMEGA_LAMBDA * total / max(count, 1),
                        device=DEVICE, requires_grad=False)

# ---------------------------------------------------------------------------
# 3. ENTRENAMIENTO CON OMEGA-S
# ---------------------------------------------------------------------------
def train(model, loader):
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    step = 0
    for epoch in range(EPOCHS):
        total, n = 0.0, 0
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            labs = batch["labels"].to(DEVICE)
            mask = batch.get("attention_mask",
                              torch.ones_like(ids)).to(DEVICE)
            loss = model(input_ids=ids, attention_mask=mask,
                         labels=labs).loss / GRAD_ACCUM
            if step % OMEGA_EVERY_K == 0:
                loss = loss + omega_pen(model) / GRAD_ACCUM
            loss.backward()
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            total += loss.item() * GRAD_ACCUM
            n += 1; step += 1
            if step % 100 == 0:
                print(f"  step {step}/{len(loader)} loss={total/n:.4f}")
        print(f"  Epoch {epoch+1}/{EPOCHS} loss={total/n:.4f}")
    return model

# ---------------------------------------------------------------------------
# 4. MAPA TOPOLÓGICO : qué cabezas tienen mayor concentración
# ---------------------------------------------------------------------------
def compute_topological_map(model):
    """
    Para cada capa de atención, calcula Tr(A³) de q_proj y v_proj
    descompuesto por cabeza (grupos de HEAD_DIM filas).
    Devuelve lista de (layer_idx, head_idx, score) ordenada por score desc.
    """
    print("\nCalculando mapa topologico por cabeza de atencion...")
    head_scores = []

    for name, param in model.named_parameters():
        # Buscar q_proj en capas de atención del modelo base
        if "self_attn.q_proj.weight" in name and not "lora" in name:
            layer_idx = int(name.split(".layers.")[1].split(".")[0])
            W = param.data.float()

            # Dividir la matriz en cabezas (HEAD_DIM filas por cabeza)
            for head_idx in range(N_HEADS_TOTAL):
                start = head_idx * HEAD_DIM
                end   = start + HEAD_DIM
                W_head = W[start:end, :]
                score = hutchinson_tr_a3(W_head, n=OMEGA_PROBES)
                head_scores.append({
                    "layer": layer_idx,
                    "head":  head_idx,
                    "score": score,
                    "param": name,
                })

    # Ordenar por score descendente (mayor score = mayor monopolio)
    head_scores.sort(key=lambda x: x["score"], reverse=True)

    print(f"  Top 5 cabezas con mayor concentracion topologica:")
    for h in head_scores[:5]:
        print(f"    Layer {h['layer']:2d} Head {h['head']:2d} "
              f"score={h['score']:.4f}")

    return head_scores

# ---------------------------------------------------------------------------
# 5. MERGE LORA Y PRUNING ESTRUCTURADO
# ---------------------------------------------------------------------------
def merge_lora(model):
    """Mergea los adaptadores LoRA al modelo base."""
    print("\nMergeando LoRA al modelo base...")
    model = model.merge_and_unload()
    print("  Merge completado.")
    return model

def count_flops(model):
    """Cuenta FLOPs aproximados en capas de atención (q, k, v, o proj)."""
    total = 0
    for name, p in model.named_parameters():
        if any(x in name for x in
               ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]):
            if p.dim() == 2:
                total += 2 * p.shape[0] * p.shape[1]
    return total

def prune_heads(model, heads_to_prune):
    """
    Poda las cabezas especificadas zeroing sus pesos en q_proj y o_proj.
    heads_to_prune: lista de (layer_idx, head_idx)
    
    Nota: zeroing es una approximación de pruning estructurado.
    Para pruning real (reducción de dimensión) se necesita reescribir
    la arquitectura : aquí medimos el impacto potencial via sparsity.
    """
    print(f"\nPodando {len(heads_to_prune)} cabezas...")
    pruned_params = 0

    for layer_idx, head_idx in heads_to_prune:
        # Acceder a la capa de atención
        layer = model.model.layers[layer_idx]

        # Zero q_proj rows (HEAD_DIM filas)
        start = head_idx * HEAD_DIM
        end   = start + HEAD_DIM
        with torch.no_grad():
            layer.self_attn.q_proj.weight[start:end, :] = 0.0
            # Zero correspondientes columnas en o_proj
            layer.self_attn.o_proj.weight[:, start:end] = 0.0
        pruned_params += HEAD_DIM * layer.self_attn.q_proj.weight.shape[1]
        pruned_params += HEAD_DIM * layer.self_attn.o_proj.weight.shape[0]
        print(f"  Podada capa {layer_idx} cabeza {head_idx}")

    return model, pruned_params

def measure_sparsity(model):
    """Mide el porcentaje de pesos a cero en capas de atención."""
    total_zero, total_params = 0, 0
    for name, p in model.named_parameters():
        if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            w = p.data.detach().cpu()
            total_zero   += (w == 0).sum().item()
            total_params += w.numel()
    return total_zero / max(total_params, 1)

# ---------------------------------------------------------------------------
# 6. HUMANEVAL
# ---------------------------------------------------------------------------
def humaneval(model, tok, n=HUMANEVAL_N, tag=""):
    print(f"\nHumanEval {tag} - {n} problemas...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = [ds[i] for i in range(min(n, len(ds)))]
    passed = 0
    def _timeout(s, f): raise TimeoutError()
    signal.signal(signal.SIGALRM, _timeout)
    for i, p in enumerate(problems):
        gen_in = tok(p["prompt"], return_tensors="pt",
                     truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **gen_in, max_new_tokens=MAX_NEW_TOK,
                temperature=0.1, do_sample=True,
                pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][gen_in["input_ids"].shape[1]:],
                         skip_special_tokens=True)
        full = p["prompt"] + gen + "\n" + p["test"]
        try:
            signal.alarm(10)
            g = {}; exec(compile(full, "<s>", "exec"), g)
            exec(f"check({p['entry_point']})", g)
            signal.alarm(0); passed += 1
        except TimeoutError: pass
        except Exception: signal.alarm(0)
    signal.alarm(0)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    pa1 = passed / max(n, 1)
    print(f"  pass@1: {pa1:.4%} ({passed}/{n})")
    return pa1

# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*65)
    print("FASE 4: PRUNING ESTRUCTURADO GUIADO POR OMEGA-S")
    print("Llama-3-8B | LoRA → Merge → Pruning por mapa topologico")
    print("="*65)

    torch.manual_seed(SEED)
    model, tok = load_model_lora()
    loader = get_loader(tok)

    # PASO 1: Entrenar con Omega-S
    print("\n[PASO 1] Entrenando con Omega-S...")
    model = train(model, loader)

    # PASO 2: Mapa topológico (antes del merge, sobre pesos LoRA)
    print("\n[PASO 2] Mapa topologico...")
    head_scores = compute_topological_map(model)

    # PASO 3: Merge LoRA
    print("\n[PASO 3] Merge LoRA...")
    model_merged = merge_lora(model)
    del model; torch.cuda.empty_cache()

    # PASO 4: Evaluar antes del pruning
    print("\n[PASO 4] Evaluando modelo mergeado (sin podar)...")
    flops_before = count_flops(model_merged)
    he_before = humaneval(model_merged, tok, HUMANEVAL_N, "pre-pruning")
    sparsity_before = measure_sparsity(model_merged)
    print(f"  FLOPs: {flops_before:,}")
    print(f"  Sparsity: {sparsity_before:.2%}")

    # PASO 5A: Pruning guiado por Omega-S
    print(f"\n[PASO 5A] Pruning GUIADO por Omega-S "
          f"({N_HEADS_TO_PRUNE} cabezas)...")
    omega_heads = [(h["layer"], h["head"])
                   for h in head_scores[:N_HEADS_TO_PRUNE]]
    model_omega_pruned = copy.deepcopy(model_merged)
    model_omega_pruned, _ = prune_heads(model_omega_pruned, omega_heads)
    he_omega = humaneval(model_omega_pruned, tok, HUMANEVAL_N,
                         "omega-guided pruning")
    sparsity_omega = measure_sparsity(model_omega_pruned)
    flops_omega = flops_before * (1 - sparsity_omega)
    del model_omega_pruned; torch.cuda.empty_cache()

    # PASO 5B: Pruning ALEATORIO (mismas N cabezas, selección random)
    print(f"\n[PASO 5B] Pruning ALEATORIO "
          f"({N_HEADS_TO_PRUNE} cabezas)...")
    torch.manual_seed(SEED + 1)
    all_heads = [(l, h) for l in range(32) for h in range(N_HEADS_TOTAL)]
    random_heads = [all_heads[i] for i in
                    torch.randperm(len(all_heads))[:N_HEADS_TO_PRUNE].tolist()]
    model_random_pruned = copy.deepcopy(model_merged)
    model_random_pruned, _ = prune_heads(model_random_pruned, random_heads)
    he_random = humaneval(model_random_pruned, tok, HUMANEVAL_N,
                          "random pruning")
    sparsity_random = measure_sparsity(model_random_pruned)
    flops_random = flops_before * (1 - sparsity_random)
    del model_random_pruned, model_merged; torch.cuda.empty_cache()

    # TABLA FINAL
    flop_red_omega  = sparsity_omega
    flop_red_random = sparsity_random
    acc_cost_omega  = he_before - he_omega
    acc_cost_random = he_before - he_random

    print("\n\n" + "="*70)
    print("TABLA FINAL : PRUNING GUIADO vs ALEATORIO")
    print("="*70)
    print(f"{'Metodo':<25} {'HumanEval':>10} {'Coste acc':>10} "
          f"{'Sparsity':>10} {'ΔFLOP est.':>12}")
    print("-"*70)
    print(f"{'Sin pruning':<25} {he_before:>10.2%} {':':>10} "
          f"{sparsity_before:>10.2%} {':':>12}")
    print(f"{'Omega-S guiado':<25} {he_omega:>10.2%} "
          f"{-acc_cost_omega:>+10.2%} "
          f"{sparsity_omega:>10.2%} {flop_red_omega:>+12.2%}")
    print(f"{'Aleatorio':<25} {he_random:>10.2%} "
          f"{-acc_cost_random:>+10.2%} "
          f"{sparsity_random:>10.2%} {flop_red_random:>+12.2%}")
    print("="*70)

    ventaja = acc_cost_random - acc_cost_omega
    print(f"\nVentaja Omega-S sobre aleatorio: "
          f"{ventaja:.2%} menos coste de accuracy")

    if ventaja > 0.02:
        print("RESULTADO PUBLICABLE: Omega-S guia mejor el pruning que")
        print("la seleccion aleatoria. Argumento de FLOPs en transformer VALIDADO.")
    elif ventaja > 0:
        print("Señal positiva pero debil. Probar con mas cabezas podadas.")
    else:
        print("Sin ventaja sobre aleatorio en esta configuracion.")
        print("Considerar: mas epocas de entrenamiento Omega-S, "
              "o N_HEADS_TO_PRUNE mayor.")

    results = {
        "config": {
            "model": MODEL_ID,
            "n_heads_pruned": N_HEADS_TO_PRUNE,
            "omega_lambda": OMEGA_LAMBDA,
            "seed": SEED,
        },
        "before_pruning": {
            "humaneval": he_before,
            "flops": flops_before,
            "sparsity": sparsity_before,
        },
        "omega_guided": {
            "humaneval": he_omega,
            "sparsity": sparsity_omega,
            "acc_cost": acc_cost_omega,
            "heads_pruned": omega_heads,
        },
        "random": {
            "humaneval": he_random,
            "sparsity": sparsity_random,
            "acc_cost": acc_cost_random,
            "heads_pruned": random_heads,
        },
        "advantage_omega_vs_random": ventaja,
    }

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados en {OUT_FILE}")
