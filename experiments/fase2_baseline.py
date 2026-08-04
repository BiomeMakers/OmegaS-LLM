# =============================================================================
# Omega-S : Reproducibility Script
# Author:  Alberto Acedo (acedo@biomemakers.com)
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

"""
FASE 2 : GPU 0: Baseline LoRA (sin Omega-S)
============================================
Lanzar con: CUDA_VISIBLE_DEVICES=0 python fase2_baseline.py &
"""

import os, time, json, torch, numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import signal

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
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
EPOCHS_A      = 1
EPOCHS_B      = 1
MAX_SAMPLES   = 5000
HUMANEVAL_N   = 164
MAX_NEW_TOK   = 256
USE_OMEGA     = False
LABEL         = "Baseline_LoRA"
OUT_FILE      = "results_baseline.json"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# MODELO
# ---------------------------------------------------------------------------
def load_model():
    print(f"[{LABEL}] Cargando {MODEL_ID}...")
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

# ---------------------------------------------------------------------------
# DATASETS
# ---------------------------------------------------------------------------
def get_loader(tokenizer, domain="code", max_s=MAX_SAMPLES):
    if domain == "code":
        print(f"[{LABEL}] Cargando CodeSearchNet...")
        ds = load_dataset("code_search_net", "python",
                          split="train", trust_remote_code=True)
        ds = ds.select(range(min(max_s, len(ds))))
        def tok_fn(b):
            texts = [f"### Docstring:\n{d}\n### Code:\n{c}"
                     for d, c in zip(b["func_documentation_string"],
                                     b["whole_func_string"])]
            enc = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN,
                            padding="max_length", return_tensors="pt")
            enc["labels"] = enc["input_ids"].clone()
            return enc
        ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    else:
        print(f"[{LABEL}] Cargando OpenWebText...")
        ds = load_dataset("openwebtext", split="train", trust_remote_code=True)
        ds = ds.select(range(min(max_s, len(ds))))
        def tok_fn(b):
            enc = tokenizer(b["text"], truncation=True,
                            max_length=MAX_SEQ_LEN, padding="max_length",
                            return_tensors="pt")
            enc["labels"] = enc["input_ids"].clone()
            return enc
        ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch")
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------------------------------------------------------
# ENTRENAMIENTO
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, tag):
    model.train()
    total, step = 0.0, 0
    for batch in loader:
        ids   = batch["input_ids"].to(DEVICE)
        labs  = batch["labels"].to(DEVICE)
        mask  = batch.get("attention_mask",
                           torch.ones_like(ids)).to(DEVICE)
        loss  = model(input_ids=ids, attention_mask=mask,
                      labels=labs).loss / GRAD_ACCUM
        loss.backward()
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
        total += loss.item() * GRAD_ACCUM; step += 1
        if step % 100 == 0:
            print(f"  [{tag}] step {step}/{len(loader)} loss={total/step:.4f}")
    return total / step

def train_domain(model, loader, epochs, tag):
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    print(f"\n[{LABEL}] Entrenando {tag}...")
    for e in range(epochs):
        loss = train_epoch(model, loader, opt, f"{tag} E{e+1}")
        print(f"  Epoch {e+1}/{epochs} loss={loss:.4f}")
    return model

# ---------------------------------------------------------------------------
# EVALUACIÓN
# ---------------------------------------------------------------------------
def perplexity(model, loader, tag):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            labs = batch["labels"].to(DEVICE)
            mask = batch.get("attention_mask",
                              torch.ones_like(ids)).to(DEVICE)
            total += model(input_ids=ids, attention_mask=mask,
                           labels=labs).loss.item()
            n += 1
    ppl = torch.exp(torch.tensor(total / n)).item()
    print(f"  [{LABEL}] PPL {tag}: {ppl:.2f}")
    return ppl

def humaneval(model, tok, n=HUMANEVAL_N, tag=""):
    print(f"\n[{LABEL}] HumanEval {tag} : {n} problemas...")
    ds = load_dataset("openai_humaneval", split="test",
                      trust_remote_code=True)
    problems = [ds[i] for i in range(min(n, len(ds)))]
    passed, detail = 0, []

    def _timeout(s, f): raise TimeoutError()
    signal.signal(signal.SIGALRM, _timeout)

    for i, p in enumerate(problems):
        gen_in = tok(p["prompt"], return_tensors="pt",
                     truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**gen_in, max_new_tokens=MAX_NEW_TOK,
                                 temperature=0.1, do_sample=True,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][gen_in["input_ids"].shape[1]:],
                         skip_special_tokens=True)
        full = p["prompt"] + gen + "\n" + p["test"]
        try:
            signal.alarm(10)
            g = {}; exec(compile(full, "<s>", "exec"), g)
            exec(f"check({p['entry_point']})", g)
            signal.alarm(0); passed += 1; status = "PASS"
        except TimeoutError: status = "FAIL:Timeout"
        except Exception as e: status = f"FAIL:{type(e).__name__}"
        detail.append({"i": i, "status": status})
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{n} pass@1={passed/(i+1):.2%}")

    pa1 = passed / n
    print(f"  [{LABEL}] HumanEval {tag} pass@1: {pa1:.4%} ({passed}/{n})")
    return pa1, detail

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{'#'*60}\n{LABEL}\n{'#'*60}")
    model, tok = load_model()

    code_loader  = get_loader(tok, "code", MAX_SAMPLES)
    prose_loader = get_loader(tok, "prose", MAX_SAMPLES)
    eval_code    = get_loader(tok, "code", 300)
    eval_prose   = get_loader(tok, "prose", 300)

    res = {}

    # Paso 0: baseline
    res["he_baseline"], _ = humaneval(model, tok, HUMANEVAL_N, "baseline")
    res["ppl_code_baseline"] = perplexity(model, eval_code, "código-baseline")

    # Paso 1: fine-tune código
    model = train_domain(model, code_loader, EPOCHS_A, "Dominio-A código")
    res["he_post_code"], _ = humaneval(model, tok, HUMANEVAL_N, "post-código")
    res["ppl_code_post_code"] = perplexity(model, eval_code, "código-post-código")

    # Paso 2: fine-tune prosa → medir olvido
    model = train_domain(model, prose_loader, EPOCHS_B, "Dominio-B prosa")
    res["he_post_prose"], he_det = humaneval(model, tok, HUMANEVAL_N, "post-prosa")
    res["ppl_code_post_prose"]  = perplexity(model, eval_code,  "código-post-prosa")
    res["ppl_prose_post_prose"] = perplexity(model, eval_prose, "prosa-post-prosa")

    # Métricas derivadas
    res["forgetting"]    = res["he_post_code"] - res["he_post_prose"]
    res["retention_pct"] = res["he_post_prose"] / max(res["he_post_code"], 1e-6)
    res["label"]         = LABEL
    res["use_omega"]     = USE_OMEGA

    with open(OUT_FILE, "w") as f:
        json.dump({**res, "he_detail": he_det}, f, indent=2)

    print(f"\n[{LABEL}] COMPLETADO → {OUT_FILE}")
    print(f"  HumanEval baseline:   {res['he_baseline']:.4%}")
    print(f"  HumanEval post-code:  {res['he_post_code']:.4%}")
    print(f"  HumanEval post-prose: {res['he_post_prose']:.4%}")
    print(f"  Olvido:               {res['forgetting']:.4%}")
    print(f"  Retención:            {res['retention_pct']:.4%}")
