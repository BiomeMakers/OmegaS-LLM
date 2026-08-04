# =============================================================================
# Omega-S : Reproducibility Script
# Author:  Alberto Acedo
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. CONFIGURACIÓN DEL MODELO Y LoRA
# ==========================================
# Usamos la réplica de NousResearch para evitar el bloqueo de Meta
model_id = "NousResearch/Meta-Llama-3-8B"
print(f"Cargando {model_id} en {device}...")

# Cargamos el modelo base (en bfloat16 para optimizar RAM)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

# Usamos el vocab_size real del modelo (128256)
vocab_size = model.config.vocab_size 

# Configuramos LoRA (Aplicado a las capas de atención)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)

# ==========================================
# 2. REGULARIZADOR OMEGA-S (Optimizado Zero-Overhead)
# ==========================================
def apply_omega_s_lora(model, num_samples=3):
    omega_loss = 0.0
    for name, param in model.named_parameters():
        if ('lora_A.default.weight' in name or 'lora_B.default.weight' in name) and param.requires_grad:
            W = param.to(torch.float32)
            N = W.size(0)

            tr_A3_est = 0.0
            for _ in range(num_samples):
                z = torch.randint(0, 2, (N, 1), dtype=W.dtype, device=W.device) * 2.0 - 1.0
                
                # EL TRUCO DE LA PATENTE: Multiplicación vectorial secuencial
                # Evitamos instanciar la matriz A (4096, 4096) y hacemos W @ (W.t @ z)
                Wt_z = torch.matmul(W.t(), z)         # Tamaño: (8, 1) -> Coste en RAM: Casi 0
                Az = torch.matmul(W, Wt_z)            # Tamaño: (4096, 1)
                
                Wt_Az = torch.matmul(W.t(), Az)
                A2z = torch.matmul(W, Wt_Az)
                
                Wt_A2z = torch.matmul(W.t(), A2z)
                A3z = torch.matmul(W, Wt_A2z)
                
                tr_A3_est += torch.matmul(z.t(), A3z).squeeze()
                
            omega_loss += (tr_A3_est / num_samples) / W.numel()
    return omega_loss * 0.001

def get_lora_variance(model):
    """Mide la varianza estructural de las matrices A y B de LoRA"""
    lora_weights = []
    for name, param in model.named_parameters():
        if 'lora_A.default.weight' in name or 'lora_B.default.weight' in name:
            lora_weights.append(param.data.view(-1))
    
    if len(lora_weights) == 0: return 0.0
    all_w = torch.cat(lora_weights)
    return torch.var(all_w).item()

# ==========================================
# 3. NÚCLEO DE PROFILING Y MEDICIÓN EXACTA
# ==========================================
# Lote de prueba con el tamaño real de vocabulario de Llama-3
dummy_inputs = torch.randint(0, vocab_size, (2, 512), device=device) 

def run_test_scenario(scenario_name, use_omega, omega_every_k=1, steps=10):
    print(f"\nEjecutando {scenario_name}...")
    torch.cuda.empty_cache()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    # Warmup
    for step in range(2):
        loss = model(dummy_inputs, labels=dummy_inputs).loss
        if use_omega and step % omega_every_k == 0: 
            loss += apply_omega_s_lora(model)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Reset de VRAM *después* del warmup, justo antes del profiling real
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    omega_time_total = 0.0

    for step in range(steps):
        loss = model(dummy_inputs, labels=dummy_inputs).loss
        
        # Aplicación con frecuencia K y cronometraje aislado de la patente
        if use_omega and step % omega_every_k == 0:
            torch.cuda.synchronize()
            t0_omega = time.time()
            
            loss += apply_omega_s_lora(model)
            
            torch.cuda.synchronize()
            omega_time_total += (time.time() - t0_omega)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Extracción de métricas
    latency_ms = ((end_time - start_time) / steps) * 1000
    # Promediamos el coste de Omega por iteración
    omega_overhead_ms = (omega_time_total / steps) * 1000 if use_omega else 0.0 
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    variance = get_lora_variance(model)
    
    return [scenario_name, peak_vram_gb, latency_ms, omega_overhead_ms, variance]

# ==========================================
# 4. EJECUCIÓN DEL TEST A/B/C
# ==========================================
results = []
results.append(run_test_scenario("Test A: Baseline (LoRA Normal)", use_omega=False))
results.append(run_test_scenario("Test B: Omega-S Extremo (En cada paso, K=1)", use_omega=True, omega_every_k=1))
results.append(run_test_scenario("Test C: Omega-S Producción (Cada K=10 pasos)", use_omega=True, omega_every_k=10))

# Mostrar resultados
df = pd.DataFrame(results, columns=["Escenario", "Pico VRAM (GB)", "Latencia Total/Paso (ms)", "Costo Específico Omega (ms)", "Varianza Pesos LoRA"])
print("\n" + "="*105)
print(" RESULTADOS TEST 1: VALIDACIÓN ZERO-OVERHEAD, LATENCIA DE PATENTE Y SPARSITY")
print("="*105)
print(df.to_markdown(index=False))