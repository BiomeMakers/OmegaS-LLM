# =============================================================================
# Omega-S : Reproducibility Script
# Author:  Alberto Acedo
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd

# ==========================================
# 1. INICIALIZACIÓN DISTRIBUIDA (FSDP)
# ==========================================
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# ==========================================
# 2. CONFIGURACIÓN DEL MODELO Y LoRA
# ==========================================
model_id = "NousResearch/Meta-Llama-3-8B"
if local_rank == 0:
    print(f"Cargando {model_id} en entorno distribuido (World Size: {world_size})...")

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)
vocab_size = model.config.vocab_size

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)
model.to(torch.bfloat16)

# --- PASO A: Envoltura manual de módulos LoRA (NO_SHARD) ---
if local_rank == 0:
    print("Envolviendo módulos LoRA de forma aislada con ShardingStrategy.NO_SHARD...")

for name, module in list(model.named_modules()):
    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        
        wrapped_lora = FSDP(
            module,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16
            ),
            device_id=local_rank,
            use_orig_params=True
        )
        setattr(parent, child_name, wrapped_lora)

# --- PASO B: Envoltura manual de bloques LlamaDecoderLayer (FULL_SHARD) ---
if local_rank == 0:
    print("Envolviendo bloques LlamaDecoderLayer de forma manual con ShardingStrategy.FULL_SHARD...")

transformer_layers = model.base_model.model.model.layers
for i in range(len(transformer_layers)):
    transformer_layers[i] = FSDP(
        transformer_layers[i],
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16
        ),
        device_id=local_rank,
        use_orig_params=True
    )

# --- PASO C: Envoltura del modelo raíz (FULL_SHARD) ---
if local_rank == 0:
    print("Envolviendo modelo raíz...")
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    ),
    device_id=local_rank,
    use_orig_params=True
)

# ==========================================
# 3. REGULARIZADOR OMEGA-S OPTIMIZADO
# ==========================================
def apply_omega_s_lora_fsdp(model, num_samples=3):
    omega_loss = 0.0
    torch.cuda.synchronize()
    t0_omega = time.time()
    
    for name, param in model.named_parameters():
        if ('lora_A' in name or 'lora_B' in name) and param.requires_grad:
            W = param.data.to(torch.float32)
            N = W.size(0)

            tr_A3_est = 0.0
            for _ in range(num_samples):
                z = torch.randint(0, 2, (N, 1), dtype=W.dtype, device=W.device) * 2.0 - 1.0
                
                Wt_z = torch.matmul(W.t(), z)
                Az = torch.matmul(W, Wt_z)
                
                Wt_Az = torch.matmul(W.t(), Az)
                A2z = torch.matmul(W, Wt_Az)
                
                Wt_A2z = torch.matmul(W.t(), A2z)
                A3z = torch.matmul(W, Wt_A2z)
                
                tr_A3_est += torch.matmul(z.t(), A3z).squeeze()
                
            omega_loss += (tr_A3_est / num_samples) / W.numel()
            
    torch.cuda.synchronize()
    omega_compute_time = time.time() - t0_omega
                
    return omega_loss * 0.001, omega_compute_time

def get_lora_variance_fsdp(model):
    lora_weights = []
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_weights.append(param.data.view(-1).clone())
    
    if len(lora_weights) == 0: return 0.0
    all_w = torch.cat(lora_weights)
    return torch.var(all_w).item()

# ==========================================
# 4. NÚCLEO DE PROFILING DISTRIBUIDO
# ==========================================
batch_size = 2
seq_len = 512
dummy_inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
tokens_per_step = batch_size * seq_len

def run_test_scenario_fsdp(scenario_name, use_omega, omega_every_k=1, steps=10):
    if local_rank == 0:
        print(f"Ejecutando {scenario_name}...")
        
    dist.barrier()
    torch.cuda.empty_cache()
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    model.train()
    
    for step in range(2):
        loss = model(dummy_inputs, labels=dummy_inputs).loss
        if use_omega and step % omega_every_k == 0:
            omega_l, _ = apply_omega_s_lora_fsdp(model)
            loss += omega_l
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    total_omega_time = 0.0

    for step in range(steps):
        loss = model(dummy_inputs, labels=dummy_inputs).loss
        
        if use_omega and step % omega_every_k == 0:
            omega_l, o_time = apply_omega_s_lora_fsdp(model)
            loss += omega_l
            total_omega_time += o_time
            
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    dist.barrier()
    torch.cuda.synchronize()
    end_time = time.time()
    
    latency_ms = ((end_time - start_time) / steps) * 1000
    avg_omega_ms = (total_omega_time / steps) * 1000 if use_omega else 0.0
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    variance = get_lora_variance_fsdp(model)
    throughput = tokens_per_step / (latency_ms / 1000)
    
    return [scenario_name, peak_vram_gb, latency_ms, throughput, avg_omega_ms, variance]

results = []
results.append(run_test_scenario_fsdp("Test A: Baseline FSDP", use_omega=False))
results.append(run_test_scenario_fsdp("Test B: Omega-S FSDP (K=1)", use_omega=True, omega_every_k=1))
results.append(run_test_scenario_fsdp("Test C: Omega-S FSDP (K=10)", use_omega=True, omega_every_k=10))

if local_rank == 0:
    df = pd.DataFrame(results, columns=[
        "Escenario", 
        "Pico VRAM/GPU (GB)", 
        "Latencia/Paso (ms)", 
        "Throughput (tokens/s)", 
        "Coste Cómputo Omega (ms)", 
        "Varianza Pesos LoRA"
    ])
    print("\n" + "="*115)
    print(" RESULTADOS TEST 2 CORREGIDOS: OMEGA-S OPTIMIZADO (SIN SUMMON_FULL_PARAMS)")
    print("="*115)
    print(df.to_markdown(index=False))
    df.to_csv("resultados_fsdp_corregidos.csv", index=False)

dist.destroy_process_group()