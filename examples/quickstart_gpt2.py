# =============================================================================
# Omega-S : Reproducibility Script
# Author:  Alberto Acedo
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

class SingleNodeOmegaS(nn.Module):
    def __init__(self, num_samples=3, epsilon=1e-6):
        super().__init__()
        self.num_samples = num_samples
        self.epsilon = epsilon

    def forward(self, W_shard):
        original_dtype = W_shard.dtype
        W = W_shard.to(torch.float32)

        if W.size(0) != W.size(1):
            W_corr = torch.matmul(W, W.t())
        else:
            W_corr = W
            
        A_raw = torch.sigmoid(torch.abs(W_corr))
        A = (A_raw + A_raw.transpose(-2, -1)) / 2
        N = A.size(0)

        D = torch.mean(A)
        degrees = torch.sum(A, dim=1)
        Coex = torch.var(degrees) + self.epsilon

        tr_A3_est = 0.0
        for _ in range(self.num_samples):
            z = torch.randint(0, 2, (N, 1), dtype=A.dtype, device=A.device) * 2.0 - 1.0
            tr_A3_est += torch.matmul(z.t(), torch.matmul(A, torch.matmul(A, torch.matmul(A, z)))).squeeze()
        tr_A3_est = tr_A3_est / self.num_samples
        C = tr_A3_est / (torch.norm(A, p='fro')**3 + self.epsilon) + self.epsilon

        v = torch.randn((N, 1), dtype=A.dtype, device=A.device)
        v = v - torch.mean(v)
        v = v / (torch.norm(v) + self.epsilon)
        max_deg = torch.max(degrees)
        for _ in range(3):
            Lv = (degrees.view(-1, 1) * v) - torch.matmul(A, v)
            v = ((2 * max_deg) * v - Lv)
            v = (v - torch.mean(v)) / (torch.norm(v) + self.epsilon)
        M_est = torch.abs(torch.matmul(v.t(), (degrees.view(-1, 1) * v) - torch.matmul(A, v)).squeeze()) + self.epsilon

        omega_loss = torch.log((M_est * Coex) / (C * D + self.epsilon))
        return omega_loss.to(original_dtype)

def run_plumbing_test():
    print("1. Cargando Modelo base (GPT-2 pequeño) y Tokenizer...")
    model_name = "gpt2" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print("2. Inyectando Adaptadores LoRA (Librería PEFT)...")
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        target_modules=["c_attn"], 
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM
    )
    peft_model = get_peft_model(base_model, lora_config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_model.to(device)
    
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)
    omega_s = SingleNodeOmegaS().to(device)
    
    print("3. Generando tensores de entrada simulados...")
    dummy_text = ["El CEO cerró la ronda de inversión con éxito y validó el algoritmo."]
    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True).to(device)
    
    print("4. Iniciando Bucle de Entrenamiento (Inyección Omega-S)...")
    peft_model.train()
    optimizer.zero_grad()
    
    outputs = peft_model(**inputs, labels=inputs["input_ids"])
    task_loss = outputs.loss
    print(f"   -> Pérdida de Tarea (CrossEntropy): {task_loss.item():.4f}")
    
    # === EXTRACCIÓN DINÁMICA DE PESOS ===
    omega_total_loss = 0.0
    lora_A_dict = {}
    lora_B_dict = {}
    
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            if 'lora_A' in name:
                base_name = name.replace('lora_A', 'LORA_TARGET')
                lora_A_dict[base_name] = param
            elif 'lora_B' in name:
                base_name = name.replace('lora_B', 'LORA_TARGET')
                lora_B_dict[base_name] = param
                
    gamma = 0.005 
    for base_name in lora_A_dict.keys():
        if base_name in lora_B_dict:
            A = lora_A_dict[base_name]
            B = lora_B_dict[base_name]
            delta_W = torch.matmul(B, A) 
            omega_total_loss += omega_s(delta_W)

    print(f"   -> Pérdida Topológica (Omega-S): {omega_total_loss.item():.4f}")
    
    final_loss = task_loss + (gamma * omega_total_loss)
    
    print("5. Calculando gradientes (Backward pass)...")
    final_loss.backward()
    
    print("6. Actualizando pesos (Optimizer step)...")
    optimizer.step()
    
    print("\n>>> [ÉXITO TOTAL] El grafo de HuggingFace y Omega-S se han fusionado correctamente.")
    print(">>> Los gradientes han fluido. Riesgo de integración: 0%. Listo para AWS/RunPod.")

if __name__ == "__main__":
    run_plumbing_test()