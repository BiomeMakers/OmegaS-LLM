# =============================================================================
# Omega-S : Reproducibility Script
# Author:  Alberto Acedo
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd

# =====================================================================
# 1. DEFINICIÓN DEL REGULARIZADOR OMEGA ESTOCÁSTICO (PROYECCIÓN CUADRADA)
# =====================================================================
class StochasticOmegaLoss(nn.Module):
    def __init__(self, num_samples=3, epsilon=1e-6):
        super().__init__()
        self.num_samples = num_samples
        self.epsilon = epsilon

    def forward(self, W):
        # Para matrices rectangulares (como 128x784), proyectamos W * W^T
        # Esto genera una matriz simétrica perfecta de NxN (neurona contra neurona)
        A_raw = torch.matmul(W, W.t())
        A = torch.sigmoid(torch.abs(A_raw))
        N = A.size(0)

        # Densidad (D) y Exclusión Competitiva (Coex)
        D = torch.mean(A)
        degrees = torch.sum(A, dim=1)
        Coex = torch.var(degrees) + self.epsilon

        # Clustering (C) vía Hutchinson Estocástico
        tr_A3_est = 0.0
        for _ in range(self.num_samples):
            z = torch.randint(0, 2, (N, 1), dtype=W.dtype, device=W.device) * 2.0 - 1.0
            Az = torch.matmul(A, z)
            A2z = torch.matmul(A, Az)
            A3z = torch.matmul(A, A2z)
            tr_A3_est += torch.matmul(z.t(), A3z).squeeze()
        tr_A3_est = tr_A3_est / self.num_samples
        C_est = tr_A3_est / (torch.norm(A, p='fro')**3 + self.epsilon) + self.epsilon

        # Modularidad (M) vía Power Iteration corta (3 pasos)
        v = torch.randn((N, 1), dtype=W.dtype, device=W.device)
        v = v - torch.mean(v)
        v = v / torch.norm(v)
        max_deg = torch.max(degrees)
        for _ in range(3):
            Lv = (degrees.view(-1, 1) * v) - torch.matmul(A, v)
            Bv = (2 * max_deg) * v - Lv
            v = Bv - torch.mean(Bv)
            v = v / torch.norm(v)
        M_est = torch.abs(torch.matmul(v.t(), (degrees.view(-1, 1) * v) - torch.matmul(A, v)).squeeze()) + self.epsilon

        return torch.log((M_est * Coex) / (C_est * D + self.epsilon))

# =====================================================================
# 2. ARQUITECTURA DE LA RED NEURONAL
# =====================================================================
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128) 
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =====================================================================
# 3. CARGA DE DATOS REALES (BENCHMARK MNIST)
# =====================================================================
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

criterion = nn.CrossEntropyLoss()

def evaluar_modelo(model):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    return val_loss, accuracy

def extraer_metricas_topologicas(W_tensor):
    with torch.no_grad():
        A_raw = torch.matmul(W_tensor, W_tensor.t())
        A = torch.sigmoid(torch.abs(A_raw))
        degrees = torch.sum(A, dim=1)
        coex_val = torch.var(degrees).item() 
        max_hub = torch.max(degrees).item()  
        return coex_val, max_hub

results = []

# =====================================================================
# 4. ENTRENAMIENTO RED A: BASELINE (TRADICIONAL)
# =====================================================================
print("Entrenando Red A: Baseline...")
model_A = SimpleNet()
optimizer_A = optim.Adam(model_A.parameters(), lr=0.01, weight_decay=0.0)

model_A.train()
for epoch in range(2):
    for data, target in train_loader:
        optimizer_A.zero_grad()
        loss = criterion(model_A(data), target)
        loss.backward()
        optimizer_A.step()

loss_A, acc_A = evaluar_modelo(model_A)
var_A, hub_A = extraer_metricas_topologicas(model_A.fc1.weight)
results.append(["Red A (Baseline)", loss_A, acc_A, var_A, hub_A])

# =====================================================================
# 5. ENTRENAMIENTO RED B: OMEGA-S (TU PATENTE)
# =====================================================================
print("Entrenando Red B: Omega-S Activo...")
model_B = SimpleNet()
optimizer_B = optim.Adam(model_B.parameters(), lr=0.01, weight_decay=0.0)
omega_reg = StochasticOmegaLoss()
gamma = 0.01 

model_B.train()
for epoch in range(2):
    for data, target in train_loader:
        optimizer_B.zero_grad()
        output = model_B(data)
        task_loss = criterion(output, target)
        omega_loss = omega_reg(model_B.fc1.weight)
        total_loss = task_loss + gamma * omega_loss
        total_loss.backward()
        optimizer_B.step()

loss_B, acc_B = evaluar_modelo(model_B)
var_B, hub_B = extraer_metricas_topologicas(model_B.fc1.weight)
results.append(["Red B (Omega-S)", loss_B, acc_B, var_B, hub_B])

# =====================================================================
# 6. ENTRENAMIENTO RED C: CONTROL WEIGHT DECAY (LA PETICIÓN DEL ASESOR)
# =====================================================================
print("Entrenando Red C: Control Weight Decay Fuerte...")
model_C = SimpleNet()
optimizer_C = optim.Adam(model_C.parameters(), lr=0.01, weight_decay=0.01)

model_C.train()
for epoch in range(2):
    for data, target in train_loader:
        optimizer_C.zero_grad()
        loss = criterion(model_C(data), target)
        loss.backward()
        optimizer_C.step()

loss_C, acc_C = evaluar_modelo(model_C)
var_C, hub_C = extraer_metricas_topologicas(model_C.fc1.weight)
results.append(["Red C (Weight Decay 0.01)", loss_C, acc_C, var_C, hub_C])

# =====================================================================
# 7. TABLA DE CONFLICTO DE DATOS FINAL
# =====================================================================
df = pd.DataFrame(results, columns=["Arquitectura", "Eval Loss (↓)", "Accuracy % (↑)", "Varianza Grados (↓)", "Nodo Hub Máx (↓)"])
print("\n" + "="*85)
print(" AUDITORÍA TÉCNICA DEFINITIVA PARA EL JUEVES")
print("="*85)
print(df.to_markdown(index=False))
print("="*85)