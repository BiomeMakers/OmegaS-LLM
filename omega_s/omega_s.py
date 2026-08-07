# =============================================================================
# Omega-S: A Functional Resilience Index for Catastrophic Forgetting
# =============================================================================
# Author:  Alberto Acedo
# License: AGPL-3.0 (research use). Commercial use requires a separate license.
#          See COMMERCIAL-LICENSE.md in the repository root.
# Patent:  USPTO Patent Pending
#
# Citation:
#   Acedo, A. (2026). Omega-S: A Functional Resilience Index for LLM Fine-Tuning
#   Models via Hutchinson Trace Estimation on Weight Adjacency Matrices.
#   https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

import os
import torch
import torch.nn as nn

class StochasticOmegaS(nn.Module):
    """
    O(N^2) Stochastic regularizer (topological objective, Tr(A^3)) via Hutchinson Trace Estimation
    and Power Iteration for Spectral Modularity.
    """
    def __init__(self, num_samples=3, epsilon=1e-6):
        super().__init__()
        self.num_samples = num_samples
        self.epsilon = epsilon

    def forward(self, W):
        original_dtype = W.dtype
        W = W.to(torch.float32)

        # Pseudo-adjacency construction
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

        # Hutchinson Trace Estimation for Clustering
        tr_A3_est = 0.0
        for _ in range(self.num_samples):
            z = torch.randint(0, 2, (N, 1), dtype=A.dtype, device=A.device) * 2.0 - 1.0
            tr_A3_est += torch.matmul(z.t(), torch.matmul(A, torch.matmul(A, torch.matmul(A, z)))).squeeze()
        tr_A3_est = tr_A3_est / self.num_samples
        C = tr_A3_est / (torch.norm(A, p='fro')**3 + self.epsilon) + self.epsilon

        # Power Iteration for Modularity
        v = torch.randn((N, 1), dtype=A.dtype, device=A.device)
        v = v - torch.mean(v)
        v = v / (torch.norm(v) + self.epsilon)
        max_deg = torch.max(degrees)
        for _ in range(3):
            Lv = (degrees.view(-1, 1) * v) - torch.matmul(A, v)
            v = ((2 * max_deg) * v - Lv)
            v = (v - torch.mean(v)) / (torch.norm(v) + self.epsilon)
        M_est = torch.abs(torch.matmul(v.t(), (degrees.view(-1, 1) * v) - torch.matmul(A, v)).squeeze()) + self.epsilon
M_est = torch.abs(torch.matmul(v.t(), (degrees.view(-1, 1) * v) - torch.matmul(A, v)).squeeze()) + self.epsilon

        # M = 1/lambda_2 es la orientacion que define el marco: la iteracion de
        # potencia estima lambda_2, que es la cantidad INVERSA. Con la variable a
        # 1 se corre la orientacion de results/minv_10seeds.json (retencion
        # 0.8408); a 0, la de results/merged_10seeds.json (0.766).
        if os.environ.get("OMEGA_M_INV", "0") == "1":
            M_est = 1.0 / (M_est + self.epsilon)

        # Final Omega Penalty
        # Final Omega Penalty
        omega_loss = torch.log((M_est * Coex) / (C * D + self.epsilon))
        return omega_loss.to(original_dtype)
