# =============================================================================
# Omega-S: A regularizer (topological objective)
# =============================================================================
# Author:  Alberto Acedo
# License: AGPL-3.0 (research use). Commercial use requires a separate license.
#          See COMMERCIAL-LICENSE.md in the repository root.
# Patent:  USPTO Patent Pending
#
# Citation:
#   Acedo, A. (2026). Omega-S: A Functional Resilience Index for Catastrophic Forgetting in LLMs.
#   Models via Hutchinson Trace Estimation on Weight Adjacency Matrices.
#   https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist

class DistributedOmegaS(nn.Module):
    """
    Omega-S Distributed Core (Omega-SD).
    A synchronous inter-node topological regularizer for multi-GPU clusters.
    Fully compatible with Fully Sharded Data Parallel (FSDP) architectures.
    """
    def __init__(self, num_samples=3, epsilon=1e-6):
        super().__init__()
        self.num_samples = num_samples
        self.epsilon = epsilon
        
        # Verify distributed environment initialization
        self.is_dist = dist.is_available() and dist.is_initialized()
        if self.is_dist:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def forward(self, W_shard):
        """
        Calculates the topological penalty for a local parameter shard.
        Args:
            W_shard (Tensor): The local shard of the weight matrix on the current GPU.
        """
        # 1. Dimensionality reduction for 3D/4D attention tensors
        if W_shard.ndim > 2:
            while W_shard.ndim > 2:
                W_shard = torch.mean(W_shard, dim=0)

        # 2. Local Adjacency/Covariance Matrix construction
        if W_shard.size(0) != W_shard.size(1):
            W_corr = torch.matmul(W_shard, W_shard.t())
        else:
            W_corr = W_shard
            
        A_raw = torch.sigmoid(torch.abs(W_corr))
        A = (A_raw + A_raw.transpose(-2, -1)) / 2
        N = A.size(0)

        # 3. Local Metrics Computation
        D_local = torch.mean(A)
        degrees = torch.sum(A, dim=1)
        Coex_local = torch.var(degrees)

        # 4. Stochastic Hutchinson Estimation (Local Clustering)
        tr_A3_est_local = 0.0
        for _ in range(self.num_samples):
            z = torch.randint(0, 2, (N, 1), dtype=A.dtype, device=A.device) * 2.0 - 1.0
            tr_A3_est_local += torch.matmul(z.t(), torch.matmul(A, torch.matmul(A, torch.matmul(A, z)))).squeeze()
        tr_A3_est_local = tr_A3_est_local / self.num_samples
        
        # 5. Local Spectral Synchronization (Power Iteration)
        v = torch.randn((N, 1), dtype=A.dtype, device=A.device)
        v = v - torch.mean(v)
        v = v / (torch.norm(v) + self.epsilon)
        max_deg = torch.max(degrees)
        for _ in range(3):
            Lv = (degrees.view(-1, 1) * v) - torch.matmul(A, v)
            v = ((2 * max_deg) * v - Lv)
            v = (v - torch.mean(v)) / (torch.norm(v) + self.epsilon)
        M_est_local = torch.abs(torch.matmul(v.t(), (degrees.view(-1, 1) * v) - torch.matmul(A, v)).squeeze())

        # 6. SYNCHRONOUS GLOBAL REDUCTION (Preserving Computational Graph)
        metrics_tensor = torch.stack([D_local, Coex_local, tr_A3_est_local, M_est_local])
        
        if self.is_dist:
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            metrics_tensor = metrics_tensor / self.world_size

        # Global unpacking
        D_global = metrics_tensor[0]
        Coex_global = metrics_tensor[1] + self.epsilon
        tr_A3_global = metrics_tensor[2]
        M_est_global = metrics_tensor[3] + self.epsilon

        # Re-estimation of Global Clustering Coefficient
        C_global = tr_A3_global / (torch.norm(A, p='fro')**3 + self.epsilon) + self.epsilon

        # 7. Equivalent Global Omega Penalty Output
        omega_loss = torch.log((M_est_global * Coex_global) / (C_global * D_global + self.epsilon))
        
        return omega_loss