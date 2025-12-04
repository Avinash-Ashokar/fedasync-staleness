# Trust-weighted asynchronous aggregation strategy implementing the PDF math
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class TrustWeightedConfig:
    eta: float = 1.0              # server learning rate η
    eps: float = 1e-8             # numerical stability ε
    freshness_alpha: float = 0.1  # α in s(τ) = exp(-α τ)
    beta1: float = 0.0            # Guard term coefficient on staleness
    beta2: float = 0.0            # Guard term coefficient on ||u||
    momentum_gamma: float = 0.9   # update factor for m_t
    theta: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # quality weights


class TrustWeightedAsyncStrategy:
    """Implements the aggregation rule described in the DML solution PDF.

    Core formula:

        w_{t+1} = w_t + η * Σ_i Weight_i *
            [ Proj_m_t(u_i) + Guard_i * (u_i - Proj_m_t(u_i)) ]

    with:

        Proj_m_t(u_i) = <u_i, m_t> / (||m_t||^2 + eps) * m_t
        Guard_i = 1 / (1 + β1 * τ_i + β2 * ||u_i||)
        Weight_i ∝ s(τ_i) * exp(θᵀ [ΔL̃_i, ||u_i||, cos(u_i, m_t)]) * (n_i / Σ_j n_j)
    """

    def __init__(self, dim: int, cfg: TrustWeightedConfig | None = None) -> None:
        self.dim = int(dim)
        self.cfg = cfg or TrustWeightedConfig()
        self.m = torch.zeros(self.dim, dtype=torch.float32)  # m_t, server momentum
        self.step: int = 0

        self.theta = torch.tensor(self.cfg.theta, dtype=torch.float32)

    # ------------------------------------------------------------------ helpers

    def _proj_m(self, u: torch.Tensor) -> torch.Tensor:
        # Proj_m(u) = <u, m> / (||m||^2 + eps) * m
        num = torch.dot(u, self.m)
        denom = torch.dot(self.m, self.m) + self.cfg.eps
        coef = num / denom
        return coef * self.m

    def _guard(self, tau: torch.Tensor, norm_u: torch.Tensor) -> torch.Tensor:
        # Guard_i = 1 / (1 + β1 τ_i + β2 ||u_i||)
        return 1.0 / (1.0 + self.cfg.beta1 * tau + self.cfg.beta2 * norm_u)

    def _freshness(self, tau: torch.Tensor) -> torch.Tensor:
        # s(τ) = exp(-α τ)
        return torch.exp(-self.cfg.freshness_alpha * tau)

    # ---------------------------------------------------------------- aggregate

    def aggregate(
        self,
        w_t: torch.Tensor,
        updates: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Aggregate a buffer of updates.

        Args:
            w_t: Flattened current global model.
            updates: List of dicts, each containing:
                {
                    "u": update vector (1D tensor),
                    "tau": scalar tensor τ_i,
                    "num_samples": scalar tensor n_i,
                    "delta_loss": scalar tensor ΔL̃_i,
                }

        Returns:
            new_w: updated global model vector.
            metrics: small dict with aggregation statistics.
        """
        if not updates:
            return w_t, {"avg_tau": 0.0, "buffer_size": 0.0}

        self.step += 1
        device = w_t.device
        self.m = self.m.to(device)

        # Collect basic statistics
        taus = torch.stack([u["tau"].to(device) for u in updates])  # [B]
        ns = torch.stack([u["num_samples"].to(device) for u in updates])  # [B]
        delta_losses = torch.stack([u["delta_loss"].to(device) for u in updates])  # [B]
        total_n = ns.sum().clamp_min(1.0)

        # Precompute norms, projections, sideways components, cosines
        proj_list: List[torch.Tensor] = []
        side_list: List[torch.Tensor] = []
        norm_u_list: List[torch.Tensor] = []
        cos_list: List[torch.Tensor] = []

        for u_rec in updates:
            u = u_rec["u"].to(device)
            norm_u = torch.norm(u).clamp_min(self.cfg.eps)
            norm_u_list.append(norm_u)

            proj = self._proj_m(u)
            side = u - proj
            proj_list.append(proj)
            side_list.append(side)

            # cos(u, m) = <u, m> / (||u|| ||m|| + eps)
            norm_m = torch.norm(self.m)
            if norm_m.item() > 0.0:
                cos_val = torch.dot(u, self.m) / (norm_u * norm_m + self.cfg.eps)
            else:
                cos_val = torch.tensor(0.0, device=device)
            cos_list.append(cos_val)

        norm_u_tensor = torch.stack(norm_u_list)  # [B]
        cos_tensor = torch.stack(cos_list)  # [B]

        # Guard factors per update
        guards = self._guard(taus, norm_u_tensor)  # [B]

        # Freshness
        freshness = self._freshness(taus)  # [B]

        # Quality term: exp(θᵀ [ΔL̃_i, ||u_i||, cos(u_i, m_t)])
        feats = torch.stack(
            [delta_losses, norm_u_tensor, cos_tensor],
            dim=1,
        )  # [B, 3]
        quality_logits = feats @ self.theta.to(device)
        quality = torch.exp(quality_logits)

        # Data share term: n_i / Σ_j n_j
        data_share = ns / total_n  # [B]

        # Unnormalized weights, then normalization over buffer
        raw_weights = freshness * quality * data_share  # [B]
        sum_raw = raw_weights.sum()
        if sum_raw.item() <= 0.0:
            weights = torch.full_like(raw_weights, 1.0 / len(updates))
        else:
            weights = raw_weights / sum_raw

        # Combine projection and guarded sideways components
        agg_update = torch.zeros_like(w_t)
        for i in range(len(updates)):
            comp = proj_list[i] + guards[i] * side_list[i]
            agg_update = agg_update + weights[i] * comp

        # Final aggregation step:
        # w_{t+1} = w_t + η * Σ_i Weight_i * [Proj_m(u_i) + Guard_i (u_i - Proj_m(u_i))]
        new_w = w_t + self.cfg.eta * agg_update

        # Update momentum m_t as a running average of aggregated updates
        self.m = (1.0 - self.cfg.momentum_gamma) * self.m + self.cfg.momentum_gamma * agg_update

        metrics = {
            "avg_tau": float(taus.mean().item()),
            "avg_norm_u": float(norm_u_tensor.mean().item()),
            "avg_delta_loss": float(delta_losses.mean().item()),
            "buffer_size": float(len(updates)),
        }
        return new_w, metrics
