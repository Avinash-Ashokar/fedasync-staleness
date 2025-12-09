from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class TrustWeightedConfig:
    eta: float = 1.0
    eps: float = 1e-8
    freshness_alpha: float = 0.1
    beta1: float = 0.0
    beta2: float = 0.0
    momentum_gamma: float = 0.9
    theta: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class TrustWeightedAsyncStrategy:
    def __init__(self, dim: int, cfg: TrustWeightedConfig | None = None) -> None:
        self.dim = int(dim)
        self.cfg = cfg or TrustWeightedConfig()
        self.m = torch.zeros(self.dim, dtype=torch.float32)
        self.step: int = 0
        self.theta = torch.tensor(self.cfg.theta, dtype=torch.float32)

    def _proj_m(self, u: torch.Tensor) -> torch.Tensor:
        num = torch.dot(u, self.m)
        denom = torch.dot(self.m, self.m) + self.cfg.eps
        coef = num / denom
        return coef * self.m

    def _guard(self, tau: torch.Tensor, norm_u: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + self.cfg.beta1 * tau + self.cfg.beta2 * norm_u)

    def _freshness(self, tau: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.cfg.freshness_alpha * tau)

    def aggregate(
        self,
        w_t: torch.Tensor,
        updates: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not updates:
            return w_t, {"avg_tau": 0.0, "buffer_size": 0.0}

        self.step += 1
        device = w_t.device
        self.m = self.m.to(device)

        taus = torch.stack([u["tau"].to(device) for u in updates])
        ns = torch.stack([u["num_samples"].to(device) for u in updates])
        delta_losses = torch.stack([u["delta_loss"].to(device) for u in updates])
        total_n = ns.sum().clamp_min(1.0)

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

            norm_m = torch.norm(self.m)
            if norm_m.item() > 0.0:
                cos_val = torch.dot(u, self.m) / (norm_u * norm_m + self.cfg.eps)
            else:
                cos_val = torch.tensor(0.0, device=device)
            cos_list.append(cos_val)

        norm_u_tensor = torch.stack(norm_u_list)
        cos_tensor = torch.stack(cos_list)

        guards = self._guard(taus, norm_u_tensor)
        freshness = self._freshness(taus)

        feats = torch.stack(
            [delta_losses, norm_u_tensor, cos_tensor],
            dim=1,
        )
        quality_logits = feats @ self.theta.to(device)
        quality = torch.exp(quality_logits)

        data_share = ns / total_n

        raw_weights = freshness * quality * data_share
        sum_raw = raw_weights.sum()
        if sum_raw.item() <= 0.0:
            weights = torch.full_like(raw_weights, 1.0 / len(updates))
        else:
            weights = raw_weights / sum_raw

        agg_update = torch.zeros_like(w_t)
        for i in range(len(updates)):
            comp = proj_list[i] + guards[i] * side_list[i]
            agg_update = agg_update + weights[i] * comp

        new_w = w_t + self.cfg.eta * agg_update

        self.m = (1.0 - self.cfg.momentum_gamma) * self.m + self.cfg.momentum_gamma * agg_update

        metrics = {
            "avg_tau": float(taus.mean().item()),
            "avg_norm_u": float(norm_u_tensor.mean().item()),
            "avg_delta_loss": float(delta_losses.mean().item()),
            "buffer_size": float(len(updates)),
        }
        return new_w, metrics
