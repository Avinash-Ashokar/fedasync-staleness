# Trust-weighted asynchronous aggregation strategy with auto-tuning and compression
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
    
    # Auto-tuning parameters
    enable_auto_tune: bool = True
    theta_lr: float = 0.01        # learning rate for θ adaptation
    beta_lr: float = 0.001        # learning rate for β adaptation
    alpha_lr: float = 0.001       # learning rate for α adaptation
    adaptation_window: int = 10    # number of aggregations to consider for adaptation


class TrustWeightedAsyncStrategy:
    """Implements the aggregation rule with auto-tuning and compression support.

    Improvements:
    1. Auto-tuning of θ, β, and staleness thresholds based on performance
    2. Update compression for communication efficiency
    """

    def __init__(self, dim: int, cfg: TrustWeightedConfig | None = None) -> None:
        self.dim = int(dim)
        self.cfg = cfg or TrustWeightedConfig()
        self.m = torch.zeros(self.dim, dtype=torch.float32)  # m_t, server momentum
        self.step: int = 0

        # Initialize θ as learnable parameter for auto-tuning
        self.theta = torch.tensor(self.cfg.theta, dtype=torch.float32, requires_grad=False)
        if self.cfg.enable_auto_tune:
            self.theta = torch.nn.Parameter(torch.tensor(self.cfg.theta, dtype=torch.float32))
        
        # Track performance history for auto-tuning
        self.performance_history: List[Dict] = []
        self.beta1_adaptive = self.cfg.beta1
        self.beta2_adaptive = self.cfg.beta2
        self.alpha_adaptive = self.cfg.freshness_alpha

    # ------------------------------------------------------------------ helpers

    def _proj_m(self, u: torch.Tensor) -> torch.Tensor:
        # Proj_m(u) = <u, m> / (||m||^2 + eps) * m
        num = torch.dot(u, self.m)
        denom = torch.dot(self.m, self.m) + self.cfg.eps
        coef = num / denom
        return coef * self.m

    def _guard(self, tau: torch.Tensor, norm_u: torch.Tensor) -> torch.Tensor:
        # Guard_i = 1 / (1 + β1 τ_i + β2 ||u_i||)
        # Use adaptive β values if auto-tuning is enabled
        beta1 = self.beta1_adaptive if self.cfg.enable_auto_tune else self.cfg.beta1
        beta2 = self.beta2_adaptive if self.cfg.enable_auto_tune else self.cfg.beta2
        return 1.0 / (1.0 + beta1 * tau + beta2 * norm_u)

    def _freshness(self, tau: torch.Tensor) -> torch.Tensor:
        # s(τ) = exp(-α τ)
        # Use adaptive α if auto-tuning is enabled
        alpha = self.alpha_adaptive if self.cfg.enable_auto_tune else self.cfg.freshness_alpha
        return torch.exp(-alpha * tau)

    # ---------------------------------------------------------------- auto-tuning

    def _update_theta(self, quality_scores: torch.Tensor, test_acc_improvement: float) -> None:
        """Adaptively update θ based on quality score correlation with performance."""
        if not self.cfg.enable_auto_tune or len(self.performance_history) < 2:
            return
        
        # Compute gradient: if high quality scores correlate with good performance, increase θ
        # Simple gradient ascent on correlation
        recent_perf = self.performance_history[-self.cfg.adaptation_window:]
        if len(recent_perf) < 2:
            return
        
        # Compute correlation between quality scores and accuracy improvements
        avg_quality = quality_scores.mean().item()
        avg_improvement = sum(p.get('acc_improvement', 0.0) for p in recent_perf) / len(recent_perf)
        
        # Update θ: increase weights if quality correlates with improvement
        if avg_improvement > 0:
            # Positive correlation: increase θ components
            self.theta.data += self.cfg.theta_lr * torch.tensor([0.1, 0.1, 0.1])
        else:
            # Negative correlation: decrease θ components
            self.theta.data -= self.cfg.theta_lr * torch.tensor([0.1, 0.1, 0.1])
        
        # Clamp θ to reasonable bounds
        self.theta.data = torch.clamp(self.theta.data, -2.0, 2.0)

    def _update_beta(self, staleness_impact: float, norm_impact: float) -> None:
        """Adaptively update β1 and β2 based on staleness and norm impact on performance."""
        if not self.cfg.enable_auto_tune:
            return
        
        # If high staleness hurts performance, increase β1
        if staleness_impact < 0:
            self.beta1_adaptive = min(1.0, self.beta1_adaptive + self.cfg.beta_lr)
        else:
            self.beta1_adaptive = max(0.0, self.beta1_adaptive - self.cfg.beta_lr * 0.5)
        
        # If large norms hurt performance, increase β2
        if norm_impact < 0:
            self.beta2_adaptive = min(1.0, self.beta2_adaptive + self.cfg.beta_lr)
        else:
            self.beta2_adaptive = max(0.0, self.beta2_adaptive - self.cfg.beta_lr * 0.5)

    def _update_alpha(self, avg_staleness: float, performance_trend: float) -> None:
        """Adaptively update freshness α based on staleness distribution and performance."""
        if not self.cfg.enable_auto_tune:
            return
        
        # If staleness is high and performance is degrading, increase α (penalize staleness more)
        if avg_staleness > 5.0 and performance_trend < 0:
            self.alpha_adaptive = min(1.0, self.alpha_adaptive + self.cfg.alpha_lr)
        # If staleness is low and performance is good, decrease α (less penalty)
        elif avg_staleness < 2.0 and performance_trend > 0:
            self.alpha_adaptive = max(0.01, self.alpha_adaptive - self.cfg.alpha_lr * 0.5)

    # ---------------------------------------------------------------- compression

    def _compress_update(self, u: torch.Tensor, compression_ratio: float = 0.5) -> Tuple[torch.Tensor, Dict]:
        """Compress update vector using quantization and sparsification.
        
        Args:
            u: Update vector to compress
            compression_ratio: Target compression ratio (0.5 = 50% compression)
        
        Returns:
            Compressed vector and metadata for decompression
        """
        # 1. Quantization: reduce precision to 8-bit
        u_min = u.min()
        u_max = u.max()
        scale = (u_max - u_min) / 255.0 if (u_max - u_min) > 1e-8 else 1.0
        u_quantized = torch.round((u - u_min) / scale).clamp(0, 255).byte()
        
        # 2. Sparsification: keep only top-k values
        k = int(len(u) * compression_ratio)
        if k < len(u):
            _, top_indices = torch.topk(torch.abs(u), k)
            u_sparse = torch.zeros_like(u)
            u_sparse[top_indices] = u[top_indices]
        else:
            u_sparse = u
            top_indices = torch.arange(len(u))
        
        metadata = {
            'min': u_min.item(),
            'scale': scale.item(),
            'indices': top_indices,
            'original_shape': u.shape
        }
        
        return u_quantized, metadata

    def _decompress_update(self, u_compressed: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """Decompress update vector from compressed format."""
        # Dequantize
        u_dequantized = u_compressed.float() * metadata['scale'] + metadata['min']
        
        # Reconstruct sparse vector
        u_reconstructed = torch.zeros(metadata['original_shape'], dtype=torch.float32)
        u_reconstructed[metadata['indices']] = u_dequantized[metadata['indices']]
        
        return u_reconstructed

    # ---------------------------------------------------------------- aggregate

    def aggregate(
        self,
        w_t: torch.Tensor,
        updates: List[Dict[str, torch.Tensor]],
        test_acc_improvement: float = 0.0,
        use_compression: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Aggregate a buffer of updates with auto-tuning and optional compression.

        Args:
            w_t: Flattened current global model.
            updates: List of dicts, each containing:
                {
                    "u": update vector (1D tensor),
                    "tau": scalar tensor τ_i,
                    "num_samples": scalar tensor n_i,
                    "delta_loss": scalar tensor ΔL̃_i,
                }
            test_acc_improvement: Recent test accuracy improvement for auto-tuning
            use_compression: Whether to use compression for updates

        Returns:
            new_w: updated global model vector.
            metrics: small dict with aggregation statistics.
        """
        if not updates:
            return w_t, {"avg_tau": 0.0, "buffer_size": 0.0}

        self.step += 1
        device = w_t.device
        self.m = self.m.to(device)

        # Decompress updates if compressed
        if use_compression:
            for u_rec in updates:
                if 'compressed' in u_rec and u_rec['compressed']:
                    u_rec['u'] = self._decompress_update(u_rec['u'], u_rec['metadata'])

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
        theta_tensor = self.theta.to(device) if isinstance(self.theta, torch.nn.Parameter) else self.theta.to(device)
        quality_logits = feats @ theta_tensor
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

        # Auto-tuning: update parameters based on performance
        if self.cfg.enable_auto_tune:
            avg_staleness = float(taus.mean().item())
            avg_norm = float(norm_u_tensor.mean().item())
            
            # Compute performance trend
            if len(self.performance_history) > 0:
                recent_improvements = [p.get('acc_improvement', 0.0) for p in self.performance_history[-5:]]
                performance_trend = sum(recent_improvements) / len(recent_improvements) if recent_improvements else 0.0
            else:
                performance_trend = 0.0
            
            # Update parameters
            self._update_theta(quality, test_acc_improvement)
            self._update_beta(
                staleness_impact=-avg_staleness * 0.1,  # Simplified: high staleness = negative impact
                norm_impact=-avg_norm * 0.01  # Simplified: very large norms = negative impact
            )
            self._update_alpha(avg_staleness, performance_trend)
            
            # Store performance for next iteration
            self.performance_history.append({
                'acc_improvement': test_acc_improvement,
                'avg_staleness': avg_staleness,
                'avg_norm': avg_norm,
            })
            # Keep only recent history
            if len(self.performance_history) > self.cfg.adaptation_window * 2:
                self.performance_history = self.performance_history[-self.cfg.adaptation_window * 2:]

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
            "theta_0": float(self.theta[0].item() if isinstance(self.theta, torch.nn.Parameter) else self.theta[0]),
            "theta_1": float(self.theta[1].item() if isinstance(self.theta, torch.nn.Parameter) else self.theta[1]),
            "theta_2": float(self.theta[2].item() if isinstance(self.theta, torch.nn.Parameter) else self.theta[2]),
            "beta1_adaptive": float(self.beta1_adaptive),
            "beta2_adaptive": float(self.beta2_adaptive),
            "alpha_adaptive": float(self.alpha_adaptive),
        }
        return new_w, metrics
