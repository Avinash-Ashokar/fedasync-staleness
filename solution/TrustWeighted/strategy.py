# TrustWeighted/strategy.py

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg


TensorWeights = List[torch.Tensor]


@dataclass
class ClientUpdate:
    update: TensorWeights
    num_examples: int
    staleness: int
    delta_loss_norm: float
    update_norm: Optional[float] = None
    alignment_cos: Optional[float] = None


class TrustWeightedAsyncRule:
    """Math rule implementing trust-weighted async aggregation on PyTorch tensors."""

    def __init__(
        self,
        eta: float = 1e-2,
        alpha_freshness: float = 0.1,
        beta1_guard: float = 0.01,
        beta2_guard: float = 0.01,
        theta: Optional[np.ndarray] = None,
        momentum_mu: float = 0.9,
        eps: float = 1e-8,
        clip_update_norm: Optional[float] = None,
    ) -> None:
        self.eta = float(eta)
        self.alpha_freshness = float(alpha_freshness)
        self.beta1_guard = float(beta1_guard)
        self.beta2_guard = float(beta2_guard)
        self.theta = theta.astype(float) if theta is not None else None
        self.momentum_mu = float(momentum_mu)
        self.eps = float(eps)
        self.clip_update_norm = clip_update_norm

    def aggregate(
        self,
        w_t: TensorWeights,
        direction: TensorWeights,
        buffer: Sequence[ClientUpdate],
    ) -> Tuple[TensorWeights, TensorWeights]:
        if len(buffer) == 0:
            return w_t, direction

        w_flat, shapes = self._flatten_weights(w_t)
        m_flat, _ = self._flatten_weights(direction)

        total_examples = max(sum(cu.num_examples for cu in buffer), 1)

        raw_weights: List[float] = []
        adj_updates: List[torch.Tensor] = []

        for cu in buffer:
            ui_flat, _ = self._flatten_weights(cu.update)

            proj_flat = self._projection(ui_flat, m_flat)
            orth_flat = ui_flat - proj_flat

            ui_norm = cu.update_norm
            if ui_norm is None:
                ui_norm = self._safe_norm(ui_flat)
            if self.clip_update_norm is not None:
                ui_norm = min(ui_norm, self.clip_update_norm)

            align_cos = cu.alignment_cos
            if align_cos is None:
                align_cos = self._cosine_similarity(ui_flat, m_flat)

            guard_i = self._guard_factor(cu.staleness, ui_norm)
            fresh_i = self._freshness(cu.staleness)
            quality_i = self._quality_score(
                delta_loss_norm=cu.delta_loss_norm,
                update_norm=ui_norm,
                alignment_cos=align_cos,
            )
            fairness_i = cu.num_examples / total_examples

            raw_w_i = fresh_i * quality_i * fairness_i
            raw_weights.append(float(raw_w_i))

            adj_flat = proj_flat + guard_i * orth_flat
            adj_updates.append(adj_flat)

        raw = np.asarray(raw_weights, dtype=float)
        sum_raw = float(raw.sum() + self.eps)
        norm = raw / sum_raw

        agg_update_flat = torch.zeros_like(w_flat)
        for wi, adj_flat in zip(norm, adj_updates):
            agg_update_flat += float(wi) * adj_flat

        w_next_flat = w_flat + self.eta * agg_update_flat
        w_next = self._unflatten_weights(w_next_flat, shapes)

        m_next_flat = self.momentum_mu * m_flat + (1.0 - self.momentum_mu) * agg_update_flat
        m_next = self._unflatten_weights(m_next_flat, shapes)

        return w_next, m_next

    # ===== Helper functions (PyTorch) =====

    def _flatten_weights(
        self, weights: TensorWeights
    ) -> Tuple[torch.Tensor, List[torch.Size]]:
        if not weights:
            return torch.tensor([], dtype=torch.float32), []
        flat_parts: List[torch.Tensor] = []
        shapes: List[torch.Size] = []
        for w in weights:
            shapes.append(w.shape)
            flat_parts.append(w.reshape(-1))
        flat = torch.cat(flat_parts, dim=0)
        return flat, shapes

    def _unflatten_weights(
        self, flat: torch.Tensor, shapes: List[torch.Size]
    ) -> TensorWeights:
        weights: TensorWeights = []
        offset = 0
        for shape in shapes:
            size = int(np.prod(shape))
            chunk = flat[offset : offset + size]
            weights.append(chunk.reshape(shape))
            offset += size
        return weights

    def _safe_norm(self, v: torch.Tensor) -> float:
        return float(torch.linalg.norm(v).item() + self.eps)

    def _projection(self, u: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        denom = float(torch.dot(m, m).item() + self.eps)
        if denom <= self.eps:
            return torch.zeros_like(u)
        num = float(torch.dot(u, m).item())
        scalar = num / denom
        return scalar * m

    def _cosine_similarity(self, u: torch.Tensor, m: torch.Tensor) -> float:
        u_norm = self._safe_norm(u)
        m_norm = self._safe_norm(m)
        denom = u_norm * m_norm + self.eps
        if denom <= self.eps:
            return 0.0
        num = float(torch.dot(u, m).item())
        cos_val = num / denom
        cos_val = max(min(cos_val, 1.0), -1.0)
        return float(cos_val)

    def _guard_factor(self, staleness: int, update_norm: float) -> float:
        denom = 1.0 + self.beta1_guard * float(staleness) + self.beta2_guard * float(
            update_norm
        )
        denom = max(denom, self.eps)
        return float(1.0 / denom)

    def _freshness(self, staleness: int) -> float:
        return float(np.exp(-self.alpha_freshness * float(staleness)))

    def _quality_score(
        self,
        delta_loss_norm: float,
        update_norm: float,
        alignment_cos: float,
    ) -> float:
        if self.theta is None:
            return 1.0
        features = np.array(
            [float(delta_loss_norm), float(update_norm), float(alignment_cos)],
            dtype=float,
        )
        score = float(np.dot(self.theta, features))
        return float(np.exp(score))


class AsyncTrustFedAvg(FedAvg):
    """ServerApp-compatible strategy using TrustWeightedAsyncRule.

    - Inherits from flwr.serverapp.strategy.FedAvg, so it has .start(...)
    - Overrides configure_train to remember the global model (for deltas)
    - Overrides aggregate_train to use TrustWeightedAsyncRule
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rule = TrustWeightedAsyncRule()
        self._current_arrays: Optional[ArrayRecord] = None
        self._direction_state: Optional[OrderedDict[str, torch.Tensor]] = None

    # ---- configure_train: remember global model arrays ----

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        # Save the global model used in this round to compute deltas later
        self._current_arrays = arrays
        return super().configure_train(server_round, arrays, config, grid)

    # ---- aggregate_train: custom trust-weighted aggregation ----

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        if self._current_arrays is None:
            # Fallback to FedAvg behaviour if something is off
            return super().aggregate_train(server_round, replies)

        # Global model at the beginning of this round
        global_state: OrderedDict[str, torch.Tensor] = (
            self._current_arrays.to_torch_state_dict()
        )

        # Initialise direction if needed
        if self._direction_state is None:
            self._direction_state = OrderedDict(
                (k, torch.zeros_like(v)) for k, v in global_state.items()
            )

        direction_state = self._direction_state

        # Build ClientUpdate buffer
        buffer: List[ClientUpdate] = []
        total_examples = 0
        weighted_loss_sum = 0.0

        valid_replies = [msg for msg in replies if msg.has_content()]

        for msg in valid_replies:
            content: RecordDict = msg.content
            arrays: ArrayRecord = content[self.arrayrecord_key]  # default "arrays"
            metrics: MetricRecord = content["metrics"]

            local_state: OrderedDict[str, torch.Tensor] = arrays.to_torch_state_dict()

            # u_i = local - global (per-parameter)
            update_tensors: TensorWeights = [
                local_state[k] - global_state[k] for k in global_state.keys()
            ]

            flat_u = torch.cat([u.reshape(-1) for u in update_tensors])
            flat_m = torch.cat([direction_state[k].reshape(-1) for k in direction_state])

            update_norm = float(torch.linalg.norm(flat_u).item())
            m_norm = float(torch.linalg.norm(flat_m).item()) + 1e-12
            dot = float(torch.dot(flat_u, flat_m).item())
            alignment_cos = dot / (update_norm * m_norm)

            train_loss = float(metrics.get("train_loss", 0.0))
            num_examples = int(metrics.get(self.weighted_by_key, 1))
            base_round = int(metrics.get("base-round", server_round))
            staleness = max(server_round - base_round, 0)

            total_examples += num_examples
            weighted_loss_sum += train_loss * num_examples

            delta_loss_norm = -train_loss  # simple proxy

            cu = ClientUpdate(
                update=update_tensors,
                num_examples=num_examples,
                staleness=staleness,
                delta_loss_norm=delta_loss_norm,
                update_norm=update_norm,
                alignment_cos=alignment_cos,
            )
            buffer.append(cu)

        if not buffer:
            return self._current_arrays, None

        # Prepare tensors for rule
        global_tensors: TensorWeights = [v for v in global_state.values()]
        dir_tensors: TensorWeights = [v for v in direction_state.values()]

        w_next_tensors, m_next_tensors = self.rule.aggregate(
            w_t=global_tensors,
            direction=dir_tensors,
            buffer=buffer,
        )

        # Map tensors back into an OrderedDict
        new_state = OrderedDict(
            (k, t) for (k, _), t in zip(global_state.items(), w_next_tensors)
        )
        new_dir_state = OrderedDict(
            (k, t) for (k, _), t in zip(direction_state.items(), m_next_tensors)
        )

        # Persist direction for next round
        self._direction_state = new_dir_state

        # Create new ArrayRecord for next global model
        new_arrays = ArrayRecord(torch_state_dict=new_state)

        # Aggregate metrics (weighted average of train_loss)
        avg_loss = weighted_loss_sum / max(total_examples, 1)
        agg_metrics = MetricRecord({"train_loss": avg_loss, self.weighted_by_key: total_examples})

        return new_arrays, agg_metrics
