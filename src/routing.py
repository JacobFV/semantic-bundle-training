"""Gradient routing system - routes different loss types through different bundles."""

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import torch
import torch.nn as nn


class LossType(Enum):
    """Types of losses that can be routed through bundles."""

    # Self-supervised / autoregressive
    LM_LOSS = "lm_loss"

    # RL components (TRPO/PPO decomposition)
    POLICY_LOSS = "policy_loss"
    VALUE_LOSS = "value_loss"
    ENTROPY_LOSS = "entropy_loss"
    KL_LOSS = "kl_loss"
    ADVANTAGE_LOSS = "advantage_loss"

    # Error correction
    CONSISTENCY_LOSS = "consistency_loss"
    CALIBRATION_LOSS = "calibration_loss"
    FACTUAL_LOSS = "factual_loss"
    LOGICAL_LOSS = "logical_loss"

    # Multi-agent / social
    CURIOSITY_LOSS = "curiosity_loss"
    COOPERATION_LOSS = "cooperation_loss"


@dataclass
class GradientStats:
    """Statistics about gradients for monitoring."""

    bundle_name: str
    loss_type: str
    grad_norm: float
    grad_mean: float
    grad_std: float
    num_params: int


class GradientRouter:
    """
    Routes gradients from different losses through appropriate bundles.

    Uses hooks to mask gradients in bundle subspaces based on which
    loss type is currently being backpropagated.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.current_loss_type: Optional[str] = None
        self.hooks: List[Any] = []
        self.gradient_stats: List[GradientStats] = []
        self._hooks_registered = False

        # Build mapping from bundle name to its gradient sources
        self._bundle_sources: Dict[str, Set[str]] = {}
        self._extract_bundle_sources()

    def _extract_bundle_sources(self):
        """Extract gradient sources from bundle specs in the model."""
        if not hasattr(self.model, "bundle_layers"):
            return

        for bundle_layer in self.model.bundle_layers:
            for name, bundle in bundle_layer.bundles.items():
                if name not in self._bundle_sources:
                    self._bundle_sources[name] = set(bundle.gradient_sources)

    def register_hooks(self):
        """Register gradient hooks on all bundle parameters."""
        if self._hooks_registered:
            return

        if not hasattr(self.model, "bundle_layers"):
            return

        for bundle_layer in self.model.bundle_layers:
            for name, bundle in bundle_layer.bundles.items():
                for param_name, param in bundle.named_parameters():
                    hook = param.register_hook(
                        self._make_gradient_hook(name, bundle.gradient_sources)
                    )
                    self.hooks.append(hook)

        self._hooks_registered = True

    def _make_gradient_hook(
        self, bundle_name: str, gradient_sources: Set[str]
    ) -> Callable:
        """Create a hook that masks gradients if loss type not in bundle's sources."""

        def hook(grad: torch.Tensor) -> torch.Tensor:
            if self.current_loss_type is None:
                return grad

            if "all" in gradient_sources:
                return grad

            if self.current_loss_type in gradient_sources:
                return grad

            # Zero out gradients for this bundle from this loss
            return torch.zeros_like(grad)

        return hook

    def remove_hooks(self):
        """Remove all registered gradient hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._hooks_registered = False

    def set_loss_type(self, loss_type: Optional[str]):
        """Set which loss is currently being backpropagated."""
        self.current_loss_type = loss_type

    @contextmanager
    def route_gradients(self, loss_type: str):
        """Context manager for routing gradients from a specific loss type."""
        prev_loss_type = self.current_loss_type
        self.set_loss_type(loss_type)
        try:
            yield
        finally:
            self.set_loss_type(prev_loss_type)

    def get_bundle_gradient_mask(self, loss_type: str) -> Dict[str, bool]:
        """Get which bundles receive gradients for a given loss type."""
        mask = {}
        for bundle_name, sources in self._bundle_sources.items():
            if "all" in sources or loss_type in sources:
                mask[bundle_name] = True
            else:
                mask[bundle_name] = False
        return mask

    def collect_gradient_stats(self, loss_type: str) -> List[GradientStats]:
        """Collect gradient statistics per bundle for the current loss."""
        stats = []

        if not hasattr(self.model, "bundle_layers"):
            return stats

        for bundle_layer in self.model.bundle_layers:
            for name, bundle in bundle_layer.bundles.items():
                grad_norms = []
                grad_means = []
                grad_stds = []
                num_params = 0

                for param in bundle.parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
                        grad_means.append(param.grad.mean().item())
                        grad_stds.append(param.grad.std().item())
                        num_params += param.numel()

                if grad_norms:
                    stats.append(
                        GradientStats(
                            bundle_name=name,
                            loss_type=loss_type,
                            grad_norm=sum(g**2 for g in grad_norms) ** 0.5,
                            grad_mean=sum(grad_means) / len(grad_means),
                            grad_std=sum(grad_stds) / len(grad_stds),
                            num_params=num_params,
                        )
                    )

        return stats


class MultiLossComputer:
    """
    Computes multiple losses with proper gradient routing.

    Each loss type only updates its designated bundles based on
    the routing configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        gradient_router: GradientRouter,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        self.model = model
        self.router = gradient_router
        self.loss_weights = loss_weights or {
            "lm_loss": 1.0,
            "policy_loss": 0.1,
            "value_loss": 0.5,
            "entropy_loss": 0.01,
            "kl_loss": 0.1,
        }

    def compute_lm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute language modeling loss."""
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
            ignore_index=-100,
        )
        return loss

    def compute_policy_loss(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: Optional[torch.Tensor] = None,
        clip_epsilon: float = 0.2,
    ) -> torch.Tensor:
        """Compute PPO-style policy loss."""
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        if old_log_probs is not None:
            # PPO clipped objective
            ratio = torch.exp(action_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            policy_loss = -torch.min(
                ratio * advantages, clipped_ratio * advantages
            ).mean()
        else:
            # Simple policy gradient
            policy_loss = -(action_log_probs * advantages).mean()

        return policy_loss

    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
        clip_epsilon: float = 0.2,
    ) -> torch.Tensor:
        """Compute value function loss."""
        if old_values is not None:
            # Clipped value loss
            value_clipped = old_values + torch.clamp(
                values - old_values, -clip_epsilon, clip_epsilon
            )
            value_loss_unclipped = (values - returns) ** 2
            value_loss_clipped = (value_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        else:
            value_loss = 0.5 * ((values - returns) ** 2).mean()

        return value_loss

    def compute_entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy bonus (negative for maximization)."""
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(-1).mean()
        return -entropy  # Negative because we want to maximize entropy

    def compute_kl_loss(
        self,
        logits: torch.Tensor,
        ref_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence from reference policy."""
        kl = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits, dim=-1),
            torch.nn.functional.softmax(ref_logits, dim=-1),
            reduction="batchmean",
        )
        return kl

    def compute_all_losses(
        self,
        batch: Dict[str, torch.Tensor],
        logits: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        ref_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all applicable losses for a batch."""
        losses = {}

        # LM loss (always computed if labels present)
        if "labels" in batch or "input_ids" in batch:
            labels = batch.get("labels", batch["input_ids"])
            losses["lm_loss"] = self.compute_lm_loss(logits, labels)

        # RL losses (if RL data present)
        if "advantages" in batch and "actions" in batch:
            losses["policy_loss"] = self.compute_policy_loss(
                logits,
                batch["actions"],
                batch["advantages"],
                batch.get("old_log_probs"),
            )

        if values is not None and "returns" in batch:
            losses["value_loss"] = self.compute_value_loss(
                values,
                batch["returns"],
                batch.get("old_values"),
            )

        if "advantages" in batch:  # RL mode
            losses["entropy_loss"] = self.compute_entropy_loss(logits)

        if ref_logits is not None:
            losses["kl_loss"] = self.compute_kl_loss(logits, ref_logits)

        return losses

    def backward_with_routing(
        self,
        losses: Dict[str, torch.Tensor],
        retain_graph: bool = True,
    ) -> Dict[str, float]:
        """
        Backpropagate all losses with proper gradient routing.

        Each loss only updates its designated bundles.
        """
        loss_values = {}

        for loss_name, loss in losses.items():
            weight = self.loss_weights.get(loss_name, 0.0)
            if weight <= 0:
                continue

            loss_values[loss_name] = loss.item()

            # Set routing for this loss type
            with self.router.route_gradients(loss_name):
                # Backward pass (gradients will be masked by hooks)
                (loss * weight).backward(retain_graph=retain_graph)

        return loss_values


def create_gradient_router(model: nn.Module) -> GradientRouter:
    """Create and initialize a gradient router for the model."""
    router = GradientRouter(model)
    router.register_hooks()
    return router
