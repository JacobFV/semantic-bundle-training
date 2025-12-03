"""Evaluation metrics and analysis for bundle experiments."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EvalMetrics:
    """Evaluation metrics for a model."""

    # Language modeling
    perplexity: float = 0.0
    loss: float = 0.0

    # Task-specific (to be filled based on tasks)
    task_metrics: Dict[str, float] = field(default_factory=dict)

    # Bundle analysis
    bundle_activation_means: Dict[str, float] = field(default_factory=dict)
    bundle_activation_stds: Dict[str, float] = field(default_factory=dict)
    bundle_alphas: Dict[str, float] = field(default_factory=dict)

    # Representation analysis
    task_cluster_separation: float = 0.0
    bundle_specialization: Dict[str, float] = field(default_factory=dict)

    # Interference metrics
    forgetting_scores: Dict[str, float] = field(default_factory=dict)


class Evaluator:
    """Evaluator for bundle experiments."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate_perplexity(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Compute perplexity on a dataset."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in tqdm(dataloader, desc="Evaluating perplexity"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            logits = outputs.logits

            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["input_ids"][..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
                ignore_index=-100,
            )

            # Count non-padding tokens
            if "attention_mask" in batch:
                num_tokens = batch["attention_mask"][:, 1:].sum().item()
            else:
                num_tokens = shift_labels.numel()

            total_loss += loss.item()
            total_tokens += num_tokens

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return perplexity, avg_loss

    @torch.no_grad()
    def collect_bundle_activations(
        self,
        dataloader: DataLoader,
        max_samples: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """Collect bundle activations for analysis."""
        self.model.eval()
        activations: Dict[str, List[torch.Tensor]] = {}
        num_samples = 0

        for batch in dataloader:
            if num_samples >= max_samples:
                break

            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                output_hidden_states=True,
            )

            # Get bundle activations from each layer
            if hasattr(self.model, "bundle_layers"):
                hidden_states = outputs.hidden_states

                for layer_idx, bundle_layer in enumerate(self.model.bundle_layers):
                    h = hidden_states[layer_idx + 1]  # +1 for embedding layer
                    layer_activations = bundle_layer.get_bundle_activations(h)

                    for name, act in layer_activations.items():
                        key = f"L{layer_idx}_{name}"
                        if key not in activations:
                            activations[key] = []
                        # Pool over sequence dimension
                        pooled = act.mean(dim=1)  # [batch, rank]
                        activations[key].append(pooled.cpu())

            num_samples += batch["input_ids"].size(0)

        # Concatenate all activations
        result = {}
        for key, acts in activations.items():
            result[key] = torch.cat(acts, dim=0)

        return result

    @torch.no_grad()
    def analyze_bundle_specialization(
        self,
        task_dataloaders: Dict[str, DataLoader],
        max_samples_per_task: int = 50,
    ) -> Dict[str, float]:
        """
        Analyze how specialized each bundle is for different tasks.

        Uses linear probes to classify task type from bundle activations.
        Higher accuracy = more specialized for distinguishing that task.
        """
        self.model.eval()

        # Collect activations per task
        task_activations: Dict[str, Dict[str, torch.Tensor]] = {}
        task_labels: Dict[str, List[int]] = {}

        for task_idx, (task_name, dataloader) in enumerate(task_dataloaders.items()):
            activations = self.collect_bundle_activations(
                dataloader, max_samples=max_samples_per_task
            )
            task_activations[task_name] = activations

            # Create labels for this task
            num_samples = list(activations.values())[0].size(0)
            task_labels[task_name] = [task_idx] * num_samples

        # For each bundle, train a linear probe to classify tasks
        specialization = {}

        if not task_activations:
            return specialization

        bundle_keys = list(list(task_activations.values())[0].keys())

        for bundle_key in bundle_keys:
            # Gather all activations and labels for this bundle
            all_activations = []
            all_labels = []

            for task_name in task_dataloaders.keys():
                if bundle_key in task_activations[task_name]:
                    acts = task_activations[task_name][bundle_key]
                    all_activations.append(acts.numpy())
                    all_labels.extend(task_labels[task_name][:acts.size(0)])

            if len(all_activations) < 2:
                continue

            X = np.concatenate(all_activations, axis=0)
            y = np.array(all_labels)

            # Train linear probe
            try:
                clf = LogisticRegression(max_iter=1000, random_state=42)
                clf.fit(X, y)
                accuracy = clf.score(X, y)
                specialization[bundle_key] = accuracy
            except Exception:
                specialization[bundle_key] = 0.0

        return specialization

    @torch.no_grad()
    def compute_task_cluster_separation(
        self,
        task_dataloaders: Dict[str, DataLoader],
        use_bundle: Optional[str] = None,
        max_samples_per_task: int = 50,
    ) -> float:
        """
        Compute silhouette score for task clustering in activation space.

        Higher score = better separation between tasks.
        """
        self.model.eval()

        all_activations = []
        all_labels = []

        for task_idx, (task_name, dataloader) in enumerate(task_dataloaders.items()):
            activations = self.collect_bundle_activations(
                dataloader, max_samples=max_samples_per_task
            )

            if not activations:
                continue

            if use_bundle and use_bundle in activations:
                # Use specific bundle
                acts = activations[use_bundle]
            else:
                # Use concatenation of all bundles
                acts = torch.cat(list(activations.values()), dim=-1)

            all_activations.append(acts.numpy())
            all_labels.extend([task_idx] * acts.size(0))

        if len(all_activations) < 2:
            return 0.0

        X = np.concatenate(all_activations, axis=0)
        y = np.array(all_labels)

        # Need at least 2 samples per class for silhouette
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2 or min(counts) < 2:
            return 0.0

        try:
            score = silhouette_score(X, y)
            return float(score)
        except Exception:
            return 0.0

    @torch.no_grad()
    def compute_interference_scores(
        self,
        baseline_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute interference/forgetting scores.

        Measures how much performance degraded from baseline.
        Negative = forgetting, Positive = improvement.
        """
        scores = {}
        for task_name in baseline_metrics:
            if task_name in current_metrics:
                baseline = baseline_metrics[task_name]
                current = current_metrics[task_name]
                if baseline > 0:
                    # Relative change
                    scores[task_name] = (current - baseline) / baseline
                else:
                    scores[task_name] = current - baseline
        return scores

    @torch.no_grad()
    def analyze_alpha_distribution(self) -> Dict[str, Dict[str, float]]:
        """Get distribution of routing weights across layers and bundles."""
        if not hasattr(self.model, "bundle_layers"):
            return {}

        result = {}
        for layer_idx, bundle_layer in enumerate(self.model.bundle_layers):
            layer_alphas = bundle_layer.get_alphas()
            result[f"layer_{layer_idx}"] = layer_alphas

        return result

    @torch.no_grad()
    def analyze_subspace_overlap(self) -> Optional[torch.Tensor]:
        """Compute overlap matrix between bundle subspaces."""
        if not hasattr(self.model, "shared_bases"):
            return None

        return self.model.shared_bases.compute_overlap_matrix()

    def full_evaluation(
        self,
        eval_dataloader: DataLoader,
        task_dataloaders: Optional[Dict[str, DataLoader]] = None,
    ) -> EvalMetrics:
        """Run full evaluation suite."""
        metrics = EvalMetrics()

        # Perplexity
        perplexity, loss = self.evaluate_perplexity(eval_dataloader)
        metrics.perplexity = perplexity
        metrics.loss = loss

        # Bundle alphas
        alpha_dist = self.analyze_alpha_distribution()
        # Flatten to single dict
        for layer_name, alphas in alpha_dist.items():
            for bundle_name, alpha in alphas.items():
                metrics.bundle_alphas[f"{layer_name}/{bundle_name}"] = alpha

        # Task-specific evaluation
        if task_dataloaders:
            # Specialization analysis
            metrics.bundle_specialization = self.analyze_bundle_specialization(
                task_dataloaders
            )

            # Cluster separation
            metrics.task_cluster_separation = self.compute_task_cluster_separation(
                task_dataloaders
            )

        return metrics


def compare_configurations(
    results: Dict[str, EvalMetrics],
) -> Dict[str, Any]:
    """Compare evaluation results across different configurations."""
    comparison = {
        "perplexity": {},
        "specialization_mean": {},
        "cluster_separation": {},
    }

    for config_name, metrics in results.items():
        comparison["perplexity"][config_name] = metrics.perplexity
        comparison["cluster_separation"][config_name] = metrics.task_cluster_separation

        if metrics.bundle_specialization:
            spec_values = list(metrics.bundle_specialization.values())
            comparison["specialization_mean"][config_name] = (
                sum(spec_values) / len(spec_values) if spec_values else 0.0
            )

    # Add rankings
    for metric_name in comparison:
        values = comparison[metric_name]
        if values:
            if metric_name == "perplexity":
                # Lower is better
                sorted_configs = sorted(values.items(), key=lambda x: x[1])
            else:
                # Higher is better
                sorted_configs = sorted(values.items(), key=lambda x: -x[1])

            comparison[f"{metric_name}_ranking"] = [c[0] for c in sorted_configs]

    return comparison


def print_comparison_table(comparison: Dict[str, Any]):
    """Print a nicely formatted comparison table."""
    print("\n" + "=" * 80)
    print("CONFIGURATION COMPARISON")
    print("=" * 80)

    # Get all config names
    config_names = list(comparison.get("perplexity", {}).keys())

    if not config_names:
        print("No results to compare")
        return

    # Print header
    print(f"{'Metric':<30}", end="")
    for name in config_names:
        print(f"{name[:15]:<17}", end="")
    print()
    print("-" * 80)

    # Print each metric
    metrics_to_show = ["perplexity", "specialization_mean", "cluster_separation"]

    for metric in metrics_to_show:
        if metric in comparison:
            print(f"{metric:<30}", end="")
            for name in config_names:
                value = comparison[metric].get(name, "N/A")
                if isinstance(value, float):
                    print(f"{value:<17.4f}", end="")
                else:
                    print(f"{str(value):<17}", end="")
            print()

    print("=" * 80)

    # Print rankings
    print("\nRankings (best to worst):")
    for metric in metrics_to_show:
        ranking_key = f"{metric}_ranking"
        if ranking_key in comparison:
            ranking = comparison[ranking_key]
            print(f"  {metric}: {' > '.join(ranking)}")
