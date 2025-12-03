"""
Smoke test for bundle training on laptop.

Uses small models (~125M params) for quick iteration:
- gpt2 (124M)
- facebook/opt-125m (125M)
- EleutherAI/gpt-neo-125m (125M)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .bundles import wrap_model_with_bundles
from .config import BundleConfig, make_config
from .evaluation import Evaluator, EvalMetrics, compare_configurations, print_comparison_table
from .routing import create_gradient_router, MultiLossComputer


# Small models suitable for laptop testing
SMOKE_TEST_MODELS = [
    "gpt2",  # 124M params
    # "facebook/opt-125m",  # 125M params (optional)
]

# Configurations to test
SMOKE_TEST_CONFIGS = [
    BundleConfig.BASELINE,
    BundleConfig.SINGLE_ADAPTER,
    BundleConfig.SSL_RL_SEPARATE,
    BundleConfig.FULL_MODULAR,
]


class SyntheticDataset(Dataset):
    """Synthetic dataset for smoke testing."""

    def __init__(self, tokenizer, num_samples: int = 100, seq_length: int = 128):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_length = seq_length

        # Generate some synthetic text
        self.texts = [
            self._generate_text(i) for i in range(num_samples)
        ]

        # Tokenize
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            max_length=seq_length,
            padding="max_length",
            return_tensors="pt",
        )

    def _generate_text(self, idx: int) -> str:
        """Generate synthetic text for training."""
        templates = [
            "The quick brown fox jumps over the lazy dog. ",
            "Machine learning models can learn complex patterns. ",
            "Python is a popular programming language for data science. ",
            "Neural networks consist of layers of interconnected nodes. ",
            "The transformer architecture uses attention mechanisms. ",
        ]
        # Repeat and vary based on index
        template = templates[idx % len(templates)]
        return template * (self.seq_length // len(template) + 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }


def run_smoke_test_single(
    model_name: str,
    config: BundleConfig,
    num_steps: int = 50,
    batch_size: int = 2,
    seq_length: int = 128,
    device: str = "auto",
) -> Dict:
    """Run smoke test for a single model + config combination."""

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)

    print(f"\n{'='*60}")
    print(f"Testing: {model_name} + {config.value}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    start_time = time.time()

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Get bundle specs
    bundle_specs = make_config(config, base_model.config.hidden_size)

    print(f"  Base model params: {sum(p.numel() for p in base_model.parameters()):,}")
    print(f"  Bundles: {list(bundle_specs.keys()) if bundle_specs else 'None'}")

    # Wrap with bundles (if not baseline)
    if config == BundleConfig.BASELINE:
        model = base_model.to(device)
        trainable_params = list(model.parameters())
        gradient_router = None
    else:
        model = wrap_model_with_bundles(
            base_model=base_model,
            bundle_specs=bundle_specs,
            freeze_base=True,
        )
        model = model.to(device)
        trainable_params = model.get_trainable_parameters()
        gradient_router = create_gradient_router(model)

        print(f"  Trainable params: {model.num_trainable_params():,}")

    # Create synthetic dataset
    print("Creating synthetic dataset...")
    dataset = SyntheticDataset(tokenizer, num_samples=num_steps * batch_size, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    # Training loop
    print(f"Running {num_steps} training steps...")
    losses = []
    model.train()

    for step, batch in enumerate(tqdm(dataloader, total=num_steps)):
        if step >= num_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        # Forward
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits

        # Compute loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["input_ids"][..., 1:].contiguous()

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Backward (with routing if applicable)
        if gradient_router and config != BundleConfig.BASELINE:
            with gradient_router.route_gradients("lm_loss"):
                loss.backward()
        else:
            loss.backward()

        optimizer.step()
        losses.append(loss.item())

    # Compute stats
    avg_loss = sum(losses) / len(losses)
    final_loss = losses[-1]
    loss_reduction = (losses[0] - losses[-1]) / losses[0] * 100

    elapsed = time.time() - start_time

    # Get bundle stats if applicable
    bundle_stats = {}
    if hasattr(model, "get_all_alphas"):
        alphas = model.get_all_alphas()
        # Average alphas across layers
        for layer_idx, layer_alphas in alphas.items():
            for bundle_name, alpha in layer_alphas.items():
                if bundle_name not in bundle_stats:
                    bundle_stats[bundle_name] = []
                bundle_stats[bundle_name].append(alpha)

        bundle_stats = {k: sum(v) / len(v) for k, v in bundle_stats.items()}

    result = {
        "model": model_name,
        "config": config.value,
        "num_steps": num_steps,
        "avg_loss": avg_loss,
        "final_loss": final_loss,
        "loss_reduction_pct": loss_reduction,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "bundle_alphas": bundle_stats,
        "losses": losses,
    }

    print(f"\nResults:")
    print(f"  Avg loss: {avg_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Loss reduction: {loss_reduction:.1f}%")
    print(f"  Time: {elapsed:.1f}s")

    if bundle_stats:
        print(f"  Bundle alphas: {bundle_stats}")

    return result


def run_full_smoke_test(
    models: List[str] = None,
    configs: List[BundleConfig] = None,
    num_steps: int = 50,
    batch_size: int = 2,
    output_dir: str = "./smoke_test_outputs",
):
    """Run smoke test across multiple models and configurations."""

    models = models or SMOKE_TEST_MODELS
    configs = configs or SMOKE_TEST_CONFIGS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    print("\n" + "=" * 80)
    print("SEMANTIC BUNDLE SMOKE TEST")
    print("=" * 80)
    print(f"Models: {models}")
    print(f"Configs: {[c.value for c in configs]}")
    print(f"Steps per run: {num_steps}")
    print("=" * 80)

    for model_name in models:
        for config in configs:
            try:
                result = run_smoke_test_single(
                    model_name=model_name,
                    config=config,
                    num_steps=num_steps,
                    batch_size=batch_size,
                )
                all_results.append(result)
            except Exception as e:
                print(f"ERROR: {model_name} + {config.value} failed: {e}")
                all_results.append({
                    "model": model_name,
                    "config": config.value,
                    "error": str(e),
                })

    # Save results
    results_file = output_path / "smoke_test_results.json"
    with open(results_file, "w") as f:
        # Convert losses list for JSON serialization
        serializable_results = []
        for r in all_results:
            r_copy = r.copy()
            if "losses" in r_copy:
                r_copy["losses"] = r_copy["losses"][:10]  # Keep only first 10 for brevity
            serializable_results.append(r_copy)
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<25} {'Config':<25} {'Final Loss':<12} {'Reduction':<12}")
    print("-" * 80)

    for r in all_results:
        if "error" in r:
            print(f"{r['model']:<25} {r['config']:<25} ERROR: {r['error'][:30]}")
        else:
            print(f"{r['model']:<25} {r['config']:<25} {r['final_loss']:<12.4f} {r['loss_reduction_pct']:<12.1f}%")

    print("=" * 80)

    return all_results


def main():
    """Main entry point for smoke test."""
    parser = argparse.ArgumentParser(description="Smoke test for semantic bundle training")
    parser.add_argument("--model", type=str, default=None, help="Specific model to test (default: all smoke test models)")
    parser.add_argument("--config", type=str, default=None, help="Specific config to test (default: all)")
    parser.add_argument("--steps", type=int, default=50, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="./smoke_test_outputs")
    parser.add_argument("--quick", action="store_true", help="Quick test with minimal steps")
    args = parser.parse_args()

    # Parse model and config
    models = [args.model] if args.model else None
    configs = None
    if args.config:
        try:
            configs = [BundleConfig(args.config)]
        except ValueError:
            print(f"Unknown config: {args.config}")
            print(f"Available: {[c.value for c in BundleConfig]}")
            return

    if args.quick:
        args.steps = 10
        models = models or ["gpt2"]
        configs = configs or [BundleConfig.BASELINE, BundleConfig.SSL_RL_SEPARATE]

    run_full_smoke_test(
        models=models,
        configs=configs,
        num_steps=args.steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
