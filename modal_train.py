"""
Modal deployment for full-scale bundle training experiments.

Run with:
    modal run modal_train.py --config ssl_rl_separate --model gpt2-medium

Or deploy as a scheduled job:
    modal deploy modal_train.py
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

import modal

# Define the Modal app
app = modal.App("semantic-bundle-training")

# Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "accelerate>=0.25.0",
        "wandb>=0.16.0",
        "einops>=0.7.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
    )
    .copy_local_dir("src", "/app/src")
)

# Volume for saving checkpoints and results
volume = modal.Volume.from_name("bundle-training-vol", create_if_missing=True)

# Models to test (larger scale for Modal)
MODAL_MODELS = [
    "gpt2-medium",      # 355M
    "gpt2-large",       # 774M
    "facebook/opt-1.3b", # 1.3B
    # "mistralai/Mistral-7B-v0.1",  # 7B (needs A100)
]

# All configurations to test
MODAL_CONFIGS = [
    "baseline",
    "single_adapter",
    "all_overlap",
    "ssl_rl_separate",
    "ssl_rl_semantic_separate",
    "full_modular",
    "affective_social",
    "feedback_isolated",
    "hierarchical",
    "cognitive_semantic_split",
]


@app.function(
    image=image,
    gpu="A10G",  # A10G for medium models, A100 for 7B+
    timeout=3600 * 4,  # 4 hours
    volumes={"/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret", required=False)],
)
def train_single_config(
    model_name: str,
    config_name: str,
    num_epochs: int = 3,
    batch_size: int = 8,
    max_steps: int = None,
    use_wandb: bool = True,
) -> dict:
    """Train a single model + config combination."""
    import sys
    sys.path.insert(0, "/app")

    import torch
    from src.training import Trainer, TrainingConfig, create_simple_dataloader

    print(f"Training: {model_name} + {config_name}")
    print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

    # Check for wandb secret
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key and use_wandb:
        import wandb
        wandb.login(key=wandb_key)
    else:
        use_wandb = False

    # Create config
    config = TrainingConfig(
        model_name=model_name,
        bundle_config=config_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_steps=max_steps,
        output_dir="/outputs",
        use_wandb=use_wandb,
        experiment_name=f"{model_name.replace('/', '-')}_{config_name}",
    )

    # Setup trainer
    trainer = Trainer(config)
    trainer.setup()

    # Create dataloader
    dataloader = create_simple_dataloader(
        trainer.tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_seq_length,
    )

    # Train
    trainer.train(dataloader)

    # Return summary
    result = {
        "model": model_name,
        "config": config_name,
        "final_step": trainer.global_step,
        "output_dir": str(trainer.output_dir),
    }

    # Commit volume
    volume.commit()

    return result


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 8,  # 8 hours for evaluation
    volumes={"/outputs": volume},
)
def evaluate_all_checkpoints(model_name: str = "gpt2-medium") -> dict:
    """Evaluate all saved checkpoints for a model."""
    import sys
    sys.path.insert(0, "/app")

    from pathlib import Path
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.bundles import wrap_model_with_bundles
    from src.config import BundleConfig, make_config
    from src.evaluation import Evaluator, compare_configurations, print_comparison_table
    from src.training import create_simple_dataloader

    output_dir = Path("/outputs")
    results = {}

    # Find all checkpoints for this model
    model_prefix = model_name.replace("/", "-")

    for config_dir in output_dir.glob(f"{model_prefix}_*"):
        if not config_dir.is_dir():
            continue

        config_name = config_dir.name.replace(f"{model_prefix}_", "")
        checkpoint_dir = config_dir / "final"

        if not (checkpoint_dir / "bundle_weights.pt").exists():
            continue

        print(f"\nEvaluating: {model_name} + {config_name}")

        try:
            # Load model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(model_name)

            if config_name == "baseline":
                model = base_model.cuda()
            else:
                bundle_specs = make_config(
                    BundleConfig(config_name),
                    base_model.config.hidden_size
                )
                model = wrap_model_with_bundles(base_model, bundle_specs)
                model = model.cuda()

                # Load checkpoint
                bundle_state = torch.load(checkpoint_dir / "bundle_weights.pt")
                model.shared_bases.load_state_dict(bundle_state["shared_bases"])
                model.bundle_layers.load_state_dict(bundle_state["bundle_layers"])

            # Create eval dataloader
            eval_dataloader = create_simple_dataloader(
                tokenizer,
                split="validation",
                batch_size=8,
                max_length=512,
            )

            # Evaluate
            evaluator = Evaluator(model, torch.device("cuda"))
            perplexity, loss = evaluator.evaluate_perplexity(eval_dataloader)

            results[config_name] = {
                "perplexity": perplexity,
                "loss": loss,
            }

            print(f"  Perplexity: {perplexity:.2f}")

        except Exception as e:
            print(f"  Error: {e}")
            results[config_name] = {"error": str(e)}

    # Save results
    results_file = output_dir / f"{model_prefix}_evaluation.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    volume.commit()

    return results


@app.function(
    image=image,
    timeout=600,
)
def list_experiments() -> list:
    """List all completed experiments."""
    import sys
    sys.path.insert(0, "/app")

    from pathlib import Path

    output_dir = Path("/outputs")
    experiments = []

    for exp_dir in output_dir.iterdir():
        if exp_dir.is_dir():
            checkpoint_dir = exp_dir / "final"
            if (checkpoint_dir / "bundle_weights.pt").exists():
                experiments.append({
                    "name": exp_dir.name,
                    "has_checkpoint": True,
                })
            else:
                experiments.append({
                    "name": exp_dir.name,
                    "has_checkpoint": False,
                })

    return experiments


@app.local_entrypoint()
def main(
    model: str = "gpt2-medium",
    config: str = "ssl_rl_separate",
    epochs: int = 3,
    batch_size: int = 8,
    max_steps: int = None,
    run_all: bool = False,
    evaluate: bool = False,
    list_only: bool = False,
):
    """
    Entry point for Modal training.

    Examples:
        # Train single config
        modal run modal_train.py --config ssl_rl_separate --model gpt2-medium

        # Train all configs for a model
        modal run modal_train.py --model gpt2-medium --run-all

        # Evaluate all checkpoints
        modal run modal_train.py --model gpt2-medium --evaluate

        # List experiments
        modal run modal_train.py --list-only
    """
    if list_only:
        experiments = list_experiments.remote()
        print("\nCompleted experiments:")
        for exp in experiments:
            status = "✓" if exp["has_checkpoint"] else "○"
            print(f"  {status} {exp['name']}")
        return

    if evaluate:
        results = evaluate_all_checkpoints.remote(model)
        print("\nEvaluation results:")
        for config_name, metrics in results.items():
            if "error" in metrics:
                print(f"  {config_name}: ERROR - {metrics['error']}")
            else:
                print(f"  {config_name}: perplexity={metrics['perplexity']:.2f}")
        return

    if run_all:
        # Run all configs in parallel
        print(f"Running all configs for {model}")
        futures = []
        for cfg in MODAL_CONFIGS:
            print(f"  Spawning: {cfg}")
            futures.append(
                train_single_config.spawn(
                    model_name=model,
                    config_name=cfg,
                    num_epochs=epochs,
                    batch_size=batch_size,
                    max_steps=max_steps,
                )
            )

        # Wait for all to complete
        print("\nWaiting for training jobs...")
        results = []
        for future in futures:
            try:
                result = future.get()
                results.append(result)
                print(f"  Completed: {result['config']}")
            except Exception as e:
                print(f"  Error: {e}")

        print("\nAll training complete!")
        return results

    # Single config training
    result = train_single_config.remote(
        model_name=model,
        config_name=config,
        num_epochs=epochs,
        batch_size=batch_size,
        max_steps=max_steps,
    )
    print(f"\nTraining complete: {result}")
    return result


# Scheduled job for running experiments
@app.function(
    image=image,
    schedule=modal.Cron("0 0 * * 0"),  # Weekly on Sunday midnight
    timeout=3600 * 24,  # 24 hours
    gpu="A10G",
    volumes={"/outputs": volume},
)
def scheduled_experiment_run():
    """Weekly scheduled run of all experiments."""
    import sys
    sys.path.insert(0, "/app")

    results = []
    for model in ["gpt2-medium"]:  # Start with one model
        for config in MODAL_CONFIGS:
            try:
                result = train_single_config.local(
                    model_name=model,
                    config_name=config,
                    num_epochs=2,
                    max_steps=1000,
                )
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "model": model, "config": config})

    return results
