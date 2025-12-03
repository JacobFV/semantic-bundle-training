"""Training loop and utilities for bundle experiments."""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .bundles import BundledModel, SharedBases, wrap_model_with_bundles
from .config import BASE_POOLS, BundleConfig, BundleSpec, make_config
from .routing import GradientRouter, MultiLossComputer, create_gradient_router


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Model
    model_name: str = "gpt2"  # or "facebook/opt-125m", "EleutherAI/gpt-neo-125m"
    bundle_config: str = "ssl_rl_separate"

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_steps: Optional[int] = None
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Loss weights
    lm_loss_weight: float = 1.0
    policy_loss_weight: float = 0.1
    value_loss_weight: float = 0.5
    entropy_loss_weight: float = 0.01
    kl_loss_weight: float = 0.1

    # Bundle-specific
    alpha_init: float = 0.5
    bottleneck_ratio: float = 0.5
    freeze_base: bool = True

    # Data
    max_seq_length: int = 512
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"

    # Logging
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 500
    output_dir: str = "./outputs"
    experiment_name: str = "bundle_experiment"

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "semantic-bundles"

    # Device
    device: str = "auto"

    def get_loss_weights(self) -> Dict[str, float]:
        return {
            "lm_loss": self.lm_loss_weight,
            "policy_loss": self.policy_loss_weight,
            "value_loss": self.value_loss_weight,
            "entropy_loss": self.entropy_loss_weight,
            "kl_loss": self.kl_loss_weight,
        }


class Trainer:
    """Main trainer for bundle experiments."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._get_device()
        self.global_step = 0
        self.best_loss = float("inf")

        # Initialize components
        self.model: Optional[BundledModel] = None
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[Any] = None
        self.gradient_router: Optional[GradientRouter] = None
        self.loss_computer: Optional[MultiLossComputer] = None
        self.tokenizer: Optional[Any] = None

        # Logging
        self.train_history: List[Dict] = []
        self.eval_history: List[Dict] = []

        # Setup output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_device(self) -> torch.device:
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)

    def setup(self):
        """Initialize model, optimizer, and other components."""
        print(f"Setting up trainer on device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        print(f"Loading base model: {self.config.model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,  # Use float32 for stability
        )

        # Get bundle configuration
        bundle_config = BundleConfig(self.config.bundle_config)
        bundle_specs = make_config(bundle_config, base_model.config.hidden_size)

        print(f"Bundle config: {bundle_config.value}")
        print(f"Number of bundles: {len(bundle_specs)}")
        for name, spec in bundle_specs.items():
            print(f"  - {name}: rank={spec.get_rank(base_model.config.hidden_size)}, "
                  f"sources={spec.gradient_sources}")

        # Handle baseline (no bundles) vs bundled model
        if not bundle_specs:
            # Baseline: use base model directly, don't freeze
            self.model = base_model.to(self.device)
            self._is_baseline = True
            print("Using baseline model (no bundles)")
        else:
            # Wrap with bundles
            self.model = wrap_model_with_bundles(
                base_model=base_model,
                bundle_specs=bundle_specs,
                alpha_init=self.config.alpha_init,
                bottleneck_ratio=self.config.bottleneck_ratio,
                freeze_base=self.config.freeze_base,
            )
            self.model = self.model.to(self.device)
            self._is_baseline = False

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Setup gradient routing
        self.gradient_router = create_gradient_router(self.model)

        # Setup loss computer
        self.loss_computer = MultiLossComputer(
            self.model,
            self.gradient_router,
            self.config.get_loss_weights(),
        )

        # Setup optimizer (only trainable params)
        if self._is_baseline:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            trainable_params = self.model.get_trainable_parameters()

        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Setup scheduler
        total_steps = self.config.max_steps or (
            self.config.num_epochs * 1000  # Placeholder, will be updated
        )
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - self.config.warmup_steps,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.config.warmup_steps],
        )

        # Initialize wandb if requested
        if self.config.use_wandb:
            try:
                import wandb

                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.experiment_name,
                    config=asdict(self.config),
                )
            except ImportError:
                print("wandb not installed, skipping")
                self.config.use_wandb = False

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with gradient accumulation."""
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )
        logits = outputs.logits

        # Get values if we have a value head and RL data
        values = None
        if self.model.value_head is not None and "returns" in batch:
            hidden_states = outputs.hidden_states[-1]
            values = self.model.value_head(hidden_states).squeeze(-1)

        # Compute all losses
        losses = self.loss_computer.compute_all_losses(
            batch=batch,
            logits=logits,
            values=values,
            ref_logits=batch.get("ref_logits"),
        )

        # Backward with gradient routing
        loss_values = self.loss_computer.backward_with_routing(
            losses,
            retain_graph=False,
        )

        return loss_values

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_losses: Dict[str, List[float]] = {}
        num_batches = len(dataloader)

        progress_bar = tqdm(dataloader, desc="Training")
        accumulated_steps = 0

        for batch_idx, batch in enumerate(progress_bar):
            loss_values = self.train_step(batch)

            # Track losses
            for name, value in loss_values.items():
                if name not in epoch_losses:
                    epoch_losses[name] = []
                epoch_losses[name].append(value)

            accumulated_steps += 1

            # Gradient accumulation
            if accumulated_steps >= self.config.gradient_accumulation_steps:
                # Clip gradients
                if self.config.max_grad_norm > 0:
                    if self._is_baseline:
                        params = [p for p in self.model.parameters() if p.requires_grad]
                    else:
                        params = self.model.get_trainable_parameters()
                    torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                accumulated_steps = 0

                # Logging
                if self.global_step % self.config.log_every == 0:
                    self._log_step(loss_values)

                # Evaluation
                if self.global_step % self.config.eval_every == 0:
                    # Would run eval here
                    pass

                # Save checkpoint
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint()

            # Update progress bar
            total_loss = sum(loss_values.values())
            progress_bar.set_postfix({"loss": f"{total_loss:.4f}", "step": self.global_step})

            # Max steps check
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        # Average epoch losses
        avg_losses = {
            name: sum(values) / len(values)
            for name, values in epoch_losses.items()
        }
        return avg_losses

    def _log_step(self, loss_values: Dict[str, float]):
        """Log training step metrics."""
        log_dict = {
            "step": self.global_step,
            "lr": self.scheduler.get_last_lr()[0],
            **{f"loss/{k}": v for k, v in loss_values.items()},
            "loss/total": sum(loss_values.values()),
        }

        # Get bundle stats (only for bundled models)
        if not self._is_baseline and hasattr(self.model, "get_all_alphas"):
            alphas = self.model.get_all_alphas()
            for layer_idx, layer_alphas in alphas.items():
                for bundle_name, alpha in layer_alphas.items():
                    log_dict[f"alpha/L{layer_idx}/{bundle_name}"] = alpha

            # Get gradient norms
            grad_norms = self.model.get_bundle_gradient_norms()
            for bundle_name, norm in grad_norms.items():
                log_dict[f"grad_norm/{bundle_name}"] = norm

        self.train_history.append(log_dict)

        if self.config.use_wandb:
            import wandb
            wandb.log(log_dict, step=self.global_step)

    def save_checkpoint(self, name: Optional[str] = None):
        """Save model checkpoint."""
        checkpoint_name = name or f"checkpoint-{self.global_step}"
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)

        if self._is_baseline:
            # Save full model state for baseline
            checkpoint_state = {
                "model_state": self.model.state_dict(),
                "global_step": self.global_step,
                "config": asdict(self.config),
                "is_baseline": True,
            }
            torch.save(checkpoint_state, checkpoint_dir / "model_weights.pt")
        else:
            # Save bundle weights only (not base model)
            bundle_state = {
                "shared_bases": self.model.shared_bases.state_dict(),
                "bundle_layers": self.model.bundle_layers.state_dict(),
                "global_step": self.global_step,
                "config": asdict(self.config),
                "is_baseline": False,
            }
            torch.save(bundle_state, checkpoint_dir / "bundle_weights.pt")

        # Save training history
        with open(checkpoint_dir / "train_history.json", "w") as f:
            json.dump(self.train_history, f)

        print(f"Saved checkpoint to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        bundle_state = torch.load(checkpoint_dir / "bundle_weights.pt")

        self.model.shared_bases.load_state_dict(bundle_state["shared_bases"])
        self.model.bundle_layers.load_state_dict(bundle_state["bundle_layers"])
        self.global_step = bundle_state["global_step"]

        print(f"Loaded checkpoint from {checkpoint_dir}")

    def train(self, train_dataloader: DataLoader, num_epochs: Optional[int] = None):
        """Full training loop."""
        num_epochs = num_epochs or self.config.num_epochs

        print(f"Starting training for {num_epochs} epochs")
        print(f"Steps per epoch: ~{len(train_dataloader)}")

        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            epoch_losses = self.train_epoch(train_dataloader)

            print(f"Epoch {epoch + 1} losses:")
            for name, value in epoch_losses.items():
                print(f"  {name}: {value:.4f}")

            # Save end of epoch checkpoint
            self.save_checkpoint(f"epoch-{epoch + 1}")

            # Max steps check
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        # Final save
        self.save_checkpoint("final")

        if self.config.use_wandb:
            import wandb
            wandb.finish()


def create_simple_dataloader(
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
    batch_size: int = 4,
    max_length: int = 512,
    num_samples: Optional[int] = None,
) -> DataLoader:
    """Create a simple dataloader for language modeling."""
    from datasets import load_dataset

    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    # Filter empty texts
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    # Tokenize in batches
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Set format for PyTorch
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Create dataloader
    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return dataloader


def main():
    """Main entry point for training."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", help="Base model name")
    parser.add_argument("--config", default="ssl_rl_separate", help="Bundle config name")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model,
        bundle_config=args.config,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
        max_steps=args.max_steps,
        experiment_name=f"{args.model.replace('/', '-')}_{args.config}",
    )

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
