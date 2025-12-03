# Process Guide: LLM-Automated ML Experiments

**Purpose:** Learnings from building semantic-bundle-training for streamlining future LLM-automated experiments.

---

## 1. Project Structure That Works

```
project/
├── pyproject.toml          # uv-managed dependencies
├── main.py                 # Entry point with help text
├── modal_train.py          # Cloud deployment (separate from src/)
├── src/
│   ├── __init__.py         # Clean exports
│   ├── config.py           # All configurations, enums, dataclasses
│   ├── model.py            # Model architecture
│   ├── training.py         # Training loop
│   ├── evaluation.py       # Metrics and analysis
│   └── smoke_test.py       # Local validation
└── docs/
    ├── REPORT_*.md         # Experiment reports
    └── EXPERIMENT_*.md     # Planned experiments
```

### Key Principles

1. **Separate config from implementation** - All hyperparameters, experiment variants in one place
2. **Modal script at root level** - Not in src/, avoids import path issues
3. **Smoke test as first-class citizen** - Not an afterthought

---

## 2. Dependency Management with uv

### Setup
```bash
uv init project-name
uv add torch transformers datasets  # Core ML
uv add wandb                         # Logging
uv add modal                         # Cloud
```

### Running
```bash
uv run python -m src.smoke_test      # Run module
uv run modal run modal_train.py      # Modal commands
```

### Why uv over pip/conda
- Fast dependency resolution
- Lockfile for reproducibility
- No virtual env activation needed (`uv run` handles it)
- Clean pyproject.toml (no setup.py, requirements.txt)

---

## 3. Local Smoke Testing Pattern

### Purpose
Catch bugs before burning GPU hours/dollars.

### Implementation
```python
# src/smoke_test.py

def run_smoke_test_single(model_name, config, num_steps=10):
    """
    Minimal test:
    - Small model (gpt2, 124M)
    - Synthetic data (no download)
    - Few steps (10-50)
    - CPU or MPS (no GPU required)
    """
    # 1. Create synthetic dataset (fast, no network)
    dataset = SyntheticDataset(tokenizer, num_samples=100)

    # 2. Initialize model + bundles
    model = wrap_model_with_bundles(base_model, bundle_specs)

    # 3. Run N training steps
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break
        loss = train_step(batch)

    # 4. Return diagnostics
    return {"loss": loss, "grad_norms": get_grad_norms()}
```

### Smoke Test Checklist
- [ ] Imports work
- [ ] Model loads
- [ ] Forward pass succeeds
- [ ] Backward pass succeeds
- [ ] Gradients flow to correct parameters
- [ ] Checkpointing works
- [ ] No CUDA errors (if GPU available)

### Run Before Every Modal Deploy
```bash
uv run python -m src.smoke_test --quick  # 1-2 minutes
```

---

## 4. Modal Deployment Pattern

### Environment Isolation
```bash
# Create isolated test environment
modal environment create test-env

# Run in isolated environment
modal run --env test-env modal_train.py

# Cleanup
modal volume delete my-vol --env test-env -y
modal environment delete test-env -y
```

**Always use test environment for validation runs.**

### Modal Script Structure
```python
import modal

app = modal.App("project-name")

# Image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", ...)
    .add_local_dir("src", "/app/src")  # Copy source code
)

# Persistent storage for checkpoints
volume = modal.Volume.from_name("project-vol", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G",              # or "A100" for larger models
    timeout=3600 * 4,        # 4 hours
    volumes={"/outputs": volume},
)
def train(config: str, model: str, max_steps: int = None):
    import sys
    sys.path.insert(0, "/app")  # Make src/ importable

    from src.training import Trainer
    # ... training code ...

    volume.commit()  # Save outputs
    return results

@app.local_entrypoint()
def main(config: str = "default", model: str = "gpt2"):
    result = train.remote(config=config, model=model)
    print(result)
```

### Modal Best Practices
1. **Test locally first** - `uv run python -m src.smoke_test`
2. **Use test environment** - Never pollute main env during development
3. **Set timeouts** - Prevent runaway costs
4. **Commit volumes** - `volume.commit()` after writing checkpoints
5. **Return results** - Function should return summary dict for logging

### GPU Selection
| Model Size | GPU | Approx Cost |
|------------|-----|-------------|
| <500M | A10G | $0.60/hr |
| 500M-3B | A10G | $0.60/hr |
| 3B-13B | A100-40GB | $2.78/hr |
| 13B+ | A100-80GB | $3.95/hr |

---

## 5. PyTorch Patterns for Experiments

### Gradient Hooks for Routing
```python
class GradientRouter:
    def __init__(self, model):
        self.current_loss_type = None
        self.hooks = []

    def register_hooks(self):
        for name, param in model.named_parameters():
            hook = param.register_hook(self._make_hook(name))
            self.hooks.append(hook)

    def _make_hook(self, param_name):
        def hook(grad):
            if self.should_zero(param_name, self.current_loss_type):
                return torch.zeros_like(grad)
            return grad
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
```

### Multi-Loss Backward
```python
def backward_with_routing(losses: Dict[str, Tensor], router: GradientRouter):
    for loss_name, loss in losses.items():
        router.current_loss_type = loss_name
        loss.backward(retain_graph=True)  # retain_graph for multiple backwards
    router.current_loss_type = None
```

### Handling Baseline vs Bundled Models
```python
# Pattern: check for baseline throughout
if not bundle_specs:
    self.model = base_model
    self._is_baseline = True
else:
    self.model = wrap_with_bundles(base_model, bundle_specs)
    self._is_baseline = False

# Then guard all bundle-specific code:
if not self._is_baseline:
    alphas = self.model.get_all_alphas()
    grad_norms = self.model.get_bundle_gradient_norms()
```

### Checkpoint Compatibility
```python
def save_checkpoint(self):
    state = {
        "global_step": self.global_step,
        "config": asdict(self.config),
        "is_baseline": self._is_baseline,
    }

    if self._is_baseline:
        state["model_state"] = self.model.state_dict()
        filename = "model_weights.pt"
    else:
        state["shared_bases"] = self.model.shared_bases.state_dict()
        state["bundle_layers"] = self.model.bundle_layers.state_dict()
        filename = "bundle_weights.pt"

    torch.save(state, checkpoint_dir / filename)
```

---

## 6. Experiment Workflow

### Phase 1: Design (LLM-assisted)
1. Write hypothesis clearly
2. Define metrics that would confirm/reject
3. Specify configs to compare
4. Estimate compute budget

### Phase 2: Implement
1. Add config to `config.py`
2. Update smoke test if needed
3. Run smoke test locally
4. Fix any issues

### Phase 3: Validate
1. Create Modal test environment
2. Run minimal cloud test (10-50 steps)
3. Verify logs, checkpoints, metrics
4. Delete test environment

### Phase 4: Execute
1. Run full experiment in main environment
2. Monitor via Modal dashboard / wandb
3. Download results when complete

### Phase 5: Analyze (LLM-assisted)
1. Load results
2. Compute summary statistics
3. Generate plots
4. Write report with conclusions
5. Plan follow-up experiments

---

## 7. Common Pitfalls

### Import Errors on Modal
**Problem:** `ModuleNotFoundError: No module named 'src'`
**Fix:** Add `sys.path.insert(0, "/app")` at top of Modal function

### Backward on Frozen Model
**Problem:** `RuntimeError: element 0 of tensors does not require grad`
**Fix:** Ensure at least some parameters have `requires_grad=True`

### Modal API Changes
**Problem:** `AttributeError: 'Image' object has no attribute 'copy_local_dir'`
**Fix:** Check Modal docs, API changes frequently. Use `add_local_dir` now.

### Gradient Hooks Not Firing
**Problem:** Hooks registered but gradients not being modified
**Fix:** Hooks must return the modified gradient, not modify in-place

### Volume Not Persisting
**Problem:** Files written but not visible after function exits
**Fix:** Call `volume.commit()` before function returns

---

## 8. Checklist for New Experiments

```markdown
## Pre-Implementation
- [ ] Hypothesis written clearly
- [ ] Metrics defined
- [ ] Configs specified
- [ ] Compute budget estimated

## Implementation
- [ ] Config added to config.py
- [ ] Smoke test updated if needed
- [ ] Local smoke test passes

## Validation
- [ ] Modal test environment created
- [ ] Minimal cloud run succeeds
- [ ] Checkpoints saved correctly
- [ ] Test environment deleted

## Execution
- [ ] Full run launched
- [ ] Monitoring set up
- [ ] Results downloaded

## Analysis
- [ ] Summary statistics computed
- [ ] Plots generated
- [ ] Report written
- [ ] Next steps identified
```
