# Experiment 002: Measuring Catastrophic Interference

**Status:** Planned
**Priority:** HIGH - This is the core hypothesis test
**Estimated Compute:** 8-16 A10G hours

---

## Hypothesis

**H1:** Multi-bundle models with gradient routing show **lower catastrophic forgetting** than baseline models when trained sequentially on different tasks.

**Quantified prediction:**
- Baseline forgetting: ~30-50% performance drop on Task A after training on Task B
- Bundle model forgetting: <15% performance drop (>50% reduction in forgetting)

---

## Experimental Design

### Tasks

**Task A: Factual Question Answering**
- Dataset: TriviaQA (subset, ~10k examples)
- Metric: Exact match accuracy
- Maps to bundles: `b_ssl_main`, `b_ssl_semantic` (memory-heavy)

**Task B: Code Generation**
- Dataset: CodeAlpaca or MBPP (subset, ~5k examples)
- Metric: Pass@1 on held-out test cases
- Maps to bundles: `b_sem_code`, `b_cog_reason` (logic-heavy)

**Task C: Math Reasoning**
- Dataset: GSM8K (subset, ~5k examples)
- Metric: Final answer accuracy
- Maps to bundles: `b_sem_math`, `b_cog_reason`

### Configurations to Compare

| Config | Description | Expected Forgetting |
|--------|-------------|---------------------|
| `baseline` | No bundles | HIGH (30-50%) |
| `single_adapter` | One shared adapter | HIGH (similar to baseline) |
| `ssl_rl_separate` | SSL/RL separated | MEDIUM (routing helps) |
| `full_modular` | Many specialized bundles | LOW (<15%) |
| `cognitive_semantic_split` | Process/domain separated | LOW-MEDIUM |

### Protocol

```
Phase 1: Train on Task A until convergence
  - Train for 1000 steps or until loss plateaus
  - Evaluate on Task A test set → score_A_after_A
  - Save checkpoint

Phase 2: Train on Task B
  - Continue training for 1000 steps on Task B
  - Evaluate on Task A test set → score_A_after_B
  - Evaluate on Task B test set → score_B_after_B
  - Save checkpoint

Phase 3: Compute interference
  - forgetting_A = (score_A_after_A - score_A_after_B) / score_A_after_A
  - forward_transfer = score_B_after_B vs baseline_B

Phase 4: (Optional) Train on Task C, repeat measurements
```

### Controls

1. **Learning rate control:** Same LR schedule across all configs
2. **Parameter budget control:** Total trainable params within ±10% across configs
3. **Random seed control:** 3 seeds per configuration
4. **Data order control:** Same data order for all runs

---

## Implementation Plan

### Data Pipeline
```python
def create_sequential_dataloaders(tokenizer, tasks=["trivia", "code", "math"]):
    loaders = {}
    for task in tasks:
        if task == "trivia":
            dataset = load_dataset("trivia_qa", "rc", split="train[:10000]")
            # Format: "Question: {q}\nAnswer: {a}"
        elif task == "code":
            dataset = load_dataset("code_alpaca", split="train[:5000]")
            # Format: "Instruction: {inst}\nCode: {code}"
        elif task == "math":
            dataset = load_dataset("gsm8k", "main", split="train[:5000]")
            # Format: "Problem: {q}\nSolution: {a}"

        loaders[task] = create_dataloader(dataset, tokenizer)
    return loaders
```

### Evaluation Functions
```python
def evaluate_trivia(model, tokenizer, test_set):
    """Exact match accuracy on TriviaQA."""
    correct = 0
    for example in test_set:
        pred = generate(model, tokenizer, example["question"])
        if normalize(pred) == normalize(example["answer"]):
            correct += 1
    return correct / len(test_set)

def evaluate_code(model, tokenizer, test_set):
    """Pass@1 on code generation."""
    passed = 0
    for example in test_set:
        code = generate(model, tokenizer, example["instruction"])
        if run_tests(code, example["tests"]):
            passed += 1
    return passed / len(test_set)
```

### Metrics to Log

Per phase:
- Loss curves
- Task-specific accuracy
- Bundle activation norms
- Gradient norms per bundle
- α (routing weight) values

Final analysis:
- Forgetting matrix: `forgetting[task_i][task_j]` = drop in task_i after training task_j
- Forward transfer: Did training on earlier tasks help later tasks?
- Gradient interference: Did gradients from task_j flow into task_i bundles?

---

## Success Criteria

**Primary:**
- `full_modular` forgetting < 50% of `baseline` forgetting (statistically significant, p<0.05)

**Secondary:**
- Clear correlation between bundle separation and reduced forgetting
- Bundle activation patterns show task specificity

**Failure modes to watch:**
- All configs show similar forgetting (bundles don't help)
- Bundle model shows worse final performance (specialization hurts)
- High variance across seeds (unreliable effect)

---

## Compute Budget

| Phase | Steps | Configs | Seeds | GPU Hours |
|-------|-------|---------|-------|-----------|
| Task A training | 1000 | 5 | 3 | 2.5 |
| Task A eval | - | 5 | 3 | 0.5 |
| Task B training | 1000 | 5 | 3 | 2.5 |
| Task B eval | - | 5 | 3 | 0.5 |
| Task C training | 1000 | 5 | 3 | 2.5 |
| Task C eval | - | 5 | 3 | 0.5 |
| **Total** | | | | **~9 hours** |

At $0.60/hr (A10G): **~$5.40**

---

## Follow-up Experiments

If H1 confirmed:
- Experiment 003: Vary overlap percentage between bundles
- Experiment 004: Test with larger models (1B+)

If H1 rejected:
- Analyze why bundles didn't help
- Try different bundle configurations
- Check if gradient routing is actually working as intended
