# Experiment 003: RL/SSL Gradient Isolation

**Status:** Planned
**Priority:** HIGH - Tests whether RL fine-tuning can be done without destroying base capabilities
**Estimated Compute:** 12-20 A10G hours

---

## Motivation

A major problem in RLHF: reinforcement learning fine-tuning often degrades the model's base capabilities (coherence, factual knowledge, reasoning). The standard solution is KL penalty to a reference model, but this limits how much the model can improve.

**Question:** Can gradient routing to separate RL bundles allow stronger RL optimization while protecting base capabilities?

---

## Hypothesis

**H2:** Models with SSL/RL gradient separation can undergo stronger RL optimization (lower KL penalty) while maintaining base capability better than models with shared circuits.

**Prediction:**
- Baseline + strong RL (low KL): Large capability degradation
- Baseline + weak RL (high KL): Small improvement, small degradation
- Bundle model + strong RL: Large improvement, small degradation (best of both)

---

## Experimental Design

### Setup

**Base capability test:** Language modeling perplexity on held-out text (WikiText)

**RL task:** Helpfulness/harmlessness preference optimization
- Dataset: Anthropic HH-RLHF or synthetic preferences
- Reward: Preference model score or simple heuristic

### Configurations

| Config | RL Strength | KL Coefficient | Expected Outcome |
|--------|-------------|----------------|------------------|
| `baseline` | Strong | 0.01 | High reward, high perplexity degradation |
| `baseline` | Weak | 0.1 | Low reward, low degradation |
| `ssl_rl_separate` | Strong | 0.01 | High reward, low degradation |
| `ssl_rl_separate` | Strong | 0.0 | Very high reward, still low degradation? |

### Key Comparison

The critical test is `ssl_rl_separate` with **zero KL penalty**:
- If bundles work, RL gradients only update `b_rl_*` bundles
- SSL bundles (`b_ssl_main`, `b_ssl_semantic`) should be untouched
- Base perplexity should remain stable even without KL anchor

### Protocol

```
Phase 0: Evaluate base model
  - Perplexity on WikiText
  - Reward model score on sample outputs

Phase 1: RL training
  - PPO/DPO for 500 steps
  - Log: reward, KL from base, perplexity (every 50 steps)

Phase 2: Final evaluation
  - Perplexity on WikiText
  - Reward model score
  - Qualitative: sample outputs

Phase 3: Analysis
  - Plot reward vs perplexity tradeoff curve
  - Compare bundle vs baseline Pareto frontiers
```

---

## Implementation Details

### PPO with Routing

```python
def ppo_step_with_routing(model, batch, router, loss_weights):
    # Forward pass
    outputs = model(batch["input_ids"])
    logits = outputs.logits

    # Compute PPO losses
    policy_loss = compute_policy_loss(logits, batch["actions"], batch["advantages"])
    value_loss = compute_value_loss(values, batch["returns"])
    entropy_loss = compute_entropy_loss(logits)
    kl_loss = compute_kl_loss(logits, batch["ref_logits"])

    # Route each loss to its bundles
    with router.route_gradients("policy_loss"):
        (policy_loss * loss_weights["policy"]).backward(retain_graph=True)

    with router.route_gradients("value_loss"):
        (value_loss * loss_weights["value"]).backward(retain_graph=True)

    with router.route_gradients("entropy_loss"):
        (entropy_loss * loss_weights["entropy"]).backward(retain_graph=True)

    with router.route_gradients("kl_loss"):
        (kl_loss * loss_weights["kl"]).backward(retain_graph=True)

    # SSL bundles received ZERO gradients from above
    # Only if we also train on LM data would they update
```

### Metrics

**Primary:**
- `reward_improvement = final_reward - initial_reward`
- `perplexity_degradation = final_ppl / initial_ppl - 1`
- `efficiency = reward_improvement / perplexity_degradation`

**Secondary:**
- Bundle-specific gradient norms (verify RL gradients go to RL bundles only)
- α evolution during training
- Sample output quality (manual inspection)

---

## Bundle Gradient Flow Analysis

Critical diagnostic: verify gradients actually flow correctly.

```python
def analyze_gradient_flow(model, loss_type):
    """After backward(), check which bundles have non-zero gradients."""
    grad_norms = {}
    for layer in model.bundle_layers:
        for name, bundle in layer.bundles.items():
            norm = sum(p.grad.norm().item()**2 for p in bundle.parameters() if p.grad is not None)**0.5
            grad_norms[name] = norm

    # For policy_loss, expect:
    #   b_rl_policy: HIGH
    #   b_rl_value: ~0
    #   b_ssl_main: ~0
    #   b_ssl_semantic: ~0

    return grad_norms
```

---

## Success Criteria

**Primary:**
- Bundle model achieves >80% of baseline reward improvement
- Bundle model has <50% of baseline perplexity degradation
- Pareto frontier for bundle model dominates baseline

**Secondary:**
- Gradient flow analysis confirms isolation
- Zero-KL bundle model still maintains base capability

**Failure modes:**
- RL gradients leak into SSL bundles despite routing
- Isolated RL bundles can't improve task performance
- Bundle overhead slows training too much

---

## Compute Budget

| Phase | Steps | Configs | KL Settings | GPU Hours |
|-------|-------|---------|-------------|-----------|
| Base eval | - | 2 | - | 0.5 |
| RL training | 500 | 2 | 3 | 6 |
| Periodic eval | - | 2 | 3 | 1 |
| Final eval | - | 2 | 3 | 0.5 |
| Gradient analysis | - | 2 | 1 | 0.5 |
| **Total** | | | | **~8.5 hours** |

At $0.60/hr: **~$5.10**

---

## Expected Insights

### If successful:
- RL fine-tuning can be "sandboxed" to specific circuits
- Opens door to more aggressive RL without capability loss
- Could enable curriculum of RL objectives without interference

### If failed:
- Even with routing, RL optimization may need base model adaptation
- Bundles may be too small to capture RL-relevant representations
- Gradient isolation may not be sufficient; activation patterns matter too

---

## Follow-up

If H2 confirmed:
- Test with DPO (simpler than PPO)
- Test with multiple sequential RL objectives
- Scale to larger models

If H2 rejected:
- Analyze where gradients are actually flowing
- Try larger RL bundles
- Try partial overlap between SSL and RL bundles
