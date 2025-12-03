# Experiment 005: Optimal Subspace Overlap Structure

**Status:** Planned
**Priority:** MEDIUM - Foundational for understanding mechanism
**Estimated Compute:** 10-15 A10G hours

---

## Motivation

The bundle architecture has a key design choice: **how much should bundle subspaces overlap?**

- **Full overlap (100%):** All bundles share the same subspace → equivalent to single adapter
- **Zero overlap (0%):** Bundles are orthogonal → maximum isolation, no knowledge sharing
- **Partial overlap (10-80%):** Bundles share some directions → tradeoff

Current configs use qualitative overlap ("high", "low") but we haven't systematically studied the optimal amount.

---

## Hypothesis

**H4:** There exists an optimal overlap percentage that balances:
- **Isolation benefit:** Reducing interference between tasks
- **Sharing benefit:** Enabling knowledge transfer between related tasks

**Prediction:** Optimal overlap is task-dependent:
- Similar tasks (QA ↔ Summarization): High overlap (~60-80%) is better
- Dissimilar tasks (QA ↔ Code): Low overlap (~10-30%) is better

---

## Experimental Design

### Controlled Overlap

Create configs with explicit overlap control:

```python
def create_overlap_config(overlap_pct: float, num_bundles: int = 4):
    """
    Create bundle config with specified overlap percentage.

    overlap_pct: 0.0 = orthogonal, 1.0 = identical subspaces
    """
    # Total rank budget
    total_rank = 256

    # Shared dimensions (overlap)
    shared_rank = int(total_rank * overlap_pct)

    # Bundle-specific dimensions
    specific_rank = (total_rank - shared_rank) // num_bundles

    bundles = {}
    for i in range(num_bundles):
        bundles[f"bundle_{i}"] = BundleSpec(
            name=f"bundle_{i}",
            rank_fraction=(shared_rank + specific_rank) / d_model,
            base_pools=["shared"] + [f"specific_{i}"],
            pool_weights=[overlap_pct, 1 - overlap_pct],
            gradient_sources=[f"task_{i}"],
        )

    return bundles
```

### Overlap Levels to Test

| Overlap | Description |
|---------|-------------|
| 0% | Fully orthogonal bundles |
| 20% | Low overlap |
| 40% | Medium-low overlap |
| 60% | Medium-high overlap |
| 80% | High overlap |
| 100% | Single shared adapter |

### Task Pairs

**Similar pair:** Question Answering ↔ Reading Comprehension
- Both are text understanding tasks
- Expect high optimal overlap

**Dissimilar pair:** Question Answering ↔ Code Generation
- Very different modalities
- Expect low optimal overlap

**Mixed pair:** Math ↔ Code
- Overlapping in logic, different in syntax
- Expect medium optimal overlap

### Protocol

```
For each (task_pair, overlap_level):
    1. Initialize model with overlap_level config
    2. Train on task A (500 steps)
    3. Evaluate task A
    4. Train on task B (500 steps)
    5. Evaluate task A (measure forgetting)
    6. Evaluate task B (measure learning)

Compute:
    - Forgetting curve vs overlap
    - Learning curve vs overlap
    - Optimal overlap for each task pair
```

---

## Measuring Overlap

### Theoretical Overlap

Given projections P_i, P_j for bundles i, j:

```python
def theoretical_overlap(P_i, P_j):
    """
    Compute subspace overlap via principal angles.
    Returns value in [0, 1].
    """
    # Orthonormalize
    Q_i, _ = torch.linalg.qr(P_i.T)  # [d, r_i]
    Q_j, _ = torch.linalg.qr(P_j.T)  # [d, r_j]

    # Compute singular values of Q_i^T @ Q_j
    # These are cosines of principal angles
    S = torch.linalg.svdvals(Q_i.T @ Q_j)

    # Mean squared cosine as overlap measure
    overlap = (S ** 2).mean().item()
    return overlap
```

### Empirical Overlap

Measured from gradient flow:

```python
def empirical_overlap(model, task_a_batch, task_b_batch):
    """
    Measure how much gradients from task A affect task B bundles.
    """
    # Compute gradients from task A
    model.zero_grad()
    loss_a = compute_loss(model, task_a_batch)
    loss_a.backward()
    grads_a = {name: p.grad.clone() for name, p in model.named_parameters()}

    # Compute gradients from task B
    model.zero_grad()
    loss_b = compute_loss(model, task_b_batch)
    loss_b.backward()
    grads_b = {name: p.grad.clone() for name, p in model.named_parameters()}

    # Compute gradient overlap (cosine similarity)
    overlap = cosine_similarity(flatten(grads_a), flatten(grads_b))
    return overlap
```

---

## Analysis

### Primary Plot: Forgetting vs Overlap

```
        Forgetting
            │
      0.5 ──┤        ╱
            │       ╱
      0.4 ──┤      ╱
            │     ╱
      0.3 ──┤    ╱
            │   ╱
      0.2 ──┤──╱
            │╱
      0.1 ──┤
            │
          ──┴────────────────────
            0%   20%   40%   60%   80%  100%
                      Overlap
```

Expected: Monotonic increase in forgetting with overlap (for dissimilar tasks)

### Secondary Plot: Final Performance vs Overlap

```
        Task B Accuracy
            │
      0.8 ──┤          ──────────
            │        ╱
      0.7 ──┤      ╱
            │    ╱
      0.6 ──┤  ╱
            │╱
      0.5 ──┤
            │
          ──┴────────────────────
            0%   20%   40%   60%   80%  100%
                      Overlap
```

Expected: Learning speed/final accuracy may improve with some overlap (transfer)

### Optimal Overlap: Pareto Frontier

Plot (forgetting, learning) for each overlap level. Find Pareto-optimal overlap.

---

## Theoretical Framework

### Why Some Overlap Might Help

1. **Shared representations:** Low-level features (syntax, entities) useful for multiple tasks
2. **Regularization:** Overlap constrains capacity, may reduce overfitting
3. **Transfer:** Positive transfer between related tasks

### Why Too Much Overlap Hurts

1. **Interference:** Gradients from task B corrupt task A representations
2. **Capacity competition:** Same parameters pulled in different directions
3. **No specialization:** Can't develop task-specific features

### Predicted Optimal Overlap

```
Optimal Overlap = f(task_similarity, relative_importance, capacity_constraints)
```

For dissimilar tasks with equal importance: ~20-30%
For similar tasks: ~60-80%
For one dominant task: Higher overlap okay (auxiliary task acts as regularizer)

---

## Success Criteria

1. **Clear trend:** Forgetting increases monotonically with overlap (for dissimilar tasks)
2. **Optimal exists:** Some overlap level outperforms both extremes
3. **Task-dependent:** Different task pairs have different optimal overlaps
4. **Predictable:** Can estimate optimal overlap from task similarity measure

---

## Compute Budget

| Overlap | Task Pairs | Seeds | GPU Hours |
|---------|------------|-------|-----------|
| 6 levels | 3 pairs | 2 | 9 |
| Evaluation | - | - | 3 |
| **Total** | | | **~12 hours** |

At $0.60/hr: **~$7.20**

---

## Implications

If we can characterize optimal overlap:

1. **Automated config selection:** Given task pair, compute similarity, set overlap
2. **Curriculum learning:** Start with high overlap, anneal to low as specialization develops
3. **Dynamic overlap:** Learn overlap as a parameter during training
4. **Architecture search:** Overlap becomes a hyperparameter to optimize

This experiment provides foundational understanding for all other bundle experiments.
