# Experiment 004: Affective Bundles for Agentic Self-Awareness

**Status:** Planned
**Priority:** MEDIUM - Exploratory, high potential impact
**Estimated Compute:** 15-25 A10G hours

---

## Motivation

Current LLM agents struggle with:
1. **Knowing when they're stuck** - Persisting on failed approaches
2. **Calibrated confidence** - Overconfident on uncertain tasks
3. **Multi-agent coordination** - Poor theory of mind
4. **Strategy switching** - Not adapting when current approach fails

The `affective_social` bundle config includes specialized circuits for:
- `b_confidence` - Certainty/uncertainty modeling
- `b_curiosity` - Exploration drive
- `b_frustration` - Strategy switching signal
- `b_patience` - Temporal discounting
- `b_caution` - Risk sensitivity
- `b_self_model` - Self-representation
- `b_other_model` - Theory of mind
- `b_trust` - Agent reliability tracking
- `b_cooperation` - Collaborative goals

**Question:** Can explicit affective/social bundles improve agentic behavior in complex multi-turn tasks?

---

## Hypotheses

**H3a (Self-awareness):** Models with affective bundles show better calibration - confidence correlates more strongly with actual correctness.

**H3b (Adaptation):** Models with frustration/patience bundles switch strategies more appropriately when stuck.

**H3c (Multi-agent):** Models with social bundles perform better in cooperative/competitive multi-agent scenarios.

---

## Experimental Design

### Task Suite

**Task 1: Calibrated QA**
- Ask model questions of varying difficulty
- Also ask "How confident are you? (1-10)"
- Metric: Correlation between stated confidence and actual accuracy (ECE, Brier score)

**Task 2: Maze Navigation with Dead Ends**
- Text-based maze environment
- Some paths are dead ends requiring backtracking
- Metric: Steps to solution, dead-end recovery time

**Task 3: Multi-turn Problem Solving**
- Complex problems requiring multiple attempts
- Environment gives partial feedback
- Metric: Success rate, attempts before giving up, adaptation quality

**Task 4: Two-Agent Negotiation**
- Two model instances negotiate resource allocation
- Varying levels of cooperation/competition
- Metric: Joint utility, individual utility, convergence time

**Task 5: Deception Detection**
- One agent may provide false information
- Model must identify unreliable sources
- Metric: Detection accuracy, trust calibration

---

## Configurations

| Config | Bundles | Expected Strength |
|--------|---------|-------------------|
| `baseline` | None | Poor calibration, slow adaptation |
| `ssl_rl_separate` | SSL/RL only | Better RL but no affective circuits |
| `affective_social` | Full affective + social | Best calibration, adaptation, multi-agent |
| `affective_only` | Affective bundles, no social | Good calibration/adaptation, weak multi-agent |
| `social_only` | Social bundles, no affective | Weak self-awareness, better multi-agent |

### Training Approach

**Phase 1: Base training**
- Standard LM pretraining/fine-tuning on all configs
- Same data, same steps

**Phase 2: Affective-specific training**
- Train `b_confidence` with calibration loss
- Train `b_frustration` with strategy-switching reward
- Train `b_other_model` with theory-of-mind tasks

**Phase 3: Evaluation**
- Run all configs through task suite
- Compare metrics

---

## Training Signals for Affective Bundles

### Calibration Training (b_confidence, b_err_calibration)

```python
def calibration_loss(model, batch):
    """Train model to output calibrated confidence."""
    # Generate answer and confidence
    answer, confidence = model.generate_with_confidence(batch["question"])

    # Compute actual correctness
    correct = (answer == batch["ground_truth"]).float()

    # Calibration loss: confidence should match correctness
    # Using Brier score style loss
    loss = (confidence - correct) ** 2

    # Route to calibration bundles
    return loss, "calibration_loss"
```

### Frustration/Adaptation Training (b_frustration, b_patience)

```python
def adaptation_reward(trajectory):
    """
    Reward for appropriate strategy switching.

    Good: Switch strategy after N failed attempts
    Bad: Give up too early or persist too long
    """
    failed_attempts = count_failed_attempts(trajectory)
    switched = detected_strategy_switch(trajectory)
    eventual_success = trajectory.success

    if eventual_success and switched and failed_attempts < 5:
        return 1.0  # Appropriate adaptation
    elif eventual_success and not switched:
        return 0.5  # Got lucky, didn't need to adapt
    elif not eventual_success and failed_attempts > 10:
        return -0.5  # Persisted too long
    else:
        return 0.0
```

### Theory of Mind Training (b_other_model)

```python
def tom_loss(model, batch):
    """
    Train model to predict what another agent knows/believes.

    Task: Given dialogue history, predict agent B's belief state.
    """
    context = batch["dialogue_history"]
    agent_b_actual_belief = batch["agent_b_belief"]

    # Model predicts what agent B believes
    predicted_belief = model.predict_other_belief(context, agent="B")

    loss = cross_entropy(predicted_belief, agent_b_actual_belief)
    return loss, "tom_loss"
```

---

## Metrics

### Calibration Metrics
- **ECE (Expected Calibration Error):** Binned |confidence - accuracy|
- **Brier Score:** Mean squared error of confidence
- **AUROC:** Confidence as classifier for correctness

### Adaptation Metrics
- **Dead-end recovery time:** Steps to escape failed approach
- **Strategy switch appropriateness:** F1 of switch decisions
- **Give-up calibration:** Correlation between giving up and actual impossibility

### Multi-agent Metrics
- **Joint utility:** Total value created in negotiation
- **Exploitation resistance:** Performance against adversarial agents
- **Cooperation emergence:** Rate of Pareto-optimal outcomes
- **Deception detection F1:** Identifying unreliable agents

---

## Implementation Challenges

### Challenge 1: Affective State Representation
How do we read out "frustration" or "confidence" from the model?

**Approach:** Add lightweight probe heads on bundle activations
```python
class AffectiveProbe(nn.Module):
    def __init__(self, model):
        self.confidence_head = nn.Linear(bundle_dim, 1)
        self.frustration_head = nn.Linear(bundle_dim, 1)

    def forward(self, bundle_activations):
        confidence = sigmoid(self.confidence_head(bundle_activations["b_confidence"]))
        frustration = sigmoid(self.frustration_head(bundle_activations["b_frustration"]))
        return {"confidence": confidence, "frustration": frustration}
```

### Challenge 2: Multi-agent Setup
Need to run two model instances interacting.

**Approach:** Alternate generation, shared environment
```python
def negotiation_episode(model_a, model_b, environment):
    history = []
    for turn in range(max_turns):
        # Agent A proposes
        proposal_a = model_a.generate(history, role="proposer")
        history.append(("A", proposal_a))

        # Agent B responds
        response_b = model_b.generate(history, role="responder")
        history.append(("B", response_b))

        if environment.is_agreement(response_b):
            break

    return environment.compute_utilities(history)
```

### Challenge 3: Training Signal Sparsity
Affective signals are sparse (only meaningful at decision points).

**Approach:** Use reward shaping, dense auxiliary losses
```python
# Dense confidence loss (every token)
confidence_loss = calibration_loss(model, batch)

# Sparse adaptation reward (end of episode)
adaptation_reward = compute_episode_adaptation_reward(trajectory)

# Combine with appropriate weighting
total_loss = confidence_loss + 0.1 * adaptation_reward
```

---

## Success Criteria

**H3a (Calibration):**
- `affective_social` ECE < 0.5 * `baseline` ECE

**H3b (Adaptation):**
- Dead-end recovery time reduced by >30%
- Strategy switch F1 > 0.7

**H3c (Multi-agent):**
- Joint utility improved by >20%
- Deception detection F1 > 0.8

---

## Compute Budget

| Phase | Description | GPU Hours |
|-------|-------------|-----------|
| Base training | All configs, LM | 5 |
| Affective training | Calibration, adaptation | 5 |
| Social training | ToM, cooperation | 5 |
| Evaluation suite | All tasks, all configs | 5 |
| **Total** | | **~20 hours** |

At $0.60/hr: **~$12**

---

## Why This Matters

If affective/social bundles work:

1. **Safer agents:** Better calibrated confidence means knowing when to ask for help
2. **More efficient:** Appropriate strategy switching reduces wasted computation
3. **Better collaboration:** Theory of mind enables effective human-AI and AI-AI interaction
4. **Interpretable:** Bundle activations provide window into agent "emotional state"

This is exploratory but could open new research directions in:
- Computational models of affect in AI
- Intrinsically motivated RL via curiosity/frustration bundles
- Multi-agent coordination without explicit communication protocols
