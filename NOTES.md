ok, let’s turn this into an actual experiment plan you could hand to a grad student or an RA and they’d know what to build.

---

# spec: multi-bundle llm training experiments

## 0. high-level goal

**test whether decomposing an llm into explicit “bundles” (specialized parameter subspaces with routed gradients) improves:**

1. **continual learning** (less catastrophic forgetting between tasks)
2. **specialization** (better performance on specific task families at same or lower total params)
3. **controllability** (ability to modulate behavior at inference time by adjusting bundle routing)

vs a **monolithic baseline** with same parameter budget.

---

## 1. core hypotheses

1. **H1 – Gradient quarantine:**
   multi-bundle models will show **lower interference**: performance on task A degrades less when training on task B, compared to monolithic baseline.

2. **H2 – Specialization & efficiency:**
   given same total parameter count, a multi-bundle model **outperforms or matches** monolithic baseline on:

   * bundle-aligned tasks (logic bundle on math/code, memory bundle on factual recall, etc.)
     while preserving overall LM perplexity within a small delta.

3. **H3 – Controllability:**
   explicit routing (α_k) enables **behavior modulation** (reasoning vs recall vs reflection) via simple inference-time controls (scaling α_k), and this modulation is **predictable and monotone**.

4. **H4 – Representation separation:**
   internal activations cluster by task type **within bundles** and are less entangled across tasks than in a monolithic baseline (measured by linear probes / CCA / clustering metrics).

---

## 2. model architecture

### 2.1 base transformer

pick something in the ~1–3B range so it’s trainable:

* vocab: ~32k bpe
* layers: 24
* d_model: 1024
* n_heads: 16
* d_ff: 4096
* rotary or alibi positions, whatever you’re used to

this is the **shared trunk**.

### 2.2 bundles

we define **K = 3 bundles** for v1:

* **B_mem**: factual recall & rote memorization
* **B_logic**: math, code, chain-of-thought, algorithmic reasoning
* **B_meta**: self-critique, “is this correct?”, basic metacognition / reflection

each bundle operates as a **low-rank adapter block per layer**:

for layer ℓ, with hidden `h_ℓ ∈ ℝ^d`:

* projector: `P_k ∈ ℝ^{d_k × d}`
* inverse projector: `Q_k ∈ ℝ^{d × d_k}`
* adapter block (simple):
  `u_k = Adapter_k(P_k h_ℓ)`
  where `Adapter_k` is e.g. bottleneck MLP: `d_k → d_bottleneck → d_k`.

layer computation:

```text
h̃_ℓ = SharedBlock_ℓ(h_ℓ)                            # standard attn + MLP

z_k = P_k h̃_ℓ                                       # into bundle k
u_k = Adapter_k(z_k)                                 # bundle-specific transform
r_k = Q_k u_k                                       # back to base space

h_{ℓ+1} = h̃_ℓ + Σ_k α_{ℓ,k} ⊙ r_k
```

**routing weights α_{ℓ,k}:**

* α_{ℓ,k} ∈ [0, 1], can be:

  * scalar per bundle per layer, or
  * token-wise (vector of length sequence), v1 can keep it simple: scalar per bundle per layer per example.

routing network:

```text
α = softmax(W_r [t || g])   # t = task embedding, g = global context summary
```

for v1 we can cheat: **use task tags** to define α instead of learning them fully, then only later learn a router.

### 2.3 parameter budget

ensure total params ~match baseline:

* baseline: full 1.3B transformer.
* multi-bundle: same shared trunk size, plus bundles, but **reduce trunk width** to keep total params ≈ constant.

we’ll compute exact numbers when instantiating, but spec:

> total params of shared_trunk + Σ bundles ≈ baseline ±5%.

---

## 3. tasks & datasets

we want **three distinct task families** mapping cleanly onto bundles:

1. **T_mem – factual recall / rote LM**

   * openwebtext-like corpora
   * QA datasets (triviaqa, natural questions, etc.)
   * wikipedia slices

2. **T_logic – math & code reasoning**

   * gsm8k, math, minerva-like synthetic math
   * code completion (github code snippets)
   * leetcode-style natural language → code
   * chain-of-thought supervision (cot-augmented datasets)

3. **T_meta – self-critique / reflection**

   * generate answer **and** “critique” or “verify” answer
   * synthetic: give model candidate answer A/B, label which is more consistent with passage / solution
   * explicit “is this reasoning correct?” binary classification over chains-of-thought
   * algorithm:

     * sample outputs from weaker model
     * label correctness with a strong model or heuristic
     * train our model to classify / explain.

we’ll also have **T_lm – generic LM** (mixture of everything) to preserve general capabilities.

each example carries **task tag** ∈ {LM, MEM, LOGIC, META}.

---

## 4. training regimes

we run multiple regimes to test interference & specialization.

### 4.1 shared components

* tokenization identical for all models.
* optimizer: adamw
* LR schedule: cosine or linear warmup + decay
* max sequence length: e.g. 2k tokens
* batch size: tuned to GPU, but fixed across conditions.

### 4.2 models to compare

**M0 – baseline monolithic**

* same transformer, no bundles.
* trained on mixture of T_lm, T_mem, T_logic, T_meta with standard multitask losses.

**M1 – multi-bundle, hard-routed**

* three bundles (mem, logic, meta).

* routing not learned initially; we use deterministic α_k based on task:

  * T_mem: α_mem=1, α_logic=0, α_meta=0
  * T_logic: α_logic=1, α_mem=0, α_meta=0
  * T_meta: α_meta=1, α_mem=0, α_logic=0
  * T_lm: α_mem=α_logic=α_meta=1/3 (or small constant like 0.3 each).

* gradient rule:

  * shared trunk: always updated.
  * bundle k: only updated when α_k > 0 for that batch.

**M2 – multi-bundle, soft-routed + learned router**

same as M1, but:

* add a small router network that produces α_k based on:

  * pooled hidden at [BOS] or special token
  * task tag embedding (optional)

training objective includes:

* sparsity penalty on α (L1 to encourage few bundles per example).
* cross-entropy loss + all others as usual.
* we initialize router with the hard pattern (above) for fast warmstart, then let it learn.

**M3 – multi-bundle, sequential/continual training**

same architecture as M1, but training schedule:

1. phase A: train on T_mem only.
2. phase B: then train on T_logic only.
3. phase C: then train on T_meta only.

compare catastrophic forgetting vs M0 under same schedule.

---

## 5. loss functions

### 5.1 base LM loss

for all models:

* `L_lm` = standard cross-entropy over next token prediction for LM, MEM, LOGIC, META where applicable.

### 5.2 task-specific auxiliary losses

* **T_mem:**

  * retrieval-style: given question, predict answer span.
  * we can treat as LM with prompt formatting, so still `L_lm`, plus optional extra:

    * `L_mem = CE(answer_tokens)` emphasised by higher loss weight for answer part.

* **T_logic:**

  * optionally separate loss on chain-of-thought tokens (upweighted).
  * program correctness classification (given code & test results, classify pass/fail): `L_logic_cls`.

* **T_meta:**

  * correctness classification: `L_meta_cls`
  * explanation generation: `L_meta_lm` (LM loss for explanation tokens)

overall per-batch loss example:

```text
L_batch = w_lm * L_lm + w_task * (L_mem | L_logic_cls | L_meta_cls + ... )
```

with task-dependent weights.

### 5.3 router regularization (for M2)

* sparsity: `L_sparse = λ_sparse * Σ_k |α_k|`
* entropy penalty: optionally encourage peaky routing: `L_ent = λ_ent * H(softmax(α))`

total:

```text
L_total = L_batch + L_sparse + L_ent
```

---

## 6. evaluation protocol

### 6.1 per-task performance

measure:

1. **LM:**

   * perplexity on held-out generic text.

2. **MEM:**

   * accuracy on factual QA benchmarks (e.g. nq, triviaqa).
   * exact match / F1.

3. **LOGIC:**

   * gsm8k accuracy
   * codex-style human eval or automated test pass rate for code tasks.
   * math/proof benchmarks (if included).

4. **META:**

   * correctness discrimination accuracy (given A/B or A vs ground truth).
   * calibration of confidence vs correctness (Brier score).

compare M0 vs M1 vs M2 vs M3.

### 6.2 interference / continual learning

for sequential regime (M0 vs M3):

* after each phase (A, B, C), evaluate **all tasks**.
* measure **Δ performance** on previously-trained tasks:

  * e.g. after phase B (logic), how much did mem QA performance drop vs end of phase A?

quantify catastrophic forgetting:

```text
interference(task X) = score_X_after_new_phase - score_X_at_end_of_own_phase
```

multi-bundle should show **smaller negative deltas**.

### 6.3 controllability experiments (M1/M2)

at inference, manipulate α_k:

* for a given LOGIC task, sweep α_logic from 0 → 1, α_mem from 0 → 1, keeping sum normalized or not.
* measure task accuracy vs α.

we expect:

* performance on logic benchmarks **monotone increasing** in α_logic (up to some saturation)
* little sensitivity to α_mem for logic tasks
* the opposite pattern for mem tasks.

also test “reflection mode”:

* for T_meta prompts, increase α_meta → measure gains in self-correction / critique quality.

### 6.4 representation analysis

pull internal activations:

* choose a fixed layer ℓ.
* for a batch of examples from each task, extract:

  * h̃_ℓ (trunk)
  * u_k (bundle activations)

run:

* PCA / t-SNE / UMAP separately on trunk vs each bundle.
* linear probes to classify task type from activations.

metrics:

* classification accuracy from trunk vs from bundles.
* separation (e.g. silhouette score) of task clusters in bundle spaces.

**hypothesis:** bundle activations show stronger clustering by task than trunk or M0 monolithic activations.

---

## 7. ablations

to isolate which design choices matter:

1. **No-projector ablation**

   * instead of P_k/Q_k, use full-dim adapters (no explicit subspace).
   * see if projection structure itself is crucial or just gradient scoping.

2. **Shared-adapter ablation**

   * one big adapter shared across all tasks, no routing.
   * tests whether any advantage comes just from extra capacity vs structured bundles.

3. **Random routing**

   * for M2, random α_k during training (subject to normalization).
   * check whether structured routing actually drives specialization.

4. **Bundle dropout**

   * randomly zero α_k for some bundles during training.
   * measure robustness and reliance on specific bundles.

---

## 8. implementation details

### 8.1 code structure (pseudo)

training step (M1 hard-routed):

```python
def train_step(batch, model, optimizer):
    x, y, task_tag = batch

    # set routing coefficients
    alpha = routing_from_task_tag(task_tag)  # fixed mapping

    # forward
    logits, bundle_states = model(x, alpha=alpha)

    # compute losses based on task
    L_lm = cross_entropy(logits, y)
    L_task = task_specific_loss(task_tag, logits, bundle_states, batch)

    L = w_lm * L_lm + w_task * L_task

    # backward
    optimizer.zero_grad()
    L.backward()

    # gradient mask: zero grads for bundles where alpha_k == 0
    mask_bundle_grads(model, alpha)

    optimizer.step()
```

for M2:

* router predicts α: `alpha = router(h_global, task_embed)`
* add sparsity/entropy losses.

### 8.2 gradient masking

bundle parameters are grouped by bundle. for each bundle k:

```python
if alpha_k == 0:
    for p in bundle_k.parameters():
        p.grad = 0
```

for soft α, we can choose either to:

* still update all bundles but scale LR per bundle by α_k, or
* treat α_k as probability of updating (stochastic routing).

v1: deterministic mask via hard routing, simple.

### 8.3 logging

log per-step:

* loss per task type
* α distributions (for M2)
* norm of gradients per bundle
* activation magnitudes per bundle
* interference metrics across phases.

---

## 9. success criteria

we call multi-bundle “succeeded” if:

1. **continual learning:**
   under sequential training (M3 vs M0), average performance drop on old tasks is **significantly smaller** (e.g. ≥ 30% reduction in forgetting).

2. **specialization:**
   for each task family:

   * best multi-bundle model ≥ monolithic baseline accuracy at same total params,
   * or if below, offers stronger controllability tradeoffs.

3. **controllability:**
   changing α_k at inference consistently modulates performance on bundle-aligned tasks more than on others, in a predictable direction.

4. **geometry:**
   bundle activations show clear task-aligned structure (higher cluster separation / probe accuracy than trunk and M0).

---

if you want, next round we can pin **actual numbers** (hidden sizes per bundle, param counts), plus a concrete dataset mix table and early-stopping / LR sched tables so this is turnkey runnable.

---

but lets make it have a longer more specific and more intentionally thought out list of bundles. like some higher dimensional general 'memorization' bundles, but specific ones for specific semanatic categories like the tasks involving 'animal' data or the tasks involving machine learning which get lower dimensional. also diffferent types of error and RL flow feedback and gradients through different partially overlapping, non-full-rank subspaces. make sense? lsit it all out technical formal and detailed:

---

yeah this makes sense. let’s formalize the “many bundles, overlapping low-rank subspaces, topic- and error-specific pipes” thing.

i’ll write it like a design note you can build from.

---

## 0. objective

we want an LLM whose parameter/activation space is decomposed into **many partially overlapping low-rank bundles**, where:

* **semantic bundles** specialize in content domains (animals, ml, finance, law, etc.)
* **cognitive bundles** specialize in process types (memorization, abstraction, logical reasoning, metacognition, policy / RL)
* **error / feedback bundles** carry *specific* training signals (confusion, calibration, inconsistency, RL advantage, etc.)

and **each loss type routes its gradients through a different non-full-rank subspace** of parameters, with overlaps controlled intentionally.

---

## 1. parameterization: multi-bundle, overlapping low-rank subspaces

### 1.1 base notation

* transformer layers ℓ = 1…L
* hidden dimension d
* base hidden at layer ℓ: `h_ℓ ∈ ℝ^{B×T×d}` (batch × tokens × dim)

we introduce **M bundles** indexed by `k = 1…M`.

for each bundle k:

* rank `r_k << d` (low-rank subspace)
* projection matrix `P_k ∈ ℝ^{r_k×d}`
* inverse projection `Q_k ∈ ℝ^{d×r_k}`
* bundle parameters `θ_k` (adapter weights etc.)

importantly, **P_k are not orthogonal nor spanning**; they **overlap**.

formal:

* define a shared basis `U ∈ ℝ^{d×R}` of subspace directions (R still ≤ d).
* each bundle k is defined via:

  * shape weights `W_k ∈ ℝ^{r_k×R}`
  * so `P_k = W_k Uᵀ` and `Q_k = U W_kᵀ`.

this makes the overlapping explicit:
bundles share parts of U (global directions) but mix them differently.

### 1.2 layer computation with bundles

for each layer ℓ:

```text
h̃_ℓ = SharedBlock_ℓ(h_ℓ; θ_shared_ℓ)        # normal attention + MLP

for each bundle k:
  z_k = P_k h̃_ℓ                             # project into subspace r_k
  u_k = Adapter_k(z_k; θ_k)                  # bundle-specific nonlinearity
  r_k = Q_k u_k                              # back to ℝ^d

h_{ℓ+1} = h̃_ℓ + Σ_k α_{ℓ,k} ⊙ r_k
```

* α_{ℓ,k} ∈ ℝ (or ℝ^{B×T}) are routing weights for bundle k at layer ℓ.
* you can gate per token or per example; v1: per-example scalar per bundle per layer.

---

## 2. bundle taxonomy (long, intentional, high-dimensional)

we’ll organize bundles into three classes:

1. **general bundles** – broad functions (memorization, abstraction, reasoning, etc.)
2. **semantic bundles** – domain/topic-specific “memorization + patterns”
3. **signal bundles** – error, RL, and auxiliary supervision channels.

### 2.1 general cognitive bundles

**B_mem_global** (high-dim memory bundle)

* dimension: r_mem_big ~ 0.3–0.5 d (big subspace)
* role: high-capacity, general long-term memorization (facts, names, n-grams, templates)
* update signals:

  * LM loss on all corpora, but especially factual QA, encyclopedic text
  * retrieval-style losses (question → answer span)
* gradient routing:

  * LM cross-entropy **heavily projects** into this bundle via `P_mem_global`
  * RL / meta signals mostly *avoid* this bundle (to protect stability)

**B_logic_global** (global reasoning / algorithmic bundle)

* dimension: r_logic_big ~ 0.2–0.3 d
* role: algorithmic reasoning, math, code, complex transformations, CoT skeletons
* signals:

  * math/code datasets, gsm8k, codeforces-style tasks, CoT-supervised dumps
  * algorithmic synthetic tasks (sorting, counting, regex-like pattern tasks)
* gradient routing:

  * logic / CoT loss predominantly flows into B_logic_global, with minor shared trunk updates.

**B_abstraction** (concept formation / compression)

* dimension: r_abs ~ 0.15–0.25 d
* role: compress repeated patterns into abstract concepts; cluster semantics
* signals:

  * contrastive learning across paraphrases
  * doc-level representation learning; Siamese sentence/document tasks
  * topic clustering, style identity
* gradient routing:

  * contrastive / metric-learning losses flow here
  * LM gradients *lightly* touch this, mostly via shared trunk.

**B_temporal** (planning / sequence coherence)

* dimension: r_temp ~ 0.1–0.2 d
* role: cross-token / cross-turn plan & goal state; temporal coherence
* internal state:

  * global goal vector `G ∈ ℝ^{r_temp}` updated per step/segment
* signals:

  * trajectory consistency loss (same plan across windows)
  * next-subgoal prediction
  * alignment between high-level instructions and low-level token sequence.

**B_meta** (self-awareness / self-evaluation)

* dimension: r_meta ~ 0.05–0.1 d
* role: confusion, consistency, calibration, “am i right?”
* signals:

  * auxiliary heads for confusion, error prediction, self-critique
  * tasks where model must judge correctness / consistency of text (including its own output).

**B_policy** (RL / decision bundle)

* dimension: r_policy ~ 0.05–0.1 d
* role: map world-model → action distributions; RL-specific adaptation.
* signals:

  * RL policy gradient, bandit rewards, preference optimization
* strongly KL-clamped to avoid global drift.

### 2.2 semantic topic bundles (low-rank, domain-scoped)

these are **narrower bundles** specializing in semantic slices, realized as small r_k (e.g. 0.02–0.05 d), with strong overlaps with B_mem_global and B_abstraction.

example set:

* **B_sem_animals**
* **B_sem_ml**
* **B_sem_programming**
* **B_sem_finance**
* **B_sem_law**
* **B_sem_medical**
* etc., depending on data.

**how they’re defined:**

each semantic bundle’s P_k is constructed to **overlap**:

* with `P_mem_global` (since they store domain-specific facts)
* with `P_abstraction` (since they capture domain-typical abstractions)

formally:

let U_mem, U_abs be sub-bases:

* U_mem ∈ ℝ^{d×R_mem}, U_abs ∈ ℝ^{d×R_abs}
* stack them: `[U_mem | U_abs] = U_sem_base ∈ ℝ^{d×R_sem}`

for B_sem_ml:

* W_sem_ml ∈ ℝ^{r_sem×R_sem},
* P_sem_ml = W_sem_ml U_sem_baseᵀ

so it’s **literally a low-rank mixture of memory+abstraction subspaces**, tuned to ML domain.

**training data assignment:**

* use topic classifier (or simple lexicon heuristics) to label each training example with domain tags.
* for domain-specific examples:

  * route more gradients into corresponding B_sem_k.

**gradient routing semantics:**

for a datapoint with topics {animals, ml}:

* α_sem_animals ≈ 0.7 (if strongly animal)
* α_sem_ml ≈ 0.8 (if strongly ml)
* α_mem_global, α_abstraction also non-zero (since they’re supersets).

the update to parameters related to animals doesn’t fully touch ML, but both share B_mem_global / B_abstraction, so **high-level patterns generalize**.

### 2.3 signal / error / RL bundles

these capture **different types of feedback** across tasks, sharing some base directions but with distinct projections.

**B_err_confusion** (uncertainty / confusion)

* subspace for tracking where model is uncertain.
* signals:

  * train a head to predict entropy of token distribution
  * or correctness of answer given ground truth
* updates:

  * classification/regression loss for predicted entropy vs actual error.

**B_err_consistency** (logical / narrative consistency)

* subspace encoding self-contradiction, narrative breaks, logical violations.
* signals:

  * tasks where model detects contradictions in pairs/sequences
  * synthetic examples with inserted inconsistencies.

**B_err_calibration** (probability calibration)

* subspace for mapping logit patterns into calibrated confidences.
* signals:

  * supervised calibration (e.g. ECE/Brier optimization).

**B_rl_value** (value function / critic)

* subspace dedicated to value estimation in RL settings.
* signals:

  * TD-error / critic loss.

**B_rl_advantage** (advantage / exploration)

* subspace encoding advantage estimates, exploration bonuses, curiosity.
* signals:

  * advantage regression
  * curiosity-style intrinsic rewards (prediction error / information gain).

each of these B_err/B_rl bundles shares some base directions from U_meta, U_policy, U_abstraction:

* U_err_base = concat(U_meta, U_policy, maybe U_abs).
* P_err_confusion = W_conf U_err_baseᵀ, etc.

so e.g. confusion & consistency bundles overlap in meta/policy space but diverge.

---

## 3. routing & gradient flow by signal type

### 3.1 loss-type → subspace mapping

for each training signal S, define a mapping:

* **S_lm** (LM cross-entropy):

  * main: B_mem_global, B_abstraction
  * side: semantic bundles depending on topic
  * minimal: B_logic_global, B_temporal, B_meta (just trunk).

* **S_logic** (logic/maths/code-specific tasks):

  * main: B_logic_global, B_abstraction
  * side: relevant B_sem_* (e.g. B_sem_ml for ML text)
  * little: B_mem_global.

* **S_meta_correctness** (self-critique, correctness classification):

  * main: B_meta, B_err_confusion, B_err_consistency
  * side: B_logic_global (for deeper patterns).

* **S_calibration**:

  * main: B_err_calibration, B_meta
  * side: B_mem_global (so memory heavy distributions get calibrated too).

* **S_RL_policy**:

  * main: B_policy
  * side: B_temporal, B_rl_advantage

* **S_RL_value**:

  * main: B_rl_value
  * side: B_temporal, B_meta.

### 3.2 formal gradient projection

for each signal S with base gradient g_S over shared trunk parameters θ_shared:

we define an **update subspace** via matrix U_S ∈ ℝ^{d×R_S}, derived from concatenating relevant bases (U_mem, U_logic, U_meta, etc.) and projecting g_S into that subspace.

simplified view at parameter/activation level:

* for bundles k ∈ B(S) (bundles affected by S):

  * compute gradients ∂L_S/∂θ_k as usual via backprop through P_k, Adapter_k, Q_k.

* for bundles j ∉ B(S):

  * zero gradients (or strong attenuation) for θ_j under signal S.

in code terms:
**mask gradients per-bundle per-loss-type**.

you can also do **per-loss subspace projection** on the shared trunk:

* let Δθ_S be the unconstrained gradient from L_S.
* project:

  * Δθ_S' = Proj_{V_S}(Δθ_S) where V_S is a low-rank subspace for that signal.

implementation: maintain a low-rank factorization for each signal:

* V_S = A_S B_Sᵀ (like LoRA for gradients)
* update only along those directions.

but for v1, per-bundle masking is probably easier.

---

## 4. semantic labeling and bundle routing

### 4.1 topic inference

each training sample gets:

* topic vector τ ∈ ℝ^{N_topics} (multi-hot or soft).

  * derived via:

    * off-the-shelf topic classifier, or
    * unsupervised clustering of embeddings + manual labeling, or
    * heuristics (keyword lists).

### 4.2 routing weights for semantic bundles

for each semantic bundle k (mapped to topic t_k):

* α_sem_k = f(τ_t_k) = scaled topic weight.
* also, global α_mem_global and α_abstraction might depend on total semantic complexity or doc length.

example:

* input about “deep learning for computer vision on cats”:

  * τ_ml = 0.9, τ_animals = 0.7
  * α_sem_ml ≈ 0.9, α_sem_animals ≈ 0.7
  * α_mem_global ≈ 0.8 (informative text)
  * α_logic_global ≈ 0.4 (some methodology reasoning).

during backprop:

* B_sem_ml and B_sem_animals get non-zero gradient; others semi-frozen.

---

## 5. KL / stability per-bundle & per-signal

we don’t want RL / weird losses to wreck the base.

### 5.1 per-bundle KL clamps

for any risky signal (RL, preference, etc.):

* we maintain a reference model (pre-RL snapshot) with logits `π_ref`.
* current model logits: `π_curr`.

we can define **bundle-weighted KL**:

```text
KL_bundle = Σ_k β_k * KL(Π_k(π_curr) || Π_k(π_ref))
```

where:

* Π_k is projection of logits relevant to bundle k (e.g. via attention or mapping from hidden states of that bundle).
* β_k are weights: high for protected bundles (B_mem_global), lower for B_policy.

we then clamp via:

```text
L += λ * max(0, KL_bundle - τ)
```

with λ, τ tuned per-signal.

### 5.2 signal-specific stability

for S_RL_policy:

* heavy KL on B_mem_global, B_abstraction, B_sem_*
* lighter KL on B_policy, B_rl_advantage

for S_meta:

* moderate KL overall; we allow meta to reshape evaluations but not raw knowledge.

---

## 6. experimental knobs (for the actual study)

things you can systematically vary:

1. **number and size of semantic bundles**

   * few large domains vs many micro-domains.
   * see effect on specialization and interference.

2. **overlap structure**

   * share more base directions between semantic bundles and B_mem_global vs making them more disjoint.
   * measure generalization vs isolation.

3. **signal subspace rank**

   * RL updates restricted to extremely low-rank vs moderate rank.
   * measure RL performance vs base stability.

4. **hard vs soft gradient masks**

   * strict zeroing of grads outside B(S) vs attenuated.
   * measure training stability.

5. **learned topic routing vs heuristic topic routing**

   * let the model infer routing end-to-end vs using external topic labels.

---

## 7. how this looks in one layer (concrete tensor flow)

for one batch, one layer:

```python
# h: [B, T, d]
h_tilde = shared_block(h)

bundle_outputs = []
for k in bundles:
    # project into bundle subspace
    z_k = h_tilde @ P_k.T             # [B, T, r_k]
    u_k = adapter_k(z_k)              # [B, T, r_k]
    r_k = u_k @ Q_k.T                 # [B, T, d]

    bundle_outputs.append((k, r_k))

# combine with routing weights (per-example or per-batch)
h_next = h_tilde.clone()
for k, r_k in bundle_outputs:
    alpha_k = alpha[k]                # scalar or [B, 1, 1]
    h_next = h_next + alpha_k * r_k
```

backward pass:

* compute loss L = Σ_S L_S (sum over signals active for this batch).
* for each signal S, we optionally:

  * mask gradients for bundles not in B(S).
  * or project shared gradients into signal subspace.

---

this gives you a **long, specific, intentionally structured bundle taxonomy** with:

* high-dim global memorization/logic/abstraction bundles
* low-dim topic bundles layered on top, sharing those directions
* multiple error/RL bundles with their own overlapping subspaces
* different losses and RL feedback constrained to specific, partially overlapping low-rank regions.

next step, if you want: we can write a **concrete experiment matrix** (models × bundles × training regimes) and a minimal v0 config (exact dims, number of domains, which public datasets to use) so you can actually implement and compare.
