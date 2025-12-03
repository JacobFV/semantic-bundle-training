"""Bundle configuration system - defines bundle specs and experimental configurations."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class BundleConfig(Enum):
    """Experimental configurations for bundle overlap/separation."""

    BASELINE = "baseline"  # No bundles, standard fine-tuning
    SINGLE_ADAPTER = "single_adapter"  # One shared LoRA-style adapter
    ALL_OVERLAP = "all_overlap"  # Multiple bundles, ~80% overlap
    SSL_RL_SEPARATE = "ssl_rl_separate"  # SSL and RL circuits disjoint (~10% overlap)
    SSL_RL_SEMANTIC_SEPARATE = "ssl_rl_semantic_separate"  # SSL, RL, semantic all separated
    COGNITIVE_SEMANTIC_SPLIT = "cognitive_semantic_split"  # Process vs domain bundles
    FULL_MODULAR = "full_modular"  # Many small bundles, ~5% overlap
    HIERARCHICAL = "hierarchical"  # Semantic nested inside cognitive
    FEEDBACK_ISOLATED = "feedback_isolated"  # Only feedback/error bundles isolated
    AFFECTIVE_SOCIAL = "affective_social"  # Emotional + multi-agent bundles for agentic tasks


# Shared basis pools - these are mixed to create overlapping bundle subspaces
# Values are default ranks (will be scaled by d_model)
BASE_POOLS = {
    # Self-supervised / autoregressive
    "ssl": 0.15,
    "ssl_coherence": 0.05,

    # RL signal components (TRPO/PPO decomposition)
    "rl_policy": 0.08,
    "rl_value": 0.08,
    "rl_advantage": 0.05,
    "rl_entropy": 0.04,
    "rl_kl": 0.04,

    # Cognitive processes
    "cognitive_reasoning": 0.12,
    "cognitive_memory": 0.12,
    "cognitive_attention": 0.06,
    "cognitive_planning": 0.08,
    "cognitive_abstraction": 0.08,

    # Semantic domains
    "semantic_code": 0.06,
    "semantic_math": 0.06,
    "semantic_natural": 0.08,
    "semantic_science": 0.05,
    "semantic_social": 0.05,

    # Affective / self-regulatory (for agentic self-awareness)
    "affective_confidence": 0.04,
    "affective_curiosity": 0.04,
    "affective_frustration": 0.03,
    "affective_patience": 0.03,
    "affective_caution": 0.03,

    # Social / multi-agent
    "social_self_model": 0.04,
    "social_other_model": 0.05,
    "social_trust": 0.03,
    "social_cooperation": 0.03,

    # Error / feedback signals
    "error_factual": 0.04,
    "error_logical": 0.04,
    "error_consistency": 0.04,
    "error_calibration": 0.03,
}


@dataclass
class BundleSpec:
    """Specification for a single bundle."""

    name: str
    rank_fraction: float  # Fraction of d_model for this bundle's rank
    base_pools: List[str]  # Which shared bases this bundle draws from
    pool_weights: List[float]  # Weight for each base pool
    gradient_sources: List[str]  # Which loss types flow through this bundle
    description: str = ""  # Human-readable description

    def get_rank(self, d_model: int) -> int:
        """Compute actual rank given model dimension."""
        return max(16, int(d_model * self.rank_fraction))


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    bundle_config: BundleConfig
    model_name: str = "gpt2"
    d_model: int = 768
    num_layers: int = 12

    # Training params
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 100

    # Loss weights
    lm_loss_weight: float = 1.0
    policy_loss_weight: float = 0.1
    value_loss_weight: float = 0.5
    entropy_loss_weight: float = 0.01
    kl_loss_weight: float = 0.1

    # Bundle-specific
    adapter_bottleneck_ratio: float = 0.5
    bundle_init_scale: float = 0.01
    alpha_init: float = 0.5  # Initial routing weight

    # Regularization
    bundle_l1_weight: float = 0.0  # Sparsity on bundle activations
    overlap_penalty_weight: float = 0.0  # Penalize excessive overlap


def make_config(config_type: BundleConfig, d_model: int = 768) -> Dict[str, BundleSpec]:
    """Generate bundle specifications for a given configuration."""

    if config_type == BundleConfig.BASELINE:
        return {}

    if config_type == BundleConfig.SINGLE_ADAPTER:
        return {
            "shared": BundleSpec(
                name="shared",
                rank_fraction=0.25,
                base_pools=["ssl", "rl_policy", "rl_value", "cognitive_reasoning", "cognitive_memory"],
                pool_weights=[1.0, 1.0, 1.0, 1.0, 1.0],
                gradient_sources=["all"],
                description="Single shared adapter (LoRA-style baseline)",
            )
        }

    if config_type == BundleConfig.ALL_OVERLAP:
        # Multiple bundles but highly overlapping subspaces
        return {
            "b_ssl": BundleSpec(
                name="b_ssl",
                rank_fraction=0.15,
                base_pools=["ssl", "cognitive_memory", "cognitive_reasoning"],
                pool_weights=[1.0, 0.8, 0.6],
                gradient_sources=["lm_loss"],
                description="Self-supervised learning circuit",
            ),
            "b_rl": BundleSpec(
                name="b_rl",
                rank_fraction=0.12,
                base_pools=["rl_policy", "rl_value", "cognitive_reasoning", "cognitive_memory"],
                pool_weights=[1.0, 1.0, 0.7, 0.5],
                gradient_sources=["policy_loss", "value_loss", "entropy_loss"],
                description="RL circuit with high overlap to SSL",
            ),
            "b_semantic": BundleSpec(
                name="b_semantic",
                rank_fraction=0.10,
                base_pools=["semantic_code", "semantic_math", "semantic_natural", "cognitive_memory"],
                pool_weights=[1.0, 1.0, 1.0, 0.8],
                gradient_sources=["lm_loss"],
                description="Semantic knowledge with high overlap",
            ),
        }

    if config_type == BundleConfig.SSL_RL_SEPARATE:
        # Core experiment: SSL and RL through disjoint circuits
        return {
            "b_ssl_main": BundleSpec(
                name="b_ssl_main",
                rank_fraction=0.18,
                base_pools=["ssl", "ssl_coherence", "cognitive_memory", "cognitive_abstraction"],
                pool_weights=[1.0, 1.0, 0.8, 0.6],
                gradient_sources=["lm_loss"],
                description="Main autoregressive prediction circuit",
            ),
            "b_ssl_semantic": BundleSpec(
                name="b_ssl_semantic",
                rank_fraction=0.12,
                base_pools=["semantic_code", "semantic_math", "semantic_natural", "semantic_science"],
                pool_weights=[1.0, 1.0, 1.0, 1.0],
                gradient_sources=["lm_loss"],
                description="Domain-specific SSL circuit",
            ),
            "b_rl_policy": BundleSpec(
                name="b_rl_policy",
                rank_fraction=0.10,
                base_pools=["rl_policy", "rl_advantage", "cognitive_planning"],
                pool_weights=[1.0, 1.0, 0.5],
                gradient_sources=["policy_loss"],
                description="Policy gradient circuit - isolated from SSL",
            ),
            "b_rl_value": BundleSpec(
                name="b_rl_value",
                rank_fraction=0.08,
                base_pools=["rl_value", "cognitive_planning", "cognitive_abstraction"],
                pool_weights=[1.0, 0.6, 0.4],
                gradient_sources=["value_loss"],
                description="Value estimation circuit",
            ),
            "b_rl_explore": BundleSpec(
                name="b_rl_explore",
                rank_fraction=0.05,
                base_pools=["rl_entropy", "affective_curiosity"],
                pool_weights=[1.0, 0.8],
                gradient_sources=["entropy_loss"],
                description="Exploration/entropy circuit",
            ),
            "b_rl_anchor": BundleSpec(
                name="b_rl_anchor",
                rank_fraction=0.06,
                base_pools=["rl_kl", "ssl"],
                pool_weights=[1.0, 0.3],
                gradient_sources=["kl_loss"],
                description="KL anchor / trust region circuit",
            ),
        }

    if config_type == BundleConfig.FULL_MODULAR:
        # Many small, specialized bundles
        return {
            # SSL bundles
            "b_ar_predict": BundleSpec(
                name="b_ar_predict",
                rank_fraction=0.08,
                base_pools=["ssl"],
                pool_weights=[1.0],
                gradient_sources=["lm_loss"],
                description="Core next-token prediction",
            ),
            "b_ar_coherence": BundleSpec(
                name="b_ar_coherence",
                rank_fraction=0.05,
                base_pools=["ssl_coherence", "cognitive_attention"],
                pool_weights=[1.0, 0.5],
                gradient_sources=["lm_loss"],
                description="Long-range coherence",
            ),

            # RL bundles - each TRPO/PPO term separate
            "b_rl_policy": BundleSpec(
                name="b_rl_policy",
                rank_fraction=0.06,
                base_pools=["rl_policy"],
                pool_weights=[1.0],
                gradient_sources=["policy_loss"],
                description="Policy gradient",
            ),
            "b_rl_value": BundleSpec(
                name="b_rl_value",
                rank_fraction=0.06,
                base_pools=["rl_value"],
                pool_weights=[1.0],
                gradient_sources=["value_loss"],
                description="Value function",
            ),
            "b_rl_advantage": BundleSpec(
                name="b_rl_advantage",
                rank_fraction=0.04,
                base_pools=["rl_advantage", "rl_policy"],
                pool_weights=[1.0, 0.3],
                gradient_sources=["policy_loss"],
                description="Advantage estimation",
            ),
            "b_rl_entropy": BundleSpec(
                name="b_rl_entropy",
                rank_fraction=0.03,
                base_pools=["rl_entropy"],
                pool_weights=[1.0],
                gradient_sources=["entropy_loss"],
                description="Entropy bonus",
            ),
            "b_rl_kl": BundleSpec(
                name="b_rl_kl",
                rank_fraction=0.04,
                base_pools=["rl_kl"],
                pool_weights=[1.0],
                gradient_sources=["kl_loss"],
                description="KL trust region",
            ),

            # Semantic bundles
            "b_sem_code": BundleSpec(
                name="b_sem_code",
                rank_fraction=0.05,
                base_pools=["semantic_code", "cognitive_reasoning"],
                pool_weights=[1.0, 0.3],
                gradient_sources=["lm_loss"],
                description="Code/programming domain",
            ),
            "b_sem_math": BundleSpec(
                name="b_sem_math",
                rank_fraction=0.05,
                base_pools=["semantic_math", "cognitive_reasoning"],
                pool_weights=[1.0, 0.4],
                gradient_sources=["lm_loss"],
                description="Mathematics domain",
            ),
            "b_sem_natural": BundleSpec(
                name="b_sem_natural",
                rank_fraction=0.05,
                base_pools=["semantic_natural", "cognitive_memory"],
                pool_weights=[1.0, 0.3],
                gradient_sources=["lm_loss"],
                description="Natural language/general domain",
            ),

            # Cognitive process bundles
            "b_cog_reason": BundleSpec(
                name="b_cog_reason",
                rank_fraction=0.06,
                base_pools=["cognitive_reasoning"],
                pool_weights=[1.0],
                gradient_sources=["lm_loss", "policy_loss"],
                description="Logical reasoning",
            ),
            "b_cog_memory": BundleSpec(
                name="b_cog_memory",
                rank_fraction=0.06,
                base_pools=["cognitive_memory"],
                pool_weights=[1.0],
                gradient_sources=["lm_loss"],
                description="Memory/retrieval",
            ),
            "b_cog_plan": BundleSpec(
                name="b_cog_plan",
                rank_fraction=0.05,
                base_pools=["cognitive_planning", "cognitive_attention"],
                pool_weights=[1.0, 0.5],
                gradient_sources=["policy_loss", "value_loss"],
                description="Planning/lookahead",
            ),

            # Error bundles
            "b_err_consistency": BundleSpec(
                name="b_err_consistency",
                rank_fraction=0.03,
                base_pools=["error_consistency", "cognitive_reasoning"],
                pool_weights=[1.0, 0.2],
                gradient_sources=["consistency_loss"],
                description="Self-consistency checking",
            ),
            "b_err_calibration": BundleSpec(
                name="b_err_calibration",
                rank_fraction=0.03,
                base_pools=["error_calibration", "affective_confidence"],
                pool_weights=[1.0, 0.5],
                gradient_sources=["calibration_loss"],
                description="Confidence calibration",
            ),
        }

    if config_type == BundleConfig.AFFECTIVE_SOCIAL:
        # For agentic multi-agent tasks requiring self-awareness
        return {
            # Core SSL/RL (simplified)
            "b_ssl": BundleSpec(
                name="b_ssl",
                rank_fraction=0.12,
                base_pools=["ssl", "cognitive_memory"],
                pool_weights=[1.0, 0.6],
                gradient_sources=["lm_loss"],
                description="Core SSL circuit",
            ),
            "b_rl": BundleSpec(
                name="b_rl",
                rank_fraction=0.10,
                base_pools=["rl_policy", "rl_value", "cognitive_planning"],
                pool_weights=[1.0, 1.0, 0.5],
                gradient_sources=["policy_loss", "value_loss"],
                description="Core RL circuit",
            ),

            # Affective bundles
            "b_confidence": BundleSpec(
                name="b_confidence",
                rank_fraction=0.04,
                base_pools=["affective_confidence", "error_calibration"],
                pool_weights=[1.0, 0.7],
                gradient_sources=["calibration_loss", "policy_loss"],
                description="Certainty/uncertainty modeling",
            ),
            "b_curiosity": BundleSpec(
                name="b_curiosity",
                rank_fraction=0.04,
                base_pools=["affective_curiosity", "rl_entropy"],
                pool_weights=[1.0, 0.8],
                gradient_sources=["entropy_loss", "curiosity_loss"],
                description="Exploration drive",
            ),
            "b_frustration": BundleSpec(
                name="b_frustration",
                rank_fraction=0.03,
                base_pools=["affective_frustration", "cognitive_planning"],
                pool_weights=[1.0, 0.4],
                gradient_sources=["policy_loss"],
                description="Strategy switching signal",
            ),
            "b_patience": BundleSpec(
                name="b_patience",
                rank_fraction=0.03,
                base_pools=["affective_patience", "rl_value"],
                pool_weights=[1.0, 0.5],
                gradient_sources=["value_loss"],
                description="Temporal discounting control",
            ),
            "b_caution": BundleSpec(
                name="b_caution",
                rank_fraction=0.03,
                base_pools=["affective_caution", "rl_policy"],
                pool_weights=[1.0, 0.4],
                gradient_sources=["policy_loss"],
                description="Risk sensitivity",
            ),

            # Social/multi-agent bundles
            "b_self_model": BundleSpec(
                name="b_self_model",
                rank_fraction=0.05,
                base_pools=["social_self_model", "cognitive_memory", "affective_confidence"],
                pool_weights=[1.0, 0.5, 0.4],
                gradient_sources=["lm_loss", "policy_loss"],
                description="Internal self-representation",
            ),
            "b_other_model": BundleSpec(
                name="b_other_model",
                rank_fraction=0.06,
                base_pools=["social_other_model", "cognitive_reasoning", "cognitive_abstraction"],
                pool_weights=[1.0, 0.5, 0.4],
                gradient_sources=["lm_loss", "policy_loss"],
                description="Theory of mind / other agents",
            ),
            "b_trust": BundleSpec(
                name="b_trust",
                rank_fraction=0.03,
                base_pools=["social_trust", "social_other_model", "cognitive_memory"],
                pool_weights=[1.0, 0.5, 0.3],
                gradient_sources=["policy_loss"],
                description="Trust/reputation tracking",
            ),
            "b_cooperation": BundleSpec(
                name="b_cooperation",
                rank_fraction=0.04,
                base_pools=["social_cooperation", "cognitive_planning", "social_other_model"],
                pool_weights=[1.0, 0.5, 0.4],
                gradient_sources=["policy_loss", "cooperation_loss"],
                description="Collaborative goal structures",
            ),
        }

    if config_type == BundleConfig.FEEDBACK_ISOLATED:
        # Only error/feedback bundles isolated, rest shared
        return {
            "b_main": BundleSpec(
                name="b_main",
                rank_fraction=0.25,
                base_pools=["ssl", "rl_policy", "rl_value", "cognitive_reasoning", "cognitive_memory"],
                pool_weights=[1.0, 0.8, 0.8, 1.0, 1.0],
                gradient_sources=["lm_loss", "policy_loss", "value_loss"],
                description="Main shared circuit",
            ),
            "b_err_factual": BundleSpec(
                name="b_err_factual",
                rank_fraction=0.04,
                base_pools=["error_factual", "cognitive_memory"],
                pool_weights=[1.0, 0.3],
                gradient_sources=["factual_loss"],
                description="Factual error correction - isolated",
            ),
            "b_err_logical": BundleSpec(
                name="b_err_logical",
                rank_fraction=0.04,
                base_pools=["error_logical", "cognitive_reasoning"],
                pool_weights=[1.0, 0.3],
                gradient_sources=["logical_loss"],
                description="Logical error correction - isolated",
            ),
            "b_err_consistency": BundleSpec(
                name="b_err_consistency",
                rank_fraction=0.04,
                base_pools=["error_consistency"],
                pool_weights=[1.0],
                gradient_sources=["consistency_loss"],
                description="Consistency error - isolated",
            ),
            "b_err_calibration": BundleSpec(
                name="b_err_calibration",
                rank_fraction=0.03,
                base_pools=["error_calibration", "affective_confidence"],
                pool_weights=[1.0, 0.5],
                gradient_sources=["calibration_loss"],
                description="Calibration error - isolated",
            ),
        }

    if config_type == BundleConfig.HIERARCHICAL:
        # Semantic bundles nested inside cognitive bundles
        return {
            # Top-level cognitive
            "b_cog_memory": BundleSpec(
                name="b_cog_memory",
                rank_fraction=0.15,
                base_pools=["cognitive_memory", "ssl"],
                pool_weights=[1.0, 0.6],
                gradient_sources=["lm_loss"],
                description="Top-level memory circuit",
            ),
            "b_cog_reason": BundleSpec(
                name="b_cog_reason",
                rank_fraction=0.12,
                base_pools=["cognitive_reasoning", "cognitive_abstraction"],
                pool_weights=[1.0, 0.7],
                gradient_sources=["lm_loss", "policy_loss"],
                description="Top-level reasoning circuit",
            ),

            # Semantic nested in memory
            "b_sem_code": BundleSpec(
                name="b_sem_code",
                rank_fraction=0.05,
                base_pools=["semantic_code", "cognitive_memory", "cognitive_reasoning"],
                pool_weights=[1.0, 0.6, 0.4],
                gradient_sources=["lm_loss"],
                description="Code - nested in memory+reasoning",
            ),
            "b_sem_math": BundleSpec(
                name="b_sem_math",
                rank_fraction=0.05,
                base_pools=["semantic_math", "cognitive_memory", "cognitive_reasoning"],
                pool_weights=[1.0, 0.4, 0.6],
                gradient_sources=["lm_loss"],
                description="Math - nested in memory+reasoning",
            ),
            "b_sem_natural": BundleSpec(
                name="b_sem_natural",
                rank_fraction=0.06,
                base_pools=["semantic_natural", "cognitive_memory"],
                pool_weights=[1.0, 0.7],
                gradient_sources=["lm_loss"],
                description="Natural language - nested in memory",
            ),

            # RL separate
            "b_rl": BundleSpec(
                name="b_rl",
                rank_fraction=0.10,
                base_pools=["rl_policy", "rl_value", "cognitive_planning"],
                pool_weights=[1.0, 1.0, 0.5],
                gradient_sources=["policy_loss", "value_loss", "entropy_loss"],
                description="RL circuit - separate from semantic hierarchy",
            ),
        }

    if config_type == BundleConfig.COGNITIVE_SEMANTIC_SPLIT:
        return {
            # Pure cognitive process bundles
            "b_cog_reason": BundleSpec(
                name="b_cog_reason",
                rank_fraction=0.10,
                base_pools=["cognitive_reasoning"],
                pool_weights=[1.0],
                gradient_sources=["lm_loss", "policy_loss"],
                description="Pure reasoning process",
            ),
            "b_cog_memory": BundleSpec(
                name="b_cog_memory",
                rank_fraction=0.10,
                base_pools=["cognitive_memory"],
                pool_weights=[1.0],
                gradient_sources=["lm_loss"],
                description="Pure memory process",
            ),
            "b_cog_attention": BundleSpec(
                name="b_cog_attention",
                rank_fraction=0.06,
                base_pools=["cognitive_attention"],
                pool_weights=[1.0],
                gradient_sources=["lm_loss"],
                description="Pure attention process",
            ),
            "b_cog_plan": BundleSpec(
                name="b_cog_plan",
                rank_fraction=0.08,
                base_pools=["cognitive_planning"],
                pool_weights=[1.0],
                gradient_sources=["policy_loss", "value_loss"],
                description="Pure planning process",
            ),

            # Pure semantic domain bundles
            "b_sem_code": BundleSpec(
                name="b_sem_code",
                rank_fraction=0.06,
                base_pools=["semantic_code"],
                pool_weights=[1.0],
                gradient_sources=["lm_loss"],
                description="Pure code domain",
            ),
            "b_sem_math": BundleSpec(
                name="b_sem_math",
                rank_fraction=0.06,
                base_pools=["semantic_math"],
                pool_weights=[1.0],
                gradient_sources=["lm_loss"],
                description="Pure math domain",
            ),
            "b_sem_natural": BundleSpec(
                name="b_sem_natural",
                rank_fraction=0.08,
                base_pools=["semantic_natural"],
                pool_weights=[1.0],
                gradient_sources=["lm_loss"],
                description="Pure natural language domain",
            ),

            # SSL/RL
            "b_ssl": BundleSpec(
                name="b_ssl",
                rank_fraction=0.08,
                base_pools=["ssl", "ssl_coherence"],
                pool_weights=[1.0, 0.7],
                gradient_sources=["lm_loss"],
                description="SSL signal circuit",
            ),
            "b_rl": BundleSpec(
                name="b_rl",
                rank_fraction=0.08,
                base_pools=["rl_policy", "rl_value", "rl_entropy"],
                pool_weights=[1.0, 1.0, 0.5],
                gradient_sources=["policy_loss", "value_loss", "entropy_loss"],
                description="RL signal circuit",
            ),
        }

    raise NotImplementedError(f"Config {config_type} not implemented")


def get_all_loss_types() -> List[str]:
    """Return all possible loss types that bundles can receive gradients from."""
    return [
        "lm_loss",
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "kl_loss",
        "consistency_loss",
        "calibration_loss",
        "factual_loss",
        "logical_loss",
        "curiosity_loss",
        "cooperation_loss",
    ]
