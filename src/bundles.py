"""Bundle adapter modules - low-rank adapters operating in overlapping subspaces."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .config import BASE_POOLS, BundleSpec


class SharedBases(nn.Module):
    """
    Shared basis pools that bundles mix to create their subspaces.

    This is the key to overlapping subspaces - all bundles draw from
    these shared directions, but mix them differently via their W_k matrices.
    """

    def __init__(self, d_model: int, base_pool_fractions: Optional[Dict[str, float]] = None):
        super().__init__()
        self.d_model = d_model
        self.pool_fractions = base_pool_fractions or BASE_POOLS

        # Create basis matrices for each pool
        # U_pool: [d_model, pool_rank]
        self.bases = nn.ParameterDict()
        self.pool_ranks = {}

        for pool_name, fraction in self.pool_fractions.items():
            rank = max(8, int(d_model * fraction))
            self.pool_ranks[pool_name] = rank
            # Initialize with orthogonal directions (approximately)
            basis = torch.randn(d_model, rank)
            basis = F.normalize(basis, dim=0)  # Normalize columns
            self.bases[pool_name] = nn.Parameter(basis * 0.1)

    def get_basis(self, pool_name: str) -> torch.Tensor:
        """Get the basis matrix for a specific pool."""
        return self.bases[pool_name]

    def get_concatenated_bases(self, pool_names: List[str]) -> torch.Tensor:
        """Concatenate multiple basis matrices."""
        return torch.cat([self.bases[name] for name in pool_names], dim=1)

    def compute_overlap_matrix(self) -> torch.Tensor:
        """
        Compute pairwise overlap between all basis pools.
        Returns: [num_pools, num_pools] matrix of cosine similarities.
        """
        pool_names = list(self.bases.keys())
        n = len(pool_names)
        overlap = torch.zeros(n, n, device=next(self.parameters()).device)

        for i, name_i in enumerate(pool_names):
            for j, name_j in enumerate(pool_names):
                if i <= j:
                    U_i = self.bases[name_i]  # [d, r_i]
                    U_j = self.bases[name_j]  # [d, r_j]
                    # Compute subspace similarity via principal angles
                    # Simplified: use mean of absolute dot products
                    U_i_norm = F.normalize(U_i, dim=0)
                    U_j_norm = F.normalize(U_j, dim=0)
                    dots = torch.abs(U_i_norm.T @ U_j_norm)  # [r_i, r_j]
                    overlap[i, j] = dots.mean()
                    overlap[j, i] = overlap[i, j]

        return overlap


class BundleAdapter(nn.Module):
    """
    Low-rank adapter operating in a specific subspace defined by mixing shared bases.

    The subspace is defined by: P_k = W_k @ [U_pool1; U_pool2; ...]^T
    where W_k learns how to mix the shared basis directions for this bundle.
    """

    def __init__(
        self,
        spec: BundleSpec,
        shared_bases: SharedBases,
        d_model: int,
        bottleneck_ratio: float = 0.5,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.spec = spec
        self.d_model = d_model
        self.rank = spec.get_rank(d_model)
        self.gradient_sources = set(spec.gradient_sources)

        # Compute total dimension of concatenated bases
        total_base_dim = sum(
            shared_bases.pool_ranks[pool] for pool in spec.base_pools
        )

        # Mixing matrix: learns how to combine shared bases into this bundle's subspace
        # W_k: [bundle_rank, total_base_dim]
        self.base_mixer = nn.Parameter(
            torch.randn(self.rank, total_base_dim) * init_scale
        )

        # Apply pool weights during initialization (baked in but can drift)
        self._apply_pool_weights(shared_bases, spec.pool_weights)

        # Store which bases we use (names)
        self.base_pool_names = spec.base_pools

        # Bottleneck adapter: rank -> bottleneck -> rank
        bottleneck_dim = max(8, int(self.rank * bottleneck_ratio))
        self.adapter = nn.Sequential(
            nn.Linear(self.rank, bottleneck_dim, bias=False),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.rank, bias=False),
        )

        # Initialize adapter to near-identity (small perturbation)
        nn.init.zeros_(self.adapter[0].weight)
        nn.init.zeros_(self.adapter[2].weight)

        # Output scale (learnable, starts small)
        self.out_scale = nn.Parameter(torch.ones(1) * 0.1)

    def _apply_pool_weights(self, shared_bases: SharedBases, pool_weights: List[float]):
        """Apply pool weights to the mixing matrix initialization."""
        offset = 0
        with torch.no_grad():
            for pool_name, weight in zip(self.spec.base_pools, pool_weights):
                pool_rank = shared_bases.pool_ranks[pool_name]
                self.base_mixer[:, offset:offset + pool_rank] *= weight
                offset += pool_rank

    def get_projection_matrices(
        self, shared_bases: SharedBases
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute P (project into bundle) and Q (project back) dynamically.

        P = W @ U_concat^T : [rank, d_model] - projects d_model -> rank
        Q = U_concat @ W^T : [d_model, rank] - projects rank -> d_model
        """
        # Concatenate relevant bases
        U_concat = shared_bases.get_concatenated_bases(self.base_pool_names)  # [d, total_base_dim]

        # P = W @ U^T : [rank, d]
        P = self.base_mixer @ U_concat.T

        # Q = U @ W^T : [d, rank]
        Q = U_concat @ self.base_mixer.T

        return P, Q

    def forward(
        self,
        h: torch.Tensor,
        shared_bases: SharedBases,
    ) -> torch.Tensor:
        """
        Apply bundle transformation.

        Args:
            h: [batch, seq, d_model] - input hidden states
            shared_bases: SharedBases module

        Returns:
            residual: [batch, seq, d_model] - bundle contribution to add
        """
        P, Q = self.get_projection_matrices(shared_bases)

        # Project into bundle subspace: [batch, seq, rank]
        z = torch.einsum("bsd,rd->bsr", h, P)

        # Bundle-specific transform
        u = self.adapter(z)  # [batch, seq, rank]

        # Project back to model space: [batch, seq, d_model]
        r = torch.einsum("bsr,dr->bsd", u, Q)

        return r * self.out_scale

    def should_receive_gradient(self, loss_type: str) -> bool:
        """Check if this bundle should receive gradients from the given loss type."""
        if "all" in self.gradient_sources:
            return True
        return loss_type in self.gradient_sources


class BundleLayer(nn.Module):
    """
    Wraps bundle adapters for a single transformer layer.
    Applies all bundles in parallel and sums with learned routing weights.
    """

    def __init__(
        self,
        bundles: Dict[str, BundleAdapter],
        shared_bases: SharedBases,
        alpha_init: float = 0.5,
    ):
        super().__init__()
        self.bundles = nn.ModuleDict(bundles)
        self.shared_bases = shared_bases

        # Learnable routing weights (one per bundle, soft-gated)
        self.alpha_logits = nn.ParameterDict({
            name: nn.Parameter(torch.tensor([alpha_init]).logit())
            for name in bundles.keys()
        })

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply all bundles and sum their contributions.

        Args:
            h: [batch, seq, d_model] - output from base transformer layer

        Returns:
            h_out: [batch, seq, d_model] - h + weighted bundle contributions
        """
        bundle_sum = torch.zeros_like(h)

        for name, bundle in self.bundles.items():
            alpha = torch.sigmoid(self.alpha_logits[name])
            r = bundle(h, self.shared_bases)
            bundle_sum = bundle_sum + alpha * r

        return h + bundle_sum

    def get_alphas(self) -> Dict[str, float]:
        """Get current routing weights as floats."""
        return {
            name: torch.sigmoid(logit).item()
            for name, logit in self.alpha_logits.items()
        }

    def get_bundle_activations(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get per-bundle activations for analysis."""
        activations = {}
        for name, bundle in self.bundles.items():
            P, Q = bundle.get_projection_matrices(self.shared_bases)
            z = torch.einsum("bsd,rd->bsr", h, P)
            activations[name] = z.detach()
        return activations


class BundledModel(nn.Module):
    """
    Wraps a pretrained transformer with bundle adapters at each layer.

    Supports HuggingFace models (GPT2, OPT, Llama, Mistral, etc.)
    """

    def __init__(
        self,
        base_model: nn.Module,
        bundle_specs: Dict[str, BundleSpec],
        shared_bases: SharedBases,
        alpha_init: float = 0.5,
        bottleneck_ratio: float = 0.5,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.shared_bases = shared_bases
        self.bundle_specs = bundle_specs

        # Detect model architecture and get dimensions
        self.config = base_model.config
        self.d_model = self._get_hidden_size()
        self.num_layers = self._get_num_layers()

        # Freeze base model if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Create bundle layers (one set of bundles per transformer layer)
        self.bundle_layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            layer_bundles = {}
            for name, spec in bundle_specs.items():
                layer_bundles[name] = BundleAdapter(
                    spec=spec,
                    shared_bases=shared_bases,
                    d_model=self.d_model,
                    bottleneck_ratio=bottleneck_ratio,
                )
            self.bundle_layers.append(
                BundleLayer(layer_bundles, shared_bases, alpha_init)
            )

        # Register hooks to inject bundle computation after each layer
        self._register_hooks()

        # Optional: value head for RL
        self.value_head = None

    def _get_hidden_size(self) -> int:
        """Get hidden size from model config."""
        if hasattr(self.config, "hidden_size"):
            return self.config.hidden_size
        if hasattr(self.config, "n_embd"):
            return self.config.n_embd
        if hasattr(self.config, "d_model"):
            return self.config.d_model
        raise ValueError("Cannot determine hidden size from model config")

    def _get_num_layers(self) -> int:
        """Get number of layers from model config."""
        if hasattr(self.config, "num_hidden_layers"):
            return self.config.num_hidden_layers
        if hasattr(self.config, "n_layer"):
            return self.config.n_layer
        if hasattr(self.config, "num_layers"):
            return self.config.num_layers
        raise ValueError("Cannot determine number of layers from model config")

    def _get_transformer_layers(self) -> nn.ModuleList:
        """Get the list of transformer layers from the model."""
        # GPT2
        if hasattr(self.base_model, "transformer"):
            if hasattr(self.base_model.transformer, "h"):
                return self.base_model.transformer.h
        # OPT, Llama, Mistral
        if hasattr(self.base_model, "model"):
            if hasattr(self.base_model.model, "layers"):
                return self.base_model.model.layers
            if hasattr(self.base_model.model, "decoder"):
                if hasattr(self.base_model.model.decoder, "layers"):
                    return self.base_model.model.decoder.layers
        raise ValueError("Cannot find transformer layers in model architecture")

    def _register_hooks(self):
        """Register forward hooks to apply bundles after each transformer layer."""
        self._hooks = []
        try:
            layers = self._get_transformer_layers()
        except ValueError:
            # Fallback: no hooks, will use manual forward
            return

        for layer_idx, layer in enumerate(layers):
            hook = layer.register_forward_hook(
                self._make_bundle_hook(layer_idx)
            )
            self._hooks.append(hook)

    def _make_bundle_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook(module, args, output):
            # Output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            # Apply bundle layer
            hidden_states = self.bundle_layers[layer_idx](hidden_states)

            if rest is not None:
                return (hidden_states,) + rest
            return hidden_states

        return hook

    def add_value_head(self):
        """Add a value head for RL training."""
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass through base model with bundle injection via hooks."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        return outputs

    def get_all_alphas(self) -> Dict[int, Dict[str, float]]:
        """Get routing weights for all layers."""
        return {
            layer_idx: layer.get_alphas()
            for layer_idx, layer in enumerate(self.bundle_layers)
        }

    def get_bundle_gradient_norms(self) -> Dict[str, float]:
        """Get gradient norms per bundle (aggregated across layers)."""
        norms = {}
        for layer_idx, bundle_layer in enumerate(self.bundle_layers):
            for name, bundle in bundle_layer.bundles.items():
                key = f"{name}"
                if key not in norms:
                    norms[key] = 0.0
                for param in bundle.parameters():
                    if param.grad is not None:
                        norms[key] += param.grad.norm().item() ** 2
        # Take sqrt to get L2 norm
        return {k: v ** 0.5 for k, v in norms.items()}

    def get_trainable_parameters(self):
        """Get only the trainable parameters (bundles + shared bases)."""
        params = []
        params.extend(self.shared_bases.parameters())
        for bundle_layer in self.bundle_layers:
            params.extend(bundle_layer.parameters())
        if self.value_head is not None:
            params.extend(self.value_head.parameters())
        return params

    def num_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_parameters())

    def num_total_params(self) -> int:
        """Count total parameters including frozen base."""
        return sum(p.numel() for p in self.parameters())


def wrap_model_with_bundles(
    base_model: nn.Module,
    bundle_specs: Dict[str, BundleSpec],
    alpha_init: float = 0.5,
    bottleneck_ratio: float = 0.5,
    freeze_base: bool = True,
    base_pool_fractions: Optional[Dict[str, float]] = None,
) -> BundledModel:
    """
    Convenience function to wrap a HuggingFace model with bundles.

    Args:
        base_model: Pretrained HuggingFace model
        bundle_specs: Dictionary of bundle specifications
        alpha_init: Initial routing weight
        bottleneck_ratio: Ratio of bottleneck dim to bundle rank
        freeze_base: Whether to freeze the base model
        base_pool_fractions: Override default base pool fractions

    Returns:
        BundledModel wrapping the base model
    """
    # Determine d_model from base model
    config = base_model.config
    if hasattr(config, "hidden_size"):
        d_model = config.hidden_size
    elif hasattr(config, "n_embd"):
        d_model = config.n_embd
    elif hasattr(config, "d_model"):
        d_model = config.d_model
    else:
        raise ValueError("Cannot determine d_model from model config")

    # Create shared bases
    shared_bases = SharedBases(d_model, base_pool_fractions)

    # Wrap model
    return BundledModel(
        base_model=base_model,
        bundle_specs=bundle_specs,
        shared_bases=shared_bases,
        alpha_init=alpha_init,
        bottleneck_ratio=bottleneck_ratio,
        freeze_base=freeze_base,
    )
