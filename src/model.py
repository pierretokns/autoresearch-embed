"""
ModernBERT encoder + embedding projection, implemented natively in MLX.

Loads weights from answerdotai/ModernBERT-base (safetensors format).
Architecture: 22-layer encoder with alternating local/global attention,
fused QKV projections, GLU MLP, and RoPE.

This file is AGENT-MUTABLE.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModernBERTConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
    intermediate_size: int = 1152
    max_position_embeddings: int = 8192
    vocab_size: int = 50368
    attention_bias: bool = False
    global_attn_every_n_layers: int = 3
    local_attention: int = 128
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0
    embedding_dropout: float = 0.0
    attention_dropout: float = 0.0


def get_model_config(model_id: str) -> ModernBERTConfig:
    """Return the correct ModernBERTConfig for a given model ID."""
    if "large" in model_id.lower():
        return ModernBERTConfig(
            hidden_size=1024,
            num_hidden_layers=28,
            num_attention_heads=16,
            intermediate_size=2624,
        )
    return ModernBERTConfig()


class ModernBERTAttention(nn.Module):
    """Multi-head attention with fused QKV and alternating local/global attention."""

    def __init__(self, config: ModernBERTConfig, layer_idx: int):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.n_heads
        self.hidden_size = config.hidden_size

        # Fused QKV: (3 * hidden_size, hidden_size)
        self.Wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Determine if this layer uses local or global attention
        self.is_global = (layer_idx % config.global_attn_every_n_layers == 0)
        self.local_window = config.local_attention

        # RoPE with different theta for local vs global
        rope_theta = config.global_rope_theta if self.is_global else config.local_rope_theta
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_theta)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, T, _ = x.shape

        # Fused QKV projection
        qkv = self.Wqkv(x)  # (B, T, 3*H)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)

        scale = 1.0 / math.sqrt(self.head_dim)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

        # (B, n_heads, T, head_dim) -> (B, T, hidden_size)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.Wo(out)


class ModernBERTMLP(nn.Module):
    """GLU-style MLP: Wi projects to 2*intermediate, split + gate, Wo projects back."""

    def __init__(self, config: ModernBERTConfig):
        super().__init__()
        # Wi: (hidden_size) -> (2 * intermediate_size) for GLU
        self.Wi = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.Wi(x)
        gate, value = mx.split(x, 2, axis=-1)
        return self.Wo(nn.gelu_approx(gate) * value)


class ModernBERTLayer(nn.Module):
    """Single transformer layer with pre-norm attention and MLP."""

    def __init__(self, config: ModernBERTConfig, layer_idx: int):
        super().__init__()
        self.attn = ModernBERTAttention(config, layer_idx)
        self.mlp = ModernBERTMLP(config)
        # Layer 0 has no attn_norm (embedding norm serves as first attn norm)
        self.attn_norm = nn.RMSNorm(config.hidden_size) if layer_idx > 0 else None
        self.mlp_norm = nn.RMSNorm(config.hidden_size)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        # Pre-norm attention
        residual = x
        if self.attn_norm is not None:
            x = self.attn_norm(x)
        x = self.attn(x, mask=mask)
        x = residual + x

        # Pre-norm MLP
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class ModernBERTEncoder(nn.Module):
    """Full ModernBERT encoder stack."""

    def __init__(self, config: ModernBERTConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_norm = nn.RMSNorm(config.hidden_size)
        self.layers = [ModernBERTLayer(config, i) for i in range(config.num_hidden_layers)]
        self.final_norm = nn.RMSNorm(config.hidden_size)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None,
                 return_all_layers: bool = False):
        x = self.tok_embeddings(input_ids)
        x = self.embedding_norm(x)

        # Build attention mask for local attention layers
        if attention_mask is not None:
            # Convert (B, T) binary mask to additive mask (B, 1, 1, T) for broadcasting
            mask = mx.where(attention_mask[:, None, None, :] == 0,
                            mx.array(float("-inf")), mx.array(0.0))
        else:
            mask = None

        if return_all_layers:
            all_hidden = []
            for layer in self.layers:
                x = layer(x, mask=mask)
                all_hidden.append(x)
            # Apply final norm only to last layer
            all_hidden[-1] = self.final_norm(all_hidden[-1])
            return all_hidden  # list of (B, T, H), length = num_layers
        else:
            for layer in self.layers:
                x = layer(x, mask=mask)
            x = self.final_norm(x)
            return x


class LayerWeightedPooling(nn.Module):
    """
    Multi-layer weighted pooling: learn a softmax-weighted combination of CLS tokens
    from all encoder layers. Each layer captures different semantic/syntactic info.
    Research shows middle layers often outperform last layer for STS tasks.
    """

    def __init__(self, num_layers: int):
        super().__init__()
        # Learnable logits for each layer: softmax → weights
        self.layer_logits = mx.zeros((num_layers,))

    def __call__(self, all_hidden: list) -> mx.array:
        # all_hidden: list of (B, T, H), length = num_layers
        # Extract CLS token from each layer
        cls_per_layer = mx.stack([h[:, 0] for h in all_hidden], axis=1)  # (B, L, H)
        # Compute softmax weights over layers
        weights = mx.softmax(self.layer_logits, axis=0)  # (L,)
        # Weighted sum: (B, L, H) * (L,) → (B, H)
        pooled = mx.sum(cls_per_layer * weights[None, :, None], axis=1)
        return pooled


class LatentAttentionPooling(nn.Module):
    """
    Latent attention pooling: trainable query vectors attend over encoder output.
    Used by NV-Embed-v2 (MTEB #1). A set of K latent queries cross-attend the
    token representations; the resulting K vectors are mean-pooled to one embedding.
    """

    def __init__(self, hidden_size: int, num_latents: int = 4, num_heads: int = 8):
        super().__init__()
        self.num_latents = num_latents
        self.hidden_size = hidden_size
        # Trainable latent queries: (1, K, H)
        self.latents = mx.zeros((1, num_latents, hidden_size))
        self.attn = nn.MultiHeadAttention(hidden_size, num_heads, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def __call__(self, hidden: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        B = hidden.shape[0]
        # Expand latents to batch
        queries = mx.broadcast_to(self.latents, (B, self.num_latents, self.hidden_size))
        # Key/value mask: additive mask (B, 1, K, T) from attention_mask
        if attention_mask is not None:
            kv_mask = mx.where(attention_mask[:, None, None, :] == 0,
                               mx.array(float("-inf")), mx.array(0.0))
        else:
            kv_mask = None
        # Cross-attention: queries attend over hidden states
        out = self.attn(queries, hidden, hidden, mask=kv_mask)  # (B, K, H)
        out = self.norm(out)
        # Mean-pool over latents → (B, H)
        return mx.mean(out, axis=1)


class EmbeddingModel(nn.Module):
    """Wraps ModernBERT encoder with pooling, optional projection, and L2 normalization."""

    def __init__(self, config: ModernBERTConfig, projection_dim: int | None = None,
                 pooling: str = "mean"):
        super().__init__()
        self.encoder = ModernBERTEncoder(config)
        self.pooling = pooling
        self.hidden_size = config.hidden_size

        # Pooling modules (initialized if needed)
        if pooling == "latent_attn":
            self.latent_pool = LatentAttentionPooling(config.hidden_size)
        else:
            self.latent_pool = None
        if pooling == "layer_weighted":
            self.layer_pool = LayerWeightedPooling(config.num_hidden_layers)
        else:
            self.layer_pool = None

        # cls_mean: concatenate CLS + mean → project to output_dim
        input_dim = config.hidden_size * 2 if pooling == "cls_mean" else config.hidden_size
        out_dim = projection_dim if projection_dim else config.hidden_size
        if projection_dim and (projection_dim != config.hidden_size or pooling == "cls_mean"):
            self.projection = nn.Linear(input_dim, out_dim, bias=False)
            self.output_dim = out_dim
        else:
            self.projection = None
            self.output_dim = config.hidden_size

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        if self.pooling == "layer_weighted":
            all_hidden = self.encoder(input_ids, attention_mask, return_all_layers=True)
            pooled = self.layer_pool(all_hidden)
        else:
            hidden = self.encoder(input_ids, attention_mask)  # (B, T, H)
            if self.pooling == "latent_attn":
                pooled = self.latent_pool(hidden, attention_mask)
            elif self.pooling == "cls":
                pooled = hidden[:, 0]
            elif self.pooling == "cls_mean":
                cls_vec = hidden[:, 0]
                if attention_mask is not None:
                    mask = attention_mask[:, :, None].astype(hidden.dtype)
                    mean_vec = mx.sum(hidden * mask, axis=1) / mx.maximum(mx.sum(mask, axis=1), 1e-9)
                else:
                    mean_vec = mx.mean(hidden, axis=1)
                pooled = mx.concatenate([cls_vec, mean_vec], axis=-1)
            else:  # mean pooling
                if attention_mask is not None:
                    mask = attention_mask[:, :, None].astype(hidden.dtype)
                    pooled = mx.sum(hidden * mask, axis=1) / mx.maximum(mx.sum(mask, axis=1), 1e-9)
                else:
                    pooled = mx.mean(hidden, axis=1)

        if self.projection is not None:
            pooled = self.projection(pooled)

        # L2 normalize
        norms = mx.sqrt(mx.sum(pooled * pooled, axis=-1, keepdims=True) + 1e-12)
        pooled = pooled / norms
        return pooled

    def encode_sentences(self, sentences: list[str], tokenizer, batch_size: int = 64,
                         max_length: int = 512) -> "numpy.ndarray":
        """MTEB-compatible encode method. Returns numpy array of L2-normalized embeddings."""
        import numpy as np

        n = len(sentences)
        if n == 0:
            return np.zeros((0, self.output_dim), dtype=np.float32)

        # Pre-allocate output to avoid list accumulation + concatenation memory spike
        out = np.empty((n, self.output_dim), dtype=np.float32)
        for i in range(0, n, batch_size):
            batch = sentences[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="np")
            input_ids = mx.array(enc["input_ids"])
            attention_mask = mx.array(enc["attention_mask"])
            emb = self(input_ids, attention_mask)
            mx.eval(emb)
            out[i:i + len(batch)] = np.array(emb, copy=False)
            del input_ids, attention_mask, emb

        return out


def load_from_safetensors(model: EmbeddingModel, model_id: str = "answerdotai/ModernBERT-base"):
    """Load pretrained weights from HuggingFace safetensors into our MLX model."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(model_id, "model.safetensors")

    weights = {}
    with safe_open(path, framework="numpy") as f:
        for key in f.keys():
            weights[key] = mx.array(f.get_tensor(key))

    # Map HF weight names to our MLX model structure
    mapped = {}

    # Embeddings
    mapped["encoder.tok_embeddings.weight"] = weights["model.embeddings.tok_embeddings.weight"]
    mapped["encoder.embedding_norm.weight"] = weights["model.embeddings.norm.weight"]
    mapped["encoder.final_norm.weight"] = weights["model.final_norm.weight"]

    # Layers
    for i in range(model.encoder.config.num_hidden_layers):
        prefix_src = f"model.layers.{i}"
        prefix_dst = f"encoder.layers.{i}"

        # Attention
        mapped[f"{prefix_dst}.attn.Wqkv.weight"] = weights[f"{prefix_src}.attn.Wqkv.weight"]
        mapped[f"{prefix_dst}.attn.Wo.weight"] = weights[f"{prefix_src}.attn.Wo.weight"]

        # MLP
        mapped[f"{prefix_dst}.mlp.Wi.weight"] = weights[f"{prefix_src}.mlp.Wi.weight"]
        mapped[f"{prefix_dst}.mlp.Wo.weight"] = weights[f"{prefix_src}.mlp.Wo.weight"]

        # Norms
        if i > 0:
            mapped[f"{prefix_dst}.attn_norm.weight"] = weights[f"{prefix_src}.attn_norm.weight"]
        mapped[f"{prefix_dst}.mlp_norm.weight"] = weights[f"{prefix_src}.mlp_norm.weight"]

    # Load into model (strict=False: projection layer is randomly initialized)
    model.load_weights(list(mapped.items()), strict=False)

    print(f"Loaded {len(mapped)} weight tensors from {model_id}")
    return model
