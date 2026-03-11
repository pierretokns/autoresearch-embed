"""
Encoder model for embedding training on MLX.
Wraps a pretrained encoder (ModernBERT-base or similar) and adds pooling + projection.

This file is AGENT-MUTABLE: the agent can change pooling strategy, projection dimensions,
layer selection, normalization, etc.
"""

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer


class ProjectionHead(nn.Module):
    """Optional linear projection from encoder hidden size to embedding dim."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


class EmbeddingModel(nn.Module):
    """
    Wraps a pretrained encoder with pooling and optional projection.

    The encode() method is the main interface:
        tokens (dict with input_ids, attention_mask) -> normalized embeddings
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        projection_dim: int | None = None,
        pooling: str = "mean",  # "mean", "cls", "weighted"
        normalize: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.pooling = pooling
        self.normalize = normalize

        if projection_dim and projection_dim != hidden_size:
            self.projection = ProjectionHead(hidden_size, projection_dim)
            self.output_dim = projection_dim
        else:
            self.projection = None
            self.output_dim = hidden_size

        if pooling == "weighted":
            self.layer_weights = mx.ones(1)  # learnable scalar per layer (simplified)

    def pool(self, hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
        """Pool token embeddings into a single vector."""
        if self.pooling == "cls":
            return hidden_states[:, 0]
        elif self.pooling == "mean":
            mask_expanded = mx.expand_dims(attention_mask, axis=-1)  # (B, T, 1)
            masked = hidden_states * mask_expanded
            summed = mx.sum(masked, axis=1)  # (B, H)
            counts = mx.maximum(mx.sum(mask_expanded, axis=1), mx.array(1e-9))  # (B, 1)
            return summed / counts
        elif self.pooling == "weighted":
            # Weighted mean with learnable temperature
            weights = mx.softmax(self.layer_weights * attention_mask, axis=1)
            return mx.sum(hidden_states * mx.expand_dims(weights, -1), axis=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        """Forward pass: encode tokens to embeddings."""
        hidden_states = self.encoder(input_ids, attention_mask=attention_mask)

        # Handle models that return tuples/dicts
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        elif isinstance(hidden_states, dict):
            hidden_states = hidden_states.get("last_hidden_state", hidden_states)

        pooled = self.pool(hidden_states, attention_mask)

        if self.projection is not None:
            pooled = self.projection(pooled)

        if self.normalize:
            pooled = pooled / mx.maximum(
                mx.sqrt(mx.sum(pooled * pooled, axis=-1, keepdims=True)),
                mx.array(1e-12),
            )

        return pooled

    def encode_sentences(self, sentences: list[str], tokenizer, batch_size: int = 64, max_length: int = 512) -> np.ndarray:
        """
        Encode a list of sentences to numpy embeddings.
        This is the interface MTEB expects.
        """
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )
            input_ids = mx.array(encoded["input_ids"])
            attention_mask = mx.array(encoded["attention_mask"])
            embeddings = self(input_ids, attention_mask)
            mx.eval(embeddings)
            all_embeddings.append(np.array(embeddings))
        return np.concatenate(all_embeddings, axis=0)
