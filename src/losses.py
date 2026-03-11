"""
Contrastive loss functions implemented in MLX.

This file is AGENT-MUTABLE: the agent can add new loss functions, change temperature
scaling, modify negative weighting, etc.
"""

import mlx.core as mx


def infonce_loss(
    query_emb: mx.array,
    positive_emb: mx.array,
    negative_emb: mx.array | None = None,
    temperature: float = 0.05,
    hard_neg_weight: float = 1.0,
) -> mx.array:
    """
    InfoNCE contrastive loss with in-batch negatives and optional explicit hard negatives.

    Args:
        query_emb: (B, D) normalized query embeddings
        positive_emb: (B, D) normalized positive embeddings
        negative_emb: optional (B, K, D) explicit hard negatives per query
        temperature: similarity scaling temperature
        hard_neg_weight: weight multiplier for hard negative logits
    Returns:
        scalar loss
    """
    B = query_emb.shape[0]

    # In-batch similarity matrix: (B, B)
    sim_matrix = mx.matmul(query_emb, positive_emb.T) / temperature

    if negative_emb is not None:
        # Hard negative similarities: (B, K)
        # query_emb: (B, D) -> (B, 1, D)
        # negative_emb: (B, K, D)
        hard_neg_sim = mx.sum(
            mx.expand_dims(query_emb, axis=1) * negative_emb, axis=-1
        ) / temperature
        hard_neg_sim = hard_neg_sim * hard_neg_weight

        # Concatenate: (B, B+K) — in-batch + hard negatives
        logits = mx.concatenate([sim_matrix, hard_neg_sim], axis=1)
    else:
        logits = sim_matrix

    # Labels: diagonal entries are the positives (index i for query i)
    labels = mx.arange(B)

    # Cross-entropy loss
    log_softmax = logits - mx.logsumexp(logits, axis=1, keepdims=True)
    loss = -mx.mean(log_softmax[mx.arange(B), labels])

    return loss


def multiple_negatives_ranking_loss(
    query_emb: mx.array,
    positive_emb: mx.array,
    scale: float = 20.0,
) -> mx.array:
    """
    Multiple Negatives Ranking Loss (sentence-transformers style).
    Uses all other positives in the batch as negatives.

    Args:
        query_emb: (B, D) normalized query embeddings
        positive_emb: (B, D) normalized positive embeddings
        scale: similarity scaling factor
    Returns:
        scalar loss
    """
    scores = mx.matmul(query_emb, positive_emb.T) * scale
    labels = mx.arange(scores.shape[0])
    log_softmax = scores - mx.logsumexp(scores, axis=1, keepdims=True)
    loss = -mx.mean(log_softmax[mx.arange(scores.shape[0]), labels])
    return loss


def triplet_loss(
    anchor: mx.array,
    positive: mx.array,
    negative: mx.array,
    margin: float = 0.2,
) -> mx.array:
    """
    Triplet margin loss.

    Args:
        anchor: (B, D) anchor embeddings
        positive: (B, D) positive embeddings
        negative: (B, D) negative embeddings (one per anchor)
        margin: margin threshold
    Returns:
        scalar loss
    """
    pos_dist = mx.sqrt(mx.sum((anchor - positive) ** 2, axis=-1) + 1e-12)
    neg_dist = mx.sqrt(mx.sum((anchor - negative) ** 2, axis=-1) + 1e-12)
    loss = mx.maximum(pos_dist - neg_dist + margin, mx.array(0.0))
    return mx.mean(loss)
