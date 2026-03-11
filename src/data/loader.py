"""
Contrastive dataloader for MLX embedding training.
Yields batches of (anchor, positive, negatives) with proper padding.

This file is AGENT-MUTABLE.
"""

from __future__ import annotations

import random

import mlx.core as mx
import numpy as np


def pad_and_convert(token_ids_list: list[list[int]], max_length: int) -> tuple[mx.array, mx.array]:
    """Pad a list of token ID sequences and create attention masks."""
    batch_size = len(token_ids_list)
    input_ids = np.zeros((batch_size, max_length), dtype=np.int32)
    attention_mask = np.zeros((batch_size, max_length), dtype=np.int32)

    for i, ids in enumerate(token_ids_list):
        length = min(len(ids), max_length)
        input_ids[i, :length] = ids[:length]
        attention_mask[i, :length] = 1

    return mx.array(input_ids), mx.array(attention_mask)


class ContrastiveDataLoader:
    """
    Yields batches of contrastive training data.
    Each batch contains (query_ids, query_mask, pos_ids, pos_mask, neg_ids, neg_mask).
    """

    def __init__(
        self,
        triplets: list[dict],
        tokenizer,
        batch_size: int = 128,
        max_length: int = 512,
        shuffle: bool = True,
        source_stratification: bool = False,
    ):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.source_stratification = source_stratification

    def __len__(self):
        return (len(self.triplets) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.triplets)))
        if self.shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            batch = [self.triplets[i] for i in batch_indices]

            queries = [t["query"] for t in batch]
            positives = [t["positive"] for t in batch]

            # Tokenize
            q_enc = self.tokenizer(queries, truncation=True, max_length=self.max_length)
            p_enc = self.tokenizer(positives, truncation=True, max_length=self.max_length)

            q_ids, q_mask = pad_and_convert(q_enc["input_ids"], self.max_length)
            p_ids, p_mask = pad_and_convert(p_enc["input_ids"], self.max_length)

            # Hard negatives (if available)
            neg_ids, neg_mask = None, None
            has_negs = any(t.get("negatives") for t in batch)
            if has_negs:
                negatives = []
                for t in batch:
                    negs = t.get("negatives", [])
                    if negs:
                        negatives.append(negs[0])  # take first hard negative
                    else:
                        negatives.append("")  # will be masked

                n_enc = self.tokenizer(negatives, truncation=True, max_length=self.max_length)
                neg_ids, neg_mask = pad_and_convert(n_enc["input_ids"], self.max_length)

            yield {
                "query_ids": q_ids,
                "query_mask": q_mask,
                "positive_ids": p_ids,
                "positive_mask": p_mask,
                "negative_ids": neg_ids,
                "negative_mask": neg_mask,
            }
