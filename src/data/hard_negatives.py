"""
Iterative hard negative mining.
Embeds the training corpus with the current model, finds near-miss negatives.

This file is AGENT-MUTABLE.
"""

from __future__ import annotations

import numpy as np


def mine_hard_negatives(
    model,
    tokenizer,
    triplets: list[dict],
    top_k: int = 7,
    batch_size: int = 512,
    max_length: int = 512,
    min_similarity: float = 0.3,
) -> list[dict]:
    """
    Mine hard negatives for each query using the current model.

    1. Embed all queries and all positives/documents.
    2. For each query, find top-k most similar non-positive documents.
    3. These become hard negatives for the next training stage.

    Returns triplets with updated 'negatives' lists.
    """
    print("Mining hard negatives...")

    # Collect all unique texts
    queries = [t["query"] for t in triplets]
    positives = [t["positive"] for t in triplets]
    all_docs = list(set(positives))  # deduplicate documents
    doc_to_idx = {doc: i for i, doc in enumerate(all_docs)}
    positive_indices = [doc_to_idx[p] for p in positives]

    # Embed queries
    print(f"  Embedding {len(queries)} queries...")
    query_embs = model.encode_sentences(queries, tokenizer, batch_size=batch_size, max_length=max_length)

    # Embed documents
    print(f"  Embedding {len(all_docs)} documents...")
    doc_embs = model.encode_sentences(all_docs, tokenizer, batch_size=batch_size, max_length=max_length)

    # For each query, find top-k most similar non-positive documents
    print(f"  Mining top-{top_k} hard negatives per query...")
    updated = []
    chunk_size = 1000  # process in chunks to manage memory

    for start in range(0, len(queries), chunk_size):
        end = min(start + chunk_size, len(queries))
        chunk_q = query_embs[start:end]  # (chunk, D)

        # Cosine similarity (embeddings are already normalized)
        sims = chunk_q @ doc_embs.T  # (chunk, num_docs)

        for i in range(end - start):
            global_i = start + i
            pos_idx = positive_indices[global_i]

            # Zero out the positive so it's not selected as negative
            row_sims = sims[i].copy()
            row_sims[pos_idx] = -1.0

            # Filter by minimum similarity
            mask = row_sims >= min_similarity
            valid_indices = np.where(mask)[0]

            if len(valid_indices) == 0:
                # No valid negatives above threshold, take top-k anyway
                top_indices = np.argsort(row_sims)[-top_k:][::-1]
            else:
                # Sort valid indices by similarity (descending)
                sorted_valid = valid_indices[np.argsort(row_sims[valid_indices])[::-1]]
                top_indices = sorted_valid[:top_k]

            hard_negs = [all_docs[idx] for idx in top_indices]
            updated.append({
                "query": triplets[global_i]["query"],
                "positive": triplets[global_i]["positive"],
                "negatives": hard_negs,
            })

    print(f"  Mined negatives for {len(updated)} queries")
    return updated
