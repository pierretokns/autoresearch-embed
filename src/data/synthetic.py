"""
LLM-labeled synthetic pair generation using the Anthropic API.
Generates query-positive pairs and verifies them (jxmo / pplx-embed insight).

This file is AGENT-MUTABLE.
"""

from __future__ import annotations

import os
import json
import anthropic


def get_client():
    """Get Anthropic API client."""
    return anthropic.Anthropic()


def generate_pairs(
    domain: str,
    num_pairs: int = 100,
    model: str = "claude-haiku-4-5-20251001",  # haiku for volume
) -> list[dict]:
    """
    Generate query-document pairs for a given domain.

    Args:
        domain: description of the target domain (e.g. "scientific papers about biology")
        num_pairs: number of pairs to generate per call
        model: Anthropic model to use
    Returns:
        list of {"query": str, "positive": str} dicts
    """
    client = get_client()
    pairs = []

    batch_size = min(num_pairs, 20)  # generate in batches of 20
    for batch_start in range(0, num_pairs, batch_size):
        n = min(batch_size, num_pairs - batch_start)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": f"""Generate exactly {n} query-document pairs for the domain: {domain}

Each pair should have:
- A natural search query (1-2 sentences)
- A relevant document passage (2-4 sentences) that genuinely answers/matches the query

Return as JSON array: [{{"query": "...", "positive": "..."}}]

Make queries diverse. Make documents specific and factual."""
            }],
        )
        try:
            text = response.content[0].text
            # Extract JSON from response
            start = text.index("[")
            end = text.rindex("]") + 1
            batch_pairs = json.loads(text[start:end])
            pairs.extend(batch_pairs)
        except (json.JSONDecodeError, ValueError):
            continue

    return pairs


def verify_pairs(
    pairs: list[dict],
    model: str = "claude-haiku-4-5-20251001",
) -> list[dict]:
    """
    Verify each query-document pair using an LLM.
    The pplx-embed insight: filter out false positives before training.

    Returns only pairs that pass verification.
    """
    client = get_client()
    verified = []

    # Batch verification
    batch_size = 10
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        pairs_text = "\n".join(
            f"{j+1}. Query: {p['query']}\n   Document: {p['positive']}"
            for j, p in enumerate(batch)
        )

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""For each query-document pair below, answer YES if the document genuinely answers/matches the query, NO if it doesn't.

{pairs_text}

Return one line per pair: just the number and YES or NO. Example:
1. YES
2. NO"""
            }],
        )

        text = response.content[0].text
        for j, pair in enumerate(batch):
            # Check if this pair was verified
            marker = f"{j+1}."
            if marker in text:
                line = text[text.index(marker):]
                if "YES" in line.split("\n")[0].upper():
                    verified.append(pair)

    return verified


def generate_verified_pairs(
    domain: str,
    target_count: int = 100,
    model: str = "claude-haiku-4-5-20251001",
) -> list[dict]:
    """Generate and verify pairs, returning only verified ones."""
    # Generate more than needed since some will fail verification
    raw = generate_pairs(domain, num_pairs=int(target_count * 1.5), model=model)
    verified = verify_pairs(raw, model=model)
    return verified[:target_count]
