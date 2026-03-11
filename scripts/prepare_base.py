"""
One-time setup: download base encoder model and tokenizer.
Converts to format usable by the training loop.

Usage: uv run scripts/prepare_base.py [--model answerdotai/ModernBERT-base]
"""

import argparse
from pathlib import Path

from transformers import AutoModel, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Download and prepare base encoder model")
    parser.add_argument("--model", default="answerdotai/ModernBERT-base", help="HuggingFace model ID")
    parser.add_argument("--output", default="checkpoints/base", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)

    print(f"Saving to: {output_dir}")
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    model.save_pretrained(str(output_dir / "model"))

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}")
    print(f"Parameters: {num_params / 1e6:.1f}M")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Num layers: {model.config.num_hidden_layers}")
    print(f"Saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
