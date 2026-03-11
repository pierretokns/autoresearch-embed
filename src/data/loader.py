"""
Contrastive dataloader for PyTorch embedding training.

Currently data loading is inline in src/train.py.
The agent may refactor the dataloader into here for cleaner separation
(e.g., pre-tokenization, DataLoader with workers, source stratification).

This file is AGENT-MUTABLE.
"""
