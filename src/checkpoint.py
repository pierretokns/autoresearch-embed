"""
Checkpoint save/load for PyTorch embedding models.

Currently checkpoints are saved inline in src/train.py using torch.save().
The agent may refactor checkpoint logic into here for cleaner separation.

This file is AGENT-MUTABLE.
"""
