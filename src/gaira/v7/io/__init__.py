"""GAIRA V7 — input/output resolution shared by every phase."""
from .outputs import PhaseOutputs, frozen_root, output_root, repo_root  # noqa: F401

__all__ = ["PhaseOutputs", "output_root", "frozen_root", "repo_root"]
