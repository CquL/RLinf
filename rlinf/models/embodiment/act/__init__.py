"""ACT policy adapter for embodied RLinf runs."""

from .act_rl_policy import ACTRLPolicy


def get_model(cfg, torch_dtype):
    """Build an ACT policy from an RLinf model config."""
    return ACTRLPolicy(cfg, torch_dtype=torch_dtype)


__all__ = ["ACTRLPolicy", "get_model"]
