"""Validation utilities for generation parameters."""

from typing import Tuple


def validate_prompt(prompt: str) -> Tuple[bool, str]:
    """Validate prompt text.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not prompt or not prompt.strip():
        return False, "Prompt cannot be empty"
    if len(prompt) > 5000:
        return False, "Prompt is too long (max 5000 characters)"
    return True, ""


def validate_generation_params(steps: int, guidance_scale: float, seed: int) -> Tuple[bool, str]:
    """Validate generation parameters.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not (1 <= steps <= 100):
        return False, "Steps must be between 1 and 100"
    if not (1.0 <= guidance_scale <= 10.0):
        return False, "Guidance scale must be between 1.0 and 10.0"
    if seed < -1:
        return False, "Seed must be -1 (random) or a positive integer"
    return True, ""
