"""Utility functions for FLUX.1-dev application."""

import os
from pathlib import Path

from dotenv import load_dotenv


def load_environment() -> None:
    """Load environment variables from .env file."""
    load_dotenv()


def get_asset_path(subdir: str) -> Path:
    """Get path to asset subdirectory, creating if needed."""
    from pathlib import Path

    base = Path(__file__).parent.parent / "assets" / subdir
    base.mkdir(parents=True, exist_ok=True)
    return base
