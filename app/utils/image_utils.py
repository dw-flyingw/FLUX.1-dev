"""Image utilities for saving with metadata."""

from pathlib import Path

from PIL import Image


def save_with_metadata(image: Image.Image, path: Path, metadata: dict) -> None:
    """Save image with metadata in PNG text chunk."""
    png_info = PngInfo()
    for key, value in metadata.items():
        png_info.add_text(key, str(value))
    image.save(path, pnginfo=png_info)


from PIL import PngImagePlugin

PngInfo = PngImagePlugin.PngInfo
