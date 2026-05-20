"""Unit tests for GenerationResult and FluxEngine IP-Adapter support."""
import sys
sys.path.insert(0, "app")
import base64
import io
import pytest
from unittest.mock import MagicMock
from PIL import Image as PILImage
from core import GenerationResult, FluxEngine, FluxConfig, ValidationError


def _make_png_b64() -> str:
    img = PILImage.new("RGB", (64, 64), "blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_result(**kwargs):
    defaults = dict(
        image=PILImage.new("RGB", (64, 64)),
        prompt="test",
        negative_prompt="",
        metadata={},
        seed_used=42,
        generation_time=1.0,
    )
    defaults.update(kwargs)
    return GenerationResult(**defaults)


def test_generation_result_ip_adapter_defaults():
    result = make_result()
    assert result.ip_adapter_used == False
    assert result.adapter_strength == 0.0


def test_generation_result_ip_adapter_fields():
    result = make_result(ip_adapter_used=True, adapter_strength=0.8)
    assert result.ip_adapter_used == True
    assert result.adapter_strength == 0.8


def test_to_dict_includes_ip_adapter_when_used():
    result = make_result(ip_adapter_used=True, adapter_strength=0.8)
    d = result.to_dict()
    assert d["ip_adapter_used"] == True
    assert d["adapter_strength"] == 0.8


def test_to_dict_excludes_ip_adapter_when_unused():
    result = make_result()
    d = result.to_dict()
    assert "ip_adapter_used" not in d
    assert "adapter_strength" not in d


def test_generate_image_rejects_invalid_adapter_strength():
    engine = FluxEngine(FluxConfig())
    with pytest.raises(ValidationError, match="adapter_strength"):
        engine.generate_image(
            prompt="test",
            ip_adapter_images_b64=["somebase64"],
            adapter_strength=3.0,
        )


def test_generate_image_payload_includes_ip_adapter():
    engine = FluxEngine(FluxConfig())
    engine.session = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "artifacts": [
            {
                "base64": _make_png_b64(),
                "finishReason": "SUCCESS",
                "seed": 42,
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()
    engine.session.post.return_value = mock_response

    result = engine.generate_image(
        prompt="HPE rack",
        ip_adapter_images_b64=["somebase64"],
        adapter_strength=0.8,
    )

    assert result.ip_adapter_used == True
    assert result.adapter_strength == 0.8
    payload = engine.session.post.call_args[1]["json"]
    assert payload["ip_adapter_images"] == ["somebase64"]
    assert payload["adapter_strength"] == 0.8


def test_generate_image_no_ip_adapter_in_payload_when_empty():
    engine = FluxEngine(FluxConfig())
    engine.session = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "artifacts": [
            {
                "base64": _make_png_b64(),
                "finishReason": "SUCCESS",
                "seed": 42,
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()
    engine.session.post.return_value = mock_response

    result = engine.generate_image(prompt="HPE rack")

    assert result.ip_adapter_used == False
    payload = engine.session.post.call_args[1]["json"]
    assert "ip_adapter_images" not in payload
