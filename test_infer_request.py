"""Unit tests for InferRequest IP-Adapter fields."""
import sys
from unittest.mock import MagicMock
sys.modules['torch'] = MagicMock()
sys.modules['diffusers'] = MagicMock()
sys.modules['diffusers.FluxPipeline'] = MagicMock()
sys.modules['uvicorn'] = MagicMock()
sys.modules['fastapi'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
# pydantic must be real for BaseModel/Field to work
sys.path.insert(0, "inference")
import pytest
from server import InferRequest


def test_infer_request_defaults():
    req = InferRequest(prompt="test")
    assert req.ip_adapter_images == []
    assert req.adapter_strength == 0.8


def test_infer_request_ip_adapter_fields():
    req = InferRequest(
        prompt="test",
        ip_adapter_images=["base64data1", "base64data2"],
        adapter_strength=1.2,
    )
    assert req.ip_adapter_images == ["base64data1", "base64data2"]
    assert req.adapter_strength == 1.2


def test_infer_request_adapter_strength_too_high():
    with pytest.raises(Exception):
        InferRequest(prompt="test", adapter_strength=2.5)


def test_infer_request_adapter_strength_negative():
    with pytest.raises(Exception):
        InferRequest(prompt="test", adapter_strength=-0.1)
