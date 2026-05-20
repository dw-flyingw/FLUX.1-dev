"""Core backend engine for FLUX.1-dev API integration."""

import base64
import io
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FluxConfig(BaseSettings):
    """Configuration for FLUX.1-dev inference service."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    nim_host: str = Field(default="localhost", alias="NIM_HOST")
    flux_inference_port: int = Field(default=8630, alias="FLUX_INFERENCE_PORT")
    verify_ssl: bool = Field(default=True)

    @property
    def base_url(self) -> str:
        """Get base URL for inference service."""
        return f"http://{self.nim_host}:{self.flux_inference_port}"


class PromptExtendConfig(BaseSettings):
    """Configuration for LLM-based prompt extension."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    prompt_extend_model: str = Field(default="", alias="PROMPT_EXTEND_MODEL")

    @property
    def is_configured(self) -> bool:
        """Return True if a prompt extend model URL is set."""
        return bool(self.prompt_extend_model.strip())


CONTROL_MODES = ["canny", "tile", "depth", "blur", "pose", "gray", "low_quality"]


@dataclass
class GenerationResult:
    """Result from image generation."""

    image: Image.Image
    prompt: str
    negative_prompt: str
    metadata: dict[str, Any]
    seed_used: int
    generation_time: float
    control_mode: str = ""
    controlnet_conditioning_scale: float = 0.0
    ip_adapter_used: bool = False
    adapter_strength: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def save(self, path: Path) -> None:
        """Save image with metadata to path."""
        from utils.image_utils import save_with_metadata

        save_metadata = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": str(self.seed_used),
            "generation_time": f"{self.generation_time:.2f}s",
            "timestamp": self.timestamp.isoformat(),
            **self.metadata,
        }
        if self.control_mode:
            save_metadata["control_mode"] = self.control_mode
            save_metadata["controlnet_conditioning_scale"] = str(self.controlnet_conditioning_scale)
        if self.ip_adapter_used:
            save_metadata["ip_adapter_used"] = "true"
            save_metadata["adapter_strength"] = str(self.adapter_strength)

        save_with_metadata(self.image, path, save_metadata)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        d = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed_used": self.seed_used,
            "generation_time": self.generation_time,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
        if self.control_mode:
            d["control_mode"] = self.control_mode
            d["controlnet_conditioning_scale"] = self.controlnet_conditioning_scale
        if self.ip_adapter_used:
            d["ip_adapter_used"] = self.ip_adapter_used
            d["adapter_strength"] = self.adapter_strength
        return d


class APIConnectionError(Exception):
    """Error connecting to NIM API."""

    pass


class GenerationError(Exception):
    """Error during image generation."""

    pass


class ValidationError(Exception):
    """Parameter validation error."""

    pass


class FluxEngine:
    """Engine for interacting with FLUX.1-dev inference service."""

    ASPECT_RATIOS = {
        "1:1": "1024x1024",
        "16:9": "1344x768",
        "9:16": "768x1344",
        "4:5": "832x1024",
        "5:4": "1024x832",
        "3:2": "1280x853",
        "2:3": "853x1280",
    }

    SAMPLERS = ["euler"]

    def __init__(self, config: FluxConfig):
        """Initialize engine with configuration.

        Args:
            config: FluxConfig instance
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        self.session.verify = config.verify_ssl

    def health_check(self) -> dict[str, Any]:
        """Check if inference service is ready.

        Returns:
            Health status dictionary

        Raises:
            APIConnectionError: If service is not reachable
        """
        try:
            response = self.session.get(
                f"{self.config.base_url}/v1/health/ready", timeout=10
            )
            response.raise_for_status()
            return {"status": "ready", "message": "Inference service is ready"}
        except requests.exceptions.ConnectionError as e:
            raise APIConnectionError(
                f"Cannot connect to inference service at {self.config.base_url}. Ensure the service is running."
            ) from e
        except requests.exceptions.Timeout as e:
            raise APIConnectionError(
                f"Timeout connecting to inference service at {self.config.base_url}"
            ) from e
        except requests.exceptions.HTTPError as e:
            raise APIConnectionError(f"HTTP error: {e}") from e

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 28,
        guidance_scale: float = 3.5,
        sampler: str = "euler",
        aspect_ratio: str = "1:1",
        seed: int = -1,
        control_image_b64: str = "",
        control_mode: str = "canny",
        controlnet_conditioning_scale: float = 0.5,
        ip_adapter_images_b64: list[str] | None = None,
        adapter_strength: float = 0.8,
    ) -> GenerationResult:
        """Generate image from prompt, optionally with a control image.

        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid in generation
            steps: Number of sampling steps (10-50)
            guidance_scale: Guidance scale value (1.0-10.0)
            sampler: Sampling method
            aspect_ratio: Image aspect ratio
            seed: Random seed (-1 for random)
            control_image_b64: Base64-encoded control image (empty for text-only)
            control_mode: ControlNet mode (canny, tile, depth, blur, pose, gray, low_quality)
            controlnet_conditioning_scale: How strongly the control image influences output (0.0-2.0)
            ip_adapter_images_b64: List of base64-encoded reference images for IP-Adapter (None to disable)
            adapter_strength: IP-Adapter influence strength (0.0-2.0, default 0.8)

        Returns:
            GenerationResult with generated image

        Raises:
            ValidationError: If parameters are invalid
            GenerationError: If generation fails
            APIConnectionError: If API request fails
        """
        from utils.validators import validate_generation_params, validate_prompt

        is_valid, error_msg = validate_prompt(prompt)
        if not is_valid:
            raise ValidationError(error_msg)

        is_valid, error_msg = validate_generation_params(steps, guidance_scale, seed)
        if not is_valid:
            raise ValidationError(error_msg)

        if sampler not in self.SAMPLERS:
            raise ValidationError(f"Invalid sampler. Must be one of: {self.SAMPLERS}")

        if aspect_ratio not in self.ASPECT_RATIOS:
            raise ValidationError(
                f"Invalid aspect ratio. Must be one of: {list(self.ASPECT_RATIOS.keys())}"
            )

        if control_image_b64 and control_mode not in CONTROL_MODES:
            raise ValidationError(
                f"Invalid control mode. Must be one of: {CONTROL_MODES}"
            )

        if ip_adapter_images_b64:
            if not (0.0 <= adapter_strength <= 2.0):
                raise ValidationError("adapter_strength must be between 0.0 and 2.0")

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        size = self.ASPECT_RATIOS[aspect_ratio]
        width, height = map(int, size.split("x"))

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "sampler": sampler,
            "seed": seed,
        }

        if control_image_b64:
            payload["control_image"] = control_image_b64
            payload["control_mode"] = control_mode
            payload["controlnet_conditioning_scale"] = controlnet_conditioning_scale

        if ip_adapter_images_b64:
            payload["ip_adapter_images"] = ip_adapter_images_b64
            payload["adapter_strength"] = adapter_strength

        start_time = time.time()
        try:
            response = self.session.post(
                f"{self.config.base_url}/v1/infer", json=payload, timeout=300
            )
            response.raise_for_status()
            generation_time = time.time() - start_time

        except requests.exceptions.Timeout as e:
            raise GenerationError(
                "Generation timed out after 300 seconds. Try reducing steps or image size."
            ) from e
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise APIConnectionError(
                    "Rate limit exceeded. Please wait before retrying."
                ) from e
            else:
                raise GenerationError(
                    f"HTTP error {e.response.status_code}: {e}"
                ) from e
        except requests.exceptions.ConnectionError as e:
            raise APIConnectionError(
                "Lost connection to inference service during generation."
            ) from e

        try:
            response_data = response.json()

            if "artifacts" in response_data and response_data["artifacts"]:
                artifact = response_data["artifacts"][0]
                finish_reason = artifact.get("finishReason", "")
                error_reason = artifact.get("errorReason", "")
                if finish_reason == "CONTENT_FILTERED":
                    raise GenerationError(
                        "Image blocked by safety filter. "
                        "Try rephrasing your prompt to avoid potentially sensitive content."
                    )
                if finish_reason == "MODEL_NOT_READY":
                    detail = error_reason or "Model is not loaded"
                    raise APIConnectionError(f"{detail}. Please wait and try again.")
                if finish_reason == "ERROR":
                    detail = (
                        error_reason or "Try different parameters or a simpler prompt."
                    )
                    raise GenerationError(f"Generation failed: {detail}")
                b64_data = artifact["base64"]
                if "seed" in artifact:
                    seed = artifact["seed"]
            else:
                raise GenerationError(
                    f"No image data in response. Response keys: {list(response_data.keys())}"
                )

            image_bytes = base64.b64decode(b64_data)
            image = Image.open(io.BytesIO(image_bytes))

        except GenerationError:
            raise
        except (KeyError, IndexError) as e:
            raise GenerationError(
                f"Unexpected response format: {e}. "
                f"Response keys: {list(response_data.keys()) if 'response_data' in locals() else 'unknown'}"
            ) from e
        except Exception as e:
            raise GenerationError(f"Failed to decode image: {e}") from e

        metadata = {
            "steps": steps,
            "guidance_scale": guidance_scale,
            "sampler": sampler,
            "aspect_ratio": aspect_ratio,
            "size": size,
        }

        return GenerationResult(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            metadata=metadata,
            seed_used=seed,
            generation_time=generation_time,
            control_mode=control_mode if control_image_b64 else "",
            controlnet_conditioning_scale=controlnet_conditioning_scale if control_image_b64 else 0.0,
            ip_adapter_used=bool(ip_adapter_images_b64),
            adapter_strength=adapter_strength if ip_adapter_images_b64 else 0.0,
        )
