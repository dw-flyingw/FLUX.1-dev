# IP-Adapter Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add IP-Adapter support to the FLUX.1-dev inference service so reference images can guide generation (style/content) alongside existing ControlNet structural conditioning.

**Architecture:** The NIM backend (`inference/server.py`) loads IP-Adapter weights at startup into both pipelines via `load_ip_adapter()`. The `/v1/infer` endpoint accepts `ip_adapter_images` (list of base64 strings) and `adapter_strength` (float 0-2). The FastAPI layer (`app/app.py`) accepts these as file uploads, encodes to base64, and forwards through `FluxEngine.generate_image()` in `app/core.py`.

**Tech Stack:** Python 3.11, FastAPI, diffusers (`FluxPipeline`, `FluxControlNetPipeline` with `IPAdapterMixin`), PyTorch, Pydantic v2, HuggingFace Hub

---

### Task 1: Download IP-Adapter weights

**Files:**
- No code changes — environment setup only

- [ ] **Step 1: Verify diffusers supports FLUX IP-Adapter**

Run:
```bash
cd /home/users/wrightda/src/FLUX.1-dev
.venv/bin/python -c "from diffusers import FluxPipeline; print(hasattr(FluxPipeline, 'load_ip_adapter'))"
```
Expected: `True`

If `False`, upgrade diffusers:
```bash
.venv/bin/pip install -U diffusers
```

- [ ] **Step 2: Download IP-Adapter weights from HuggingFace**

Run:
```bash
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='InstantX/FLUX.1-dev-IP-Adapter',
    local_dir='models/ip-adapter',
    ignore_patterns=['*.msgpack', '*.h5', '*.safetensors.index.json'],
)
print('Done')
"
```
Expected: prints `Done` and creates `models/ip-adapter/ip-adapter.bin`.

- [ ] **Step 3: Verify the weight file exists**

Run:
```bash
ls -lh models/ip-adapter/ip-adapter.bin
```
Expected: file exists, size ~3–10 GB.

- [ ] **Step 4: Add model directory to .gitignore and commit**

Run:
```bash
echo "models/" >> .gitignore
git add .gitignore
git commit -m "chore: ignore downloaded model weights directory"
```

---

### Task 2: Update NIM backend InferRequest model

**Files:**
- Modify: `inference/server.py` — `InferRequest` class (lines 208–219)

- [ ] **Step 1: Write the failing test**

Create `test_infer_request.py` at the project root:
```python
"""Unit tests for InferRequest IP-Adapter fields."""
import sys
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
.venv/bin/pytest test_infer_request.py -v
```
Expected: FAIL — `InferRequest` has no `ip_adapter_images` attribute.

- [ ] **Step 3: Replace InferRequest in inference/server.py**

Find the `InferRequest` class (lines 208–219) and replace the entire class body:
```python
class InferRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    control_image: str = ""
    control_mode: str = "canny"
    controlnet_conditioning_scale: float = Field(default=0.5, ge=0.0, le=2.0)
    ip_adapter_images: list[str] = []
    adapter_strength: float = Field(default=0.8, ge=0.0, le=2.0)
    width: int = 1024
    height: int = 1024
    steps: int = Field(default=30, alias="steps")
    guidance_scale: float = 3.5
    sampler: str = "euler"
    seed: int = 0
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
.venv/bin/pytest test_infer_request.py -v
```
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add inference/server.py test_infer_request.py
git commit -m "feat: add ip_adapter_images and adapter_strength to NIM InferRequest"
```

---

### Task 3: Update NIM backend model loading

**Files:**
- Modify: `inference/server.py` — constants block (after line 29) and `_load_model()` (lines 93–168)

- [ ] **Step 1: Add IP-Adapter constants after existing constants (line 29)**

After the `CONTROLNET_MODEL_ID` line, add:
```python
IP_ADAPTER_MODEL_ID = "InstantX/FLUX.1-dev-IP-Adapter"
IP_ADAPTER_WEIGHT_NAME = "ip-adapter.bin"
IP_ADAPTER_LOCAL_DIR = "models/ip-adapter"
```

- [ ] **Step 2: Load IP-Adapter into both pipelines in _load_model()**

In `_load_model()`, find the multi-GPU branch. After the line:
```python
        logger.info("Models loaded with device_map='balanced' across GPUs %s", devices)
```
Add:
```python
        new_pipe_base.load_ip_adapter(
            IP_ADAPTER_LOCAL_DIR,
            weight_name=IP_ADAPTER_WEIGHT_NAME,
            local_files_only=True,
        )
        new_pipe_cn.load_ip_adapter(
            IP_ADAPTER_LOCAL_DIR,
            weight_name=IP_ADAPTER_WEIGHT_NAME,
            local_files_only=True,
        )
        logger.info("IP-Adapter loaded from %s", IP_ADAPTER_LOCAL_DIR)
```

Find the single-GPU branch. After the line:
```python
        logger.info("Models loaded on cuda:%d", devices[0])
```
Add the same block:
```python
        new_pipe_base.load_ip_adapter(
            IP_ADAPTER_LOCAL_DIR,
            weight_name=IP_ADAPTER_WEIGHT_NAME,
            local_files_only=True,
        )
        new_pipe_cn.load_ip_adapter(
            IP_ADAPTER_LOCAL_DIR,
            weight_name=IP_ADAPTER_WEIGHT_NAME,
            local_files_only=True,
        )
        logger.info("IP-Adapter loaded from %s", IP_ADAPTER_LOCAL_DIR)
```

- [ ] **Step 3: Update root() to report IP-Adapter**

Replace the entire `root()` function:
```python
@app.get("/")
async def root():
    return {
        "model": BASE_MODEL_ID,
        "controlnet": CONTROLNET_MODEL_ID,
        "ip_adapter": IP_ADAPTER_MODEL_ID,
        "status": "ready" if pipe_base is not None else "loading",
        "control_modes": list(CONTROL_MODES.keys()),
        "endpoints": {
            "health": "/v1/health/ready",
            "infer": "/v1/infer",
            "gpu_config": "/v1/gpu/config",
        },
    }
```

- [ ] **Step 4: Verify the server starts and logs IP-Adapter loading**

Run (requires GPU):
```bash
cd inference && ../.venv/bin/python server.py 2>&1 | head -30
```
Expected log line: `IP-Adapter loaded from models/ip-adapter`

- [ ] **Step 5: Commit**

```bash
git add inference/server.py
git commit -m "feat: load IP-Adapter weights into FLUX pipelines at startup"
```

---

### Task 4: Update NIM backend inference handler

**Files:**
- Modify: `inference/server.py` — `infer()` function (lines 323–432)

- [ ] **Step 1: Replace the entire infer() function**

Replace the `@app.post("/v1/infer", ...)` function with:
```python
@app.post("/v1/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    use_controlnet = bool(req.control_image)
    use_ip_adapter = bool(req.ip_adapter_images)
    current_pipe = pipe_controlnet if use_controlnet else pipe_base

    if current_pipe is None:
        reason = (
            "Model is reloading onto new GPU(s)"
            if reload_lock.locked()
            else "Model is still loading"
        )
        logger.warning("Inference rejected: %s", reason)
        return InferResponse(
            artifacts=[
                Artifact(
                    base64="",
                    seed=0,
                    finishReason="MODEL_NOT_READY",
                    errorReason=reason,
                )
            ]
        )

    # Decode control image if provided
    control_image = None
    if use_controlnet:
        if req.control_mode not in CONTROL_MODES:
            return InferResponse(
                artifacts=[
                    Artifact(
                        base64="",
                        seed=0,
                        finishReason="ERROR",
                        errorReason=f"Invalid control_mode '{req.control_mode}'. Must be one of: {list(CONTROL_MODES.keys())}",
                    )
                ]
            )
        try:
            control_bytes = base64.b64decode(req.control_image)
            control_image = Image.open(io.BytesIO(control_bytes)).convert("RGB")
            control_image = control_image.resize((req.width, req.height))
        except Exception as e:
            return InferResponse(
                artifacts=[
                    Artifact(
                        base64="",
                        seed=0,
                        finishReason="ERROR",
                        errorReason=f"Failed to decode control_image: {e}",
                    )
                ]
            )

    # Decode IP-Adapter images if provided
    ip_adapter_pil_images = []
    if use_ip_adapter:
        try:
            for img_b64 in req.ip_adapter_images:
                img_bytes = base64.b64decode(img_b64)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                ip_adapter_pil_images.append(pil_img)
        except Exception as e:
            return InferResponse(
                artifacts=[
                    Artifact(
                        base64="",
                        seed=0,
                        finishReason="ERROR",
                        errorReason=f"Failed to decode ip_adapter_image: {e}",
                    )
                ]
            )

    scheduler_cls = SCHEDULER_MAP.get(req.sampler)
    if scheduler_cls:
        current_pipe.scheduler = scheduler_cls.from_config(
            current_pipe.scheduler.config
        )

    # Set IP-Adapter scale; reset to 0 when not in use so prior state doesn't bleed
    current_pipe.set_ip_adapter_scale(req.adapter_strength if use_ip_adapter else 0.0)

    generator = torch.Generator(device=_get_generator_device()).manual_seed(req.seed)

    try:
        with inference_lock:
            if use_controlnet and use_ip_adapter:
                result = current_pipe(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt if req.negative_prompt else None,
                    control_image=control_image,
                    control_mode=CONTROL_MODES[req.control_mode],
                    controlnet_conditioning_scale=req.controlnet_conditioning_scale,
                    ip_adapter_image=ip_adapter_pil_images,
                    width=req.width,
                    height=req.height,
                    num_inference_steps=req.steps,
                    guidance_scale=req.guidance_scale,
                    generator=generator,
                )
            elif use_controlnet:
                result = current_pipe(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt if req.negative_prompt else None,
                    control_image=control_image,
                    control_mode=CONTROL_MODES[req.control_mode],
                    controlnet_conditioning_scale=req.controlnet_conditioning_scale,
                    width=req.width,
                    height=req.height,
                    num_inference_steps=req.steps,
                    guidance_scale=req.guidance_scale,
                    generator=generator,
                )
            elif use_ip_adapter:
                result = current_pipe(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt if req.negative_prompt else None,
                    ip_adapter_image=ip_adapter_pil_images,
                    width=req.width,
                    height=req.height,
                    num_inference_steps=req.steps,
                    guidance_scale=req.guidance_scale,
                    generator=generator,
                )
            else:
                result = current_pipe(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt if req.negative_prompt else None,
                    width=req.width,
                    height=req.height,
                    num_inference_steps=req.steps,
                    guidance_scale=req.guidance_scale,
                    generator=generator,
                )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        reason = f"GPU out of memory. Try reducing resolution ({req.width}x{req.height}) or steps ({req.steps})."
        logger.error("CUDA OOM during inference: %s", reason)
        return InferResponse(
            artifacts=[
                Artifact(base64="", seed=0, finishReason="ERROR", errorReason=reason)
            ]
        )
    except Exception as e:
        reason = f"Pipeline error: {type(e).__name__}: {e}"
        logger.exception("Unhandled error during inference")
        return InferResponse(
            artifacts=[
                Artifact(base64="", seed=0, finishReason="ERROR", errorReason=reason)
            ]
        )

    image = result.images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

    return InferResponse(artifacts=[Artifact(base64=b64_data, seed=req.seed)])
```

- [ ] **Step 2: Run unit tests to confirm nothing broke**

Run:
```bash
.venv/bin/pytest test_infer_request.py -v
```
Expected: All 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add inference/server.py
git commit -m "feat: add IP-Adapter inference path to NIM backend (4 scenarios)"
```

---

### Task 5: Update GenerationResult dataclass in core.py

**Files:**
- Modify: `app/core.py` — `GenerationResult` dataclass (lines 56–101)

- [ ] **Step 1: Write the failing tests**

Create `test_core_ip_adapter.py` at the project root:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
.venv/bin/pytest test_core_ip_adapter.py -v
```
Expected: FAIL — `GenerationResult` has no `ip_adapter_used` field.

- [ ] **Step 3: Update GenerationResult in app/core.py**

Replace the entire `GenerationResult` dataclass (lines 56–101) with:
```python
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
```

- [ ] **Step 4: Run tests to verify GenerationResult tests pass**

Run:
```bash
.venv/bin/pytest test_core_ip_adapter.py::test_generation_result_ip_adapter_defaults test_core_ip_adapter.py::test_generation_result_ip_adapter_fields test_core_ip_adapter.py::test_to_dict_includes_ip_adapter_when_used test_core_ip_adapter.py::test_to_dict_excludes_ip_adapter_when_unused -v
```
Expected: All 4 PASS (engine tests still fail — that's expected).

- [ ] **Step 5: Commit**

```bash
git add app/core.py test_core_ip_adapter.py
git commit -m "feat: add ip_adapter_used and adapter_strength to GenerationResult"
```

---

### Task 6: Update FluxEngine.generate_image() in core.py

**Files:**
- Modify: `app/core.py` — `generate_image()` method (lines 179–343)

- [ ] **Step 1: Update the generate_image() signature**

Replace the method signature (lines 179–191) with:
```python
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
```

- [ ] **Step 2: Add adapter_strength validation after the existing validation block**

In the body of `generate_image()`, after the existing `validate_generation_params` call and `if control_image_b64 and control_mode not in CONTROL_MODES` check, add:
```python
        if ip_adapter_images_b64:
            if not (0.0 <= adapter_strength <= 2.0):
                raise ValidationError("adapter_strength must be between 0.0 and 2.0")
```

- [ ] **Step 3: Add IP-Adapter params to the request payload**

In the payload construction section, after the existing ControlNet block:
```python
        if control_image_b64:
            payload["control_image"] = control_image_b64
            payload["control_mode"] = control_mode
            payload["controlnet_conditioning_scale"] = controlnet_conditioning_scale
```
Add:
```python
        if ip_adapter_images_b64:
            payload["ip_adapter_images"] = ip_adapter_images_b64
            payload["adapter_strength"] = adapter_strength
```

- [ ] **Step 4: Update the GenerationResult construction at the end of generate_image()**

Replace the final `return GenerationResult(...)` statement with:
```python
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
```

- [ ] **Step 5: Run all core tests to verify they pass**

Run:
```bash
.venv/bin/pytest test_core_ip_adapter.py -v
```
Expected: All 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/core.py
git commit -m "feat: add IP-Adapter params to FluxEngine.generate_image()"
```

---

### Task 7: Update FastAPI endpoint in app/app.py

**Files:**
- Modify: `app/app.py` — `generate()` function (lines 120–206)

- [ ] **Step 1: Update the generate() function signature**

Replace the `generate` function signature (lines 120–132) with:
```python
@app.post("/flux/api/generate")
async def generate(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    steps: int = Form(28),
    guidance_scale: float = Form(3.5),
    sampler: str = Form("euler"),
    aspect_ratio: str = Form("1:1"),
    seed: int = Form(-1),
    control_mode: str = Form("canny"),
    controlnet_conditioning_scale: float = Form(0.5),
    control_image: UploadFile | None = File(None),
    ip_adapter_images: list[UploadFile] | None = File(None),
    adapter_strength: float = Form(0.8),
):
```

- [ ] **Step 2: Add IP-Adapter image encoding after the existing control_image block**

In the `generate` function body, after the existing `control_image_b64` block (the one that reads and encodes `control_image`), add:
```python
    ip_adapter_images_b64: list[str] = []
    if ip_adapter_images:
        for img_file in ip_adapter_images:
            if img_file is not None and img_file.filename:
                img_bytes = await img_file.read()
                if img_bytes:
                    ip_adapter_images_b64.append(
                        base64.b64encode(img_bytes).decode("utf-8")
                    )
```

- [ ] **Step 3: Update the generate_image() call in event_stream**

Replace the `lambda:` call inside `run_in_executor` with:
```python
                lambda: app_state.engine.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    sampler=sampler,
                    aspect_ratio=aspect_ratio,
                    seed=seed,
                    control_image_b64=control_image_b64,
                    control_mode=control_mode,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    ip_adapter_images_b64=ip_adapter_images_b64,
                    adapter_strength=adapter_strength,
                ),
```

- [ ] **Step 4: Add IP-Adapter metadata to result_dict**

In `event_stream`, after the existing ControlNet metadata block:
```python
            if result.control_mode:
                result_dict["control_mode"] = result.control_mode
                result_dict["controlnet_conditioning_scale"] = result.controlnet_conditioning_scale
```
Add:
```python
            if result.ip_adapter_used:
                result_dict["ip_adapter_used"] = result.ip_adapter_used
                result_dict["adapter_strength"] = result.adapter_strength
```

- [ ] **Step 5: Run unit tests to confirm nothing broke**

Run:
```bash
.venv/bin/pytest test_core_ip_adapter.py test_infer_request.py -v
```
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/app.py
git commit -m "feat: add IP-Adapter file upload and adapter_strength to FastAPI endpoint"
```

---

### Task 8: Write and run integration tests

**Files:**
- Create: `test_ip_adapter.py`

- [ ] **Step 1: Create the integration test file**

Create `test_ip_adapter.py` at the project root:
```python
#!/usr/bin/env python3
"""Integration tests for FLUX.1-dev IP-Adapter support.

Requires both servers running:
  - NIM backend:  cd inference && python server.py   (port 8630)
  - FastAPI app:  cd app && python main.py           (port 8080)
"""
import base64
import io
import sys
import time

import requests
from PIL import Image, ImageDraw

BASE_URL = "http://localhost:8630"
TIMEOUT = 300


def make_reference_image(width: int = 512, height: int = 512, color: str = "blue") -> str:
    img = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, width - 50, height - 50], outline="white", width=5)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_control_image(width: int = 512, height: int = 512) -> str:
    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 400, 400], outline="white", width=3)
    draw.line([100, 100, 400, 400], fill="white", width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_health():
    print("1. Health check...", end=" ", flush=True)
    resp = requests.get(f"{BASE_URL}/v1/health/ready", timeout=10)
    data = resp.json()
    if data.get("status") == "ready":
        print("PASS")
        return True
    print(f"FAIL - {data}")
    return False


def test_ip_adapter_in_root():
    print("2. Root reports ip_adapter...", end=" ", flush=True)
    resp = requests.get(f"{BASE_URL}/", timeout=10)
    data = resp.json()
    if "ip_adapter" in data and data.get("status") == "ready":
        print(f"PASS - {data['ip_adapter']}")
        return True
    print(f"FAIL - {data}")
    return False


def test_ip_adapter_only_single_image():
    print("3. IP-Adapter only (single image, 4 steps)...", end=" ", flush=True)
    payload = {
        "prompt": "HPE DL380a server in an HPE rack, product photography",
        "ip_adapter_images": [make_reference_image(color="silver")],
        "adapter_strength": 0.8,
        "width": 512,
        "height": 512,
        "steps": 4,
        "guidance_scale": 3.5,
        "seed": 42,
    }
    start = time.time()
    resp = requests.post(f"{BASE_URL}/v1/infer", json=payload, timeout=TIMEOUT)
    elapsed = time.time() - start
    artifact = resp.json().get("artifacts", [{}])[0]
    if artifact.get("finishReason") == "SUCCESS" and artifact.get("base64"):
        img = Image.open(io.BytesIO(base64.b64decode(artifact["base64"])))
        img.save("test_output_ip_adapter_single.png")
        print(f"PASS - {img.size[0]}x{img.size[1]}, {elapsed:.1f}s")
        return True
    print(f"FAIL - {artifact.get('finishReason')}: {artifact.get('errorReason')}")
    return False


def test_ip_adapter_only_multiple_images():
    print("4. IP-Adapter only (multiple images, 4 steps)...", end=" ", flush=True)
    payload = {
        "prompt": "HPE rack with multiple servers, datacenter",
        "ip_adapter_images": [
            make_reference_image(color="silver"),
            make_reference_image(color="darkgray"),
        ],
        "adapter_strength": 0.7,
        "width": 512,
        "height": 512,
        "steps": 4,
        "guidance_scale": 3.5,
        "seed": 42,
    }
    start = time.time()
    resp = requests.post(f"{BASE_URL}/v1/infer", json=payload, timeout=TIMEOUT)
    elapsed = time.time() - start
    artifact = resp.json().get("artifacts", [{}])[0]
    if artifact.get("finishReason") == "SUCCESS" and artifact.get("base64"):
        img = Image.open(io.BytesIO(base64.b64decode(artifact["base64"])))
        img.save("test_output_ip_adapter_multi.png")
        print(f"PASS - {img.size[0]}x{img.size[1]}, {elapsed:.1f}s")
        return True
    print(f"FAIL - {artifact.get('finishReason')}: {artifact.get('errorReason')}")
    return False


def test_controlnet_only_regression():
    print("5. ControlNet only (regression)...", end=" ", flush=True)
    payload = {
        "prompt": "a server rack in a clean datacenter",
        "control_image": make_control_image(),
        "control_mode": "canny",
        "controlnet_conditioning_scale": 0.6,
        "width": 512,
        "height": 512,
        "steps": 4,
        "guidance_scale": 3.5,
        "seed": 42,
    }
    start = time.time()
    resp = requests.post(f"{BASE_URL}/v1/infer", json=payload, timeout=TIMEOUT)
    elapsed = time.time() - start
    artifact = resp.json().get("artifacts", [{}])[0]
    if artifact.get("finishReason") == "SUCCESS" and artifact.get("base64"):
        print(f"PASS - {elapsed:.1f}s")
        return True
    print(f"FAIL - {artifact.get('finishReason')}: {artifact.get('errorReason')}")
    return False


def test_controlnet_and_ip_adapter_combined():
    print("6. ControlNet + IP-Adapter combined...", end=" ", flush=True)
    payload = {
        "prompt": "HPE DL380a mounted in HPE rack, professional datacenter photography",
        "control_image": make_control_image(),
        "control_mode": "canny",
        "controlnet_conditioning_scale": 0.5,
        "ip_adapter_images": [make_reference_image(color="silver")],
        "adapter_strength": 0.8,
        "width": 512,
        "height": 512,
        "steps": 4,
        "guidance_scale": 3.5,
        "seed": 42,
    }
    start = time.time()
    resp = requests.post(f"{BASE_URL}/v1/infer", json=payload, timeout=TIMEOUT)
    elapsed = time.time() - start
    artifact = resp.json().get("artifacts", [{}])[0]
    if artifact.get("finishReason") == "SUCCESS" and artifact.get("base64"):
        img = Image.open(io.BytesIO(base64.b64decode(artifact["base64"])))
        img.save("test_output_combined.png")
        print(f"PASS - {img.size[0]}x{img.size[1]}, {elapsed:.1f}s")
        return True
    print(f"FAIL - {artifact.get('finishReason')}: {artifact.get('errorReason')}")
    return False


def test_invalid_adapter_strength_rejected():
    print("7. Invalid adapter_strength (5.0) rejected...", end=" ", flush=True)
    payload = {
        "prompt": "test",
        "ip_adapter_images": [make_reference_image(64, 64)],
        "adapter_strength": 5.0,
        "width": 512,
        "height": 512,
        "steps": 1,
        "seed": 1,
    }
    resp = requests.post(f"{BASE_URL}/v1/infer", json=payload, timeout=30)
    if resp.status_code == 422:
        print("PASS - rejected with HTTP 422")
        return True
    print(f"FAIL - expected 422, got {resp.status_code}")
    return False


def test_empty_ip_adapter_falls_back_to_text_only():
    print("8. Empty ip_adapter_images → text-only fallback...", end=" ", flush=True)
    payload = {
        "prompt": "a server rack, simple render",
        "ip_adapter_images": [],
        "adapter_strength": 0.8,
        "width": 512,
        "height": 512,
        "steps": 4,
        "seed": 42,
    }
    start = time.time()
    resp = requests.post(f"{BASE_URL}/v1/infer", json=payload, timeout=TIMEOUT)
    elapsed = time.time() - start
    artifact = resp.json().get("artifacts", [{}])[0]
    if artifact.get("finishReason") == "SUCCESS" and artifact.get("base64"):
        print(f"PASS - text-only, {elapsed:.1f}s")
        return True
    print(f"FAIL - {artifact.get('finishReason')}: {artifact.get('errorReason')}")
    return False


def main():
    print(f"Testing FLUX.1-dev IP-Adapter at {BASE_URL}\n")
    results = [
        ("Health", test_health()),
        ("IP-Adapter in root", test_ip_adapter_in_root()),
        ("IP-Adapter single image", test_ip_adapter_only_single_image()),
        ("IP-Adapter multiple images", test_ip_adapter_only_multiple_images()),
        ("ControlNet regression", test_controlnet_only_regression()),
        ("ControlNet + IP-Adapter", test_controlnet_and_ip_adapter_combined()),
        ("Invalid adapter_strength", test_invalid_adapter_strength_rejected()),
        ("Empty images fallback", test_empty_ip_adapter_falls_back_to_text_only()),
    ]
    print(f"\n{'='*40}")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    for name, result in results:
        print(f"  {'PASS' if result else 'FAIL'} - {name}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all unit tests to confirm clean baseline**

Run:
```bash
.venv/bin/pytest test_infer_request.py test_core_ip_adapter.py -v
```
Expected: All 11 tests PASS.

- [ ] **Step 3: Run integration tests against live servers**

Ensure both servers are running, then:
```bash
python test_ip_adapter.py
```
Expected: All 8 tests PASS. Output images saved: `test_output_ip_adapter_single.png`, `test_output_ip_adapter_multi.png`, `test_output_combined.png`.

- [ ] **Step 4: Commit**

```bash
git add test_ip_adapter.py
git commit -m "test: add IP-Adapter integration test suite (8 tests)"
```
