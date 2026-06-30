"""FLUX.1-dev Studio -- FastAPI Application."""

import asyncio
import base64
import json
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core import PromptExtendConfig, FluxConfig, FluxEngine
from utils.config import get_asset_path, load_environment
from utils.gpu_monitor import get_gpu_info
from utils.prompt_extender import PromptExtender, PromptExtendError


class AppState:
    """Server-side state shared across requests."""

    engine: FluxEngine
    prompt_extender: PromptExtender | None
    generation_history: deque
    _gpu_cache: list | None
    _gpu_cache_time: float

    def __init__(self) -> None:
        load_environment()
        config = FluxConfig()
        self.engine = FluxEngine(config)

        extend_config = PromptExtendConfig()
        if extend_config.is_configured:
            self.prompt_extender = PromptExtender(extend_config.prompt_extend_model)
        else:
            self.prompt_extender = None

        self.generation_history: deque = deque(maxlen=50)
        self._gpu_cache = None
        self._gpu_cache_time = 0.0

        self._load_gallery_from_disk()

    def _load_gallery_from_disk(self) -> None:
        """Load saved generations from JSON sidecar files on disk."""
        gallery_path = get_asset_path("generated")
        json_files = sorted(gallery_path.glob("flux_*.json"), reverse=True)
        for jf in json_files[:50]:
            try:
                meta = json.loads(jf.read_text())
                png_path = jf.with_suffix(".png")
                if not png_path.exists():
                    continue
                image_b64 = base64.b64encode(png_path.read_bytes()).decode("utf-8")
                meta["image_b64"] = image_b64
                self.generation_history.append(meta)
            except Exception:
                continue

    def get_gpu_info(self) -> list:
        """Return GPU info, cached for 10 seconds."""
        now = time.time()
        if self._gpu_cache is None or now - self._gpu_cache_time > 10:
            self._gpu_cache = get_gpu_info()
            self._gpu_cache_time = now
        return self._gpu_cache


app_state: AppState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_state
    app_state = AppState()
    yield
    app_state = None


app = FastAPI(title="FLUX.1-dev Studio", lifespan=lifespan)

BASE_DIR = Path(__file__).parent
app.mount("/flux/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


class ExtendRequest(BaseModel):
    prompt: str


class GpuConfigRequest(BaseModel):
    devices: list[int]


@app.get("/flux", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse(request, "index.html", {"hide_purge": True})


@app.get("/flux/api/health")
async def health():
    """Proxy health check to inference service."""
    try:
        result = app_state.engine.health_check()
        return result
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


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
    ip_adapter_images: list[UploadFile] = File(default=[]),
    adapter_strength: float = Form(0.8),
):
    """Generate image with optional control image, returns SSE stream."""
    # Read and encode control image if provided
    control_image_b64 = ""
    if control_image is not None and control_image.filename:
        control_bytes = await control_image.read()
        if control_bytes:
            control_image_b64 = base64.b64encode(control_bytes).decode("utf-8")

    ip_adapter_images_b64: list[str] = []
    if ip_adapter_images:
        for img_file in ip_adapter_images:
            if img_file is not None and img_file.filename:
                img_bytes = await img_file.read()
                if img_bytes:
                    ip_adapter_images_b64.append(
                        base64.b64encode(img_bytes).decode("utf-8")
                    )

    async def event_stream():
        import json as json_module
        yield f"data: {json_module.dumps({'type': 'started'})}\n\n"

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
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
            )

            buf = BytesIO()
            result.image.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            result_dict: dict[str, Any] = {
                "prompt": result.prompt,
                "negative_prompt": result.negative_prompt,
                "seed_used": result.seed_used,
                "generation_time": result.generation_time,
                "metadata": result.metadata,
                "timestamp": result.timestamp.isoformat(),
            }
            if result.control_mode:
                result_dict["control_mode"] = result.control_mode
                result_dict["controlnet_conditioning_scale"] = result.controlnet_conditioning_scale
            if result.ip_adapter_used:
                result_dict["ip_adapter_used"] = result.ip_adapter_used
                result_dict["adapter_strength"] = result.adapter_strength

            try:
                gallery_path = get_asset_path("generated")
                ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
                result.save(gallery_path / f"flux_{ts_slug}.png")
                meta_for_disk = {
                    k: v for k, v in result_dict.items()
                }
                (gallery_path / f"flux_{ts_slug}.json").write_text(
                    json_module.dumps(meta_for_disk)
                )
            except Exception:
                pass

            app_state.generation_history.appendleft({**result_dict, "image_b64": image_b64})

            complete_data = json_module.dumps({
                'type': 'complete',
                'result': {
                    **result_dict,
                    'image_b64': image_b64
                }
            }, ensure_ascii=False)
            yield f"data: {complete_data}\n\n"

        except Exception as exc:
            yield f"data: {json_module.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/flux/api/gpu")
async def gpu_info():
    """GPU telemetry."""
    gpus = app_state.get_gpu_info()
    return {
        "gpus": [
            {
                "index": g.index,
                "name": g.name,
                "utilization": g.utilization,
                "memory_used": g.memory_used,
                "memory_total": g.memory_total,
                "memory_percent": g.memory_percent,
                "temperature": g.temperature,
            }
            for g in gpus
        ]
    }


@app.get("/flux/api/gpu/config")
async def gpu_config_get():
    """Read current GPU device selection from .env."""
    env_path = BASE_DIR / ".env"
    devices: list[int] = []
    if env_path.exists():
        from dotenv import dotenv_values

        vals = dotenv_values(env_path)
        raw = vals.get("NVIDIA_GPU_DEVICE", "")
        if raw:
            devices = [int(d.strip()) for d in raw.split(",") if d.strip().isdigit()]
    return {"devices": devices}


@app.post("/flux/api/gpu/config")
async def gpu_config_set(req: GpuConfigRequest):
    """Write GPU device selection to .env."""
    if not req.devices:
        raise HTTPException(status_code=400, detail="At least one GPU must be selected")

    env_path = BASE_DIR / ".env"
    new_value = ",".join(str(d) for d in sorted(req.devices))

    lines: list[str] = []
    found = False
    if env_path.exists():
        lines = env_path.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.startswith("NVIDIA_GPU_DEVICE="):
                lines[i] = f"NVIDIA_GPU_DEVICE={new_value}"
                found = True
                break
    if not found:
        lines.append(f"NVIDIA_GPU_DEVICE={new_value}")

    env_path.write_text("\n".join(lines) + "\n")

    reload_status = "saved"
    try:
        import requests as http_requests

        resp = http_requests.post(
            f"{app_state.engine.config.base_url}/v1/gpu/config",
            json={"devices": sorted(req.devices)},
            timeout=10,
        )
        data = resp.json()
        reload_status = data.get("status", "unknown")
    except Exception:
        reload_status = "reload_failed"

    return {
        "devices": sorted(req.devices),
        "reload_status": reload_status,
    }


@app.post("/flux/api/extend-prompt")
async def extend_prompt(req: ExtendRequest):
    """LLM prompt extension."""
    if app_state.prompt_extender is None:
        raise HTTPException(
            status_code=503,
            detail="Prompt extension service is not configured. Set PROMPT_EXTEND_MODEL in .env.",
        )

    try:
        loop = asyncio.get_event_loop()
        extended = await loop.run_in_executor(
            None, app_state.prompt_extender.extend, req.prompt
        )
        return {"extended_prompt": extended}
    except PromptExtendError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/flux/api/history")
async def history():
    """Generation history."""
    return {"history": list(app_state.generation_history)}


@app.delete("/flux/api/history")
async def history_purge():
    """Purge all generation history."""
    app_state.generation_history.clear()
    gallery_path = get_asset_path("generated")
    for f in gallery_path.glob("flux_*"):
        try:
            f.unlink(missing_ok=True)
        except Exception:
            continue
    return {"purged": True}


@app.delete("/flux/api/history/{timestamp}")
async def history_delete(timestamp: str):
    """Delete a single generation by its ISO timestamp."""
    before = len(app_state.generation_history)
    app_state.generation_history = deque(
        (r for r in app_state.generation_history if r.get("timestamp") != timestamp),
        maxlen=50,
    )
    if len(app_state.generation_history) == before:
        raise HTTPException(status_code=404, detail="Image not found")

    gallery_path = get_asset_path("generated")
    for jf in gallery_path.glob("flux_*.json"):
        try:
            meta = json.loads(jf.read_text())
            if meta.get("timestamp") == timestamp:
                jf.unlink(missing_ok=True)
                jf.with_suffix(".png").unlink(missing_ok=True)
                break
        except Exception:
            continue
    return {"deleted": timestamp}
