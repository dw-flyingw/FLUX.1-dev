"""FastAPI inference service for FLUX.1-dev with optional ControlNet."""

import base64
import io
import logging
import os
import subprocess
import threading
from contextlib import asynccontextmanager

import torch
import uvicorn
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxControlNetModel,
    FluxControlNetPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
)
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL_ID = "black-forest-labs/FLUX.1-dev"
CONTROLNET_MODEL_ID = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"

CONTROL_MODES = {
    "canny": 0,
    "tile": 1,
    "depth": 2,
    "blur": 3,
    "pose": 4,
    "gray": 5,
    "low_quality": 6,
}

SCHEDULER_MAP = {
    "euler": FlowMatchEulerDiscreteScheduler,
}

pipe_base: FluxPipeline | None = None
pipe_controlnet: FluxControlNetPipeline | None = None
inference_lock = threading.Lock()
reload_lock = threading.Lock()
active_devices: list[int] = []


def _parse_gpu_devices(raw: str) -> list[int]:
    """Parse comma-separated GPU device string into sorted list of ints."""
    return sorted(int(d.strip()) for d in raw.split(",") if d.strip().isdigit())


def _get_gpu_with_most_free_memory() -> int | None:
    """Find GPU with the most free memory using nvidia-smi.

    Returns:
        GPU index with most free memory, or None if detection fails
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        best_gpu = None
        best_free = 0
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                idx = int(parts[0])
                free_mem = int(parts[1])
                if free_mem > best_free:
                    best_free = free_mem
                    best_gpu = idx
        logger.info("Auto-selected GPU %d with %d MiB free", best_gpu, best_free)
        return best_gpu
    except Exception as e:
        logger.warning("Failed to auto-select GPU: %s", e)
        return None


def _load_model(devices: list[int]) -> tuple[FluxPipeline, FluxControlNetPipeline]:
    """Load both pipelines onto the specified GPU(s), sharing base components."""
    # Load shared base components
    vae = AutoencoderKL.from_pretrained(
        BASE_MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    transformer = FluxTransformer2DModel.from_pretrained(
        BASE_MODEL_ID,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    # Load ControlNet model
    controlnet = FluxControlNetModel.from_pretrained(
        CONTROLNET_MODEL_ID,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    logger.info("ControlNet loaded: %s", CONTROLNET_MODEL_ID)

    if len(devices) > 1:
        max_memory = {
            d: f"{torch.cuda.get_device_properties(d).total_memory // (1024**3)}GiB"
            for d in devices
        }
        new_pipe_base = FluxPipeline.from_pretrained(
            BASE_MODEL_ID,
            scheduler=FlowMatchEulerDiscreteScheduler(),
            vae=vae,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            max_memory=max_memory,
            local_files_only=True,
        )
        new_pipe_cn = FluxControlNetPipeline.from_pretrained(
            BASE_MODEL_ID,
            controlnet=controlnet,
            scheduler=FlowMatchEulerDiscreteScheduler(),
            vae=vae,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            max_memory=max_memory,
            local_files_only=True,
        )
        logger.info("Models loaded with device_map='balanced' across GPUs %s", devices)
    else:
        new_pipe_base = FluxPipeline.from_pretrained(
            BASE_MODEL_ID,
            scheduler=FlowMatchEulerDiscreteScheduler(),
            vae=vae,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        new_pipe_base.to(f"cuda:{devices[0]}")

        new_pipe_cn = FluxControlNetPipeline.from_pretrained(
            BASE_MODEL_ID,
            controlnet=controlnet,
            scheduler=FlowMatchEulerDiscreteScheduler(),
            vae=vae,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        new_pipe_cn.to(f"cuda:{devices[0]}")

        logger.info("Models loaded on cuda:%d", devices[0])
    return new_pipe_base, new_pipe_cn


def _get_generator_device() -> str:
    """Return the CUDA device string for the torch Generator."""
    if active_devices:
        return f"cuda:{active_devices[0]}"
    return "cuda"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe_base, pipe_controlnet, active_devices

    raw = os.environ.get("NVIDIA_GPU_DEVICE", "auto")

    if raw.lower() == "auto":
        auto_gpu = _get_gpu_with_most_free_memory()
        if auto_gpu is not None:
            active_devices = [auto_gpu]
        else:
            logger.warning("Auto-selection failed, defaulting to GPU 0")
            active_devices = [0]
    else:
        active_devices = _parse_gpu_devices(raw)

    gpu_count = torch.cuda.device_count()
    logger.info("Starting with %d GPU(s) visible, selection: %s", gpu_count, active_devices)

    pipe_base, pipe_controlnet = _load_model(active_devices)

    yield
    del pipe_base
    del pipe_controlnet
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


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


class GpuConfigRequest(BaseModel):
    devices: list[int]


class Artifact(BaseModel):
    base64: str
    seed: int
    finishReason: str = "SUCCESS"
    errorReason: str = ""


class InferResponse(BaseModel):
    artifacts: list[Artifact]


@app.get("/")
async def root():
    return {
        "model": BASE_MODEL_ID,
        "controlnet": CONTROLNET_MODEL_ID,
        "status": "ready" if pipe_base is not None else "loading",
        "control_modes": list(CONTROL_MODES.keys()),
        "endpoints": {
            "health": "/flux.1-dev/v1/health/ready",
            "infer": "/flux.1-dev/v1/infer",
            "gpu_config": "/flux.1-dev/v1/gpu/config",
        },
    }


@app.get("/v1/health/ready")
async def health_ready():
    if pipe_base is None:
        return {"status": "loading"}
    return {"status": "ready"}


@app.get("/v1/gpu/config")
async def gpu_config_get():
    """Return the currently active GPU device(s)."""
    return {"devices": active_devices}


@app.post("/v1/gpu/config")
async def gpu_config_set(req: GpuConfigRequest):
    """Reload model onto the specified GPU(s)."""
    if not req.devices:
        raise HTTPException(status_code=400, detail="At least one GPU must be selected")

    gpu_count = torch.cuda.device_count()
    for d in req.devices:
        if d >= gpu_count:
            raise HTTPException(
                status_code=400,
                detail=f"GPU {d} not available (only {gpu_count} visible)",
            )

    if sorted(req.devices) == active_devices:
        return {"status": "unchanged", "devices": active_devices}

    if not reload_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Model reload already in progress")

    thread = threading.Thread(
        target=_reload_model, args=(sorted(req.devices),), daemon=True
    )
    thread.start()

    return {"status": "reloading", "devices": sorted(req.devices)}


def _reload_model(devices: list[int]) -> None:
    """Unload current model and reload on new GPU(s). Runs in background thread."""
    global pipe_base, pipe_controlnet, active_devices
    try:
        logger.info(
            "GPU config change: %s -> %s - reloading model...", active_devices, devices
        )

        old_base = pipe_base
        old_cn = pipe_controlnet
        pipe_base = None
        pipe_controlnet = None

        with inference_lock:
            if old_base is not None:
                del old_base
            if old_cn is not None:
                del old_cn
            torch.cuda.empty_cache()

            pipe_base, pipe_controlnet = _load_model(devices)
            active_devices = devices

        logger.info("Model reload complete on GPU(s) %s", devices)
    except Exception:
        logger.exception("Failed to reload model on GPU(s) %s", devices)
    finally:
        reload_lock.release()


@app.post("/v1/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    use_controlnet = bool(req.control_image)
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

    scheduler_cls = SCHEDULER_MAP.get(req.sampler)
    if scheduler_cls:
        current_pipe.scheduler = scheduler_cls.from_config(
            current_pipe.scheduler.config
        )

    generator = torch.Generator(device=_get_generator_device()).manual_seed(req.seed)

    try:
        with inference_lock:
            if use_controlnet:
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


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
