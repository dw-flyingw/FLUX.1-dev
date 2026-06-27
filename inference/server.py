"""FastAPI inference service for FLUX.1-dev with optional ControlNet and IP-Adapter."""

import base64
import io
import logging
import os
import subprocess
import threading
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
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
from transformers import AutoProcessor, SiglipVisionModel

from ip_adapter_attention import IPAFluxAttnProcessor2_0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL_ID = "black-forest-labs/FLUX.1-dev"
CONTROLNET_MODEL_ID = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"
IP_ADAPTER_MODEL_ID = "InstantX/FLUX.1-dev-IP-Adapter"
IP_ADAPTER_WEIGHT_NAME = "ip-adapter.bin"
IP_ADAPTER_LOCAL_DIR = "models/ip-adapter"
SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"
IP_ADAPTER_NUM_TOKENS = 128

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
siglip_model: SiglipVisionModel | None = None
siglip_processor = None
image_proj_model = None
inference_lock = threading.Lock()
reload_lock = threading.Lock()
active_devices: list[int] = []


class MLPProjModel(nn.Module):
    """Projects SigLIP embeddings into the IP-Adapter cross-attention space."""

    def __init__(self, cross_attention_dim: int = 4096, id_embeddings_dim: int = 1152, num_tokens: int = 128):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds: torch.Tensor) -> torch.Tensor:
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


def _parse_gpu_devices(raw: str) -> list[int]:
    """Parse comma-separated GPU device string into sorted list of ints."""
    return sorted(int(d.strip()) for d in raw.split(",") if d.strip().isdigit())


def _get_gpu_with_most_free_memory() -> int | None:
    """Find GPU with the most free memory using nvidia-smi."""
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


def _setup_ip_adapter_on_transformer(transformer: FluxTransformer2DModel, ip_state_dict: dict) -> MLPProjModel:
    """Replace attention processors with IP-Adapter variants and load weights."""
    ip_attn_procs = {}
    for name in transformer.attn_processors.keys():
        if name.startswith("transformer_blocks.") or name.startswith("single_transformer_blocks"):
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                hidden_size=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
                cross_attention_dim=transformer.config.joint_attention_dim,
                num_tokens=IP_ADAPTER_NUM_TOKENS,
            )
        else:
            ip_attn_procs[name] = transformer.attn_processors[name]
    transformer.set_attn_processor(ip_attn_procs)

    # Load IP-Adapter attention weights
    ip_layers = nn.ModuleList(
        [proc for proc in transformer.attn_processors.values() if isinstance(proc, IPAFluxAttnProcessor2_0)]
    )
    ip_layers.load_state_dict(ip_state_dict["ip_adapter"], strict=False)
    logger.info("IP-Adapter attention weights loaded (%d processors)", len(ip_layers))

    # Build and load image projection model
    proj_model = MLPProjModel(
        cross_attention_dim=transformer.config.joint_attention_dim,
        id_embeddings_dim=1152,
        num_tokens=IP_ADAPTER_NUM_TOKENS,
    )
    proj_model.load_state_dict(ip_state_dict["image_proj"])
    return proj_model


def _load_model(
    devices: list[int],
) -> tuple[FluxPipeline, FluxControlNetPipeline, SiglipVisionModel, object, MLPProjModel]:
    """Load all pipelines and IP-Adapter components onto specified GPU(s)."""
    # Load ip-adapter checkpoint
    ip_ckpt_path = os.path.join(IP_ADAPTER_LOCAL_DIR, IP_ADAPTER_WEIGHT_NAME)
    logger.info("Loading IP-Adapter weights from %s", ip_ckpt_path)
    ip_state_dict = torch.load(ip_ckpt_path, map_location="cpu", weights_only=False)

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

    # Set up IP-Adapter on transformer (modifies attn_processors in-place)
    proj_model = _setup_ip_adapter_on_transformer(transformer, ip_state_dict)

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

        primary_device = f"cuda:{devices[0]}"
        # Move IP-Adapter processors to primary device (transformer may be split, processors go with their layers)
        for proc in transformer.attn_processors.values():
            if isinstance(proc, IPAFluxAttnProcessor2_0):
                proc.to(torch.bfloat16)
        proj_model = proj_model.to(primary_device, dtype=torch.bfloat16)
    else:
        primary_device = f"cuda:{devices[0]}"
        new_pipe_base = FluxPipeline.from_pretrained(
            BASE_MODEL_ID,
            scheduler=FlowMatchEulerDiscreteScheduler(),
            vae=vae,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        new_pipe_base.to(primary_device)

        new_pipe_cn = FluxControlNetPipeline.from_pretrained(
            BASE_MODEL_ID,
            controlnet=controlnet,
            scheduler=FlowMatchEulerDiscreteScheduler(),
            vae=vae,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        new_pipe_cn.to(primary_device)
        logger.info("Models loaded on %s", primary_device)

        # Move IP-Adapter processors and proj model to device
        for proc in transformer.attn_processors.values():
            if isinstance(proc, IPAFluxAttnProcessor2_0):
                proc.to(primary_device, dtype=torch.bfloat16)
        proj_model = proj_model.to(primary_device, dtype=torch.bfloat16)

    # Load SigLIP image encoder from HF cache
    logger.info("Loading SigLIP image encoder: %s", SIGLIP_MODEL_ID)
    siglip = SiglipVisionModel.from_pretrained(
        SIGLIP_MODEL_ID, torch_dtype=torch.bfloat16, local_files_only=True
    )
    siglip = siglip.to(primary_device).eval()
    siglip_proc = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID, local_files_only=True)
    logger.info("SigLIP loaded on %s", primary_device)

    return new_pipe_base, new_pipe_cn, siglip, siglip_proc, proj_model


def _get_generator_device() -> str:
    """Return the CUDA device string for the torch Generator."""
    if active_devices:
        return f"cuda:{active_devices[0]}"
    return "cuda"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe_base, pipe_controlnet, siglip_model, siglip_processor, image_proj_model, active_devices

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

    pipe_base, pipe_controlnet, siglip_model, siglip_processor, image_proj_model = _load_model(active_devices)

    yield
    del pipe_base
    del pipe_controlnet
    del siglip_model
    del image_proj_model
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
        "ip_adapter": IP_ADAPTER_MODEL_ID,
        "status": "ready" if pipe_base is not None else "loading",
        "control_modes": list(CONTROL_MODES.keys()),
        "endpoints": {
            "health": "/v1/health/ready",
            "infer": "/v1/infer",
            "gpu_config": "/v1/gpu/config",
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
    global pipe_base, pipe_controlnet, siglip_model, siglip_processor, image_proj_model, active_devices
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

            pipe_base, pipe_controlnet, siglip_model, siglip_processor, image_proj_model = _load_model(devices)
            active_devices = devices

        logger.info("Model reload complete on GPU(s) %s", devices)
    except Exception:
        logger.exception("Failed to reload model on GPU(s) %s", devices)
    finally:
        reload_lock.release()


def _compute_image_emb(pil_images: list[Image.Image], device: str) -> torch.Tensor:
    """Encode reference images with SigLIP and project to IP-Adapter embedding space."""
    inputs = siglip_processor(images=pil_images, return_tensors="pt").pixel_values
    inputs = inputs.to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        siglip_embeds = siglip_model(inputs).pooler_output  # [n, 1152]
        image_emb = image_proj_model(siglip_embeds)  # [n, 128, 4096]
    return image_emb


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

    generator = torch.Generator(device=_get_generator_device()).manual_seed(req.seed)
    device = _get_generator_device()

    try:
        with inference_lock:
            # Mutate shared pipeline state inside the lock
            scheduler_cls = SCHEDULER_MAP.get(req.sampler)
            if scheduler_cls:
                current_pipe.scheduler = scheduler_cls.from_config(
                    current_pipe.scheduler.config
                )

            # Compute image embeddings (or None) for IP-Adapter
            if use_ip_adapter:
                image_emb = _compute_image_emb(ip_adapter_pil_images, device)
                for proc in current_pipe.transformer.attn_processors.values():
                    if isinstance(proc, IPAFluxAttnProcessor2_0):
                        proc.scale = req.adapter_strength
            else:
                image_emb = None

            # image_emb=None disables IP-Adapter conditioning in the processor
            jat_kwargs = {"image_emb": image_emb}

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
                    joint_attention_kwargs=jat_kwargs,
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
                    joint_attention_kwargs=jat_kwargs,
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
