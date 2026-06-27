"""LitServe inference service for FLUX.1-dev with ControlNet + IP-Adapter.

Faithful port of the prior FastAPI server. Same /v1/infer schema, same
/v1/health/ready, /, /v1/gpu/config endpoints, same single-GPU single-flight
semantics. See docs/superpowers/specs/2026-06-27-flux-litserve-port-design.md.
"""

import base64
import io
import logging
import os
import threading

import litserve as ls
import torch
import torch.nn as nn
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxControlNetModel,
    FluxControlNetPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
)
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoProcessor, SiglipVisionModel

from fastapi import HTTPException

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
    "canny": 0, "tile": 1, "depth": 2, "blur": 3,
    "pose": 4, "gray": 5, "low_quality": 6,
}
SCHEDULER_MAP = {"euler": FlowMatchEulerDiscreteScheduler}


class MLPProjModel(nn.Module):
    """Projects SigLIP embeddings into the IP-Adapter cross-attention space."""

    def __init__(self, cross_attention_dim=4096, id_embeddings_dim=1152, num_tokens=128):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        return self.norm(x)


def _parse_gpu_devices(raw: str) -> list[int]:
    return sorted(int(d.strip()) for d in raw.split(",") if d.strip().isdigit())


def _setup_ip_adapter_on_transformer(transformer, ip_state_dict) -> MLPProjModel:
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

    ip_layers = nn.ModuleList(
        [p for p in transformer.attn_processors.values() if isinstance(p, IPAFluxAttnProcessor2_0)]
    )
    ip_layers.load_state_dict(ip_state_dict["ip_adapter"], strict=False)
    logger.info("IP-Adapter attention weights loaded (%d processors)", len(ip_layers))

    proj_model = MLPProjModel(
        cross_attention_dim=transformer.config.joint_attention_dim,
        id_embeddings_dim=1152, num_tokens=IP_ADAPTER_NUM_TOKENS,
    )
    proj_model.load_state_dict(ip_state_dict["image_proj"])
    return proj_model


def _load_model(devices: list[int]):
    """Load all pipelines + IP-Adapter components onto the given GPU(s).

    Identical logic to the prior server. Returns
    (pipe_base, pipe_controlnet, siglip, siglip_processor, proj_model).
    """
    ip_ckpt_path = os.path.join(IP_ADAPTER_LOCAL_DIR, IP_ADAPTER_WEIGHT_NAME)
    logger.info("Loading IP-Adapter weights from %s", ip_ckpt_path)
    ip_state_dict = torch.load(ip_ckpt_path, map_location="cpu", weights_only=False)

    vae = AutoencoderKL.from_pretrained(
        BASE_MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        BASE_MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True,
    )
    proj_model = _setup_ip_adapter_on_transformer(transformer, ip_state_dict)
    controlnet = FluxControlNetModel.from_pretrained(
        CONTROLNET_MODEL_ID, torch_dtype=torch.bfloat16, local_files_only=True,
    )
    logger.info("ControlNet loaded: %s", CONTROLNET_MODEL_ID)

    if len(devices) > 1:
        max_memory = {
            d: f"{torch.cuda.get_device_properties(d).total_memory // (1024**3)}GiB"
            for d in devices
        }
        pipe_base = FluxPipeline.from_pretrained(
            BASE_MODEL_ID, scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae,
            transformer=transformer, torch_dtype=torch.bfloat16,
            device_map="balanced", max_memory=max_memory, local_files_only=True,
        )
        pipe_cn = FluxControlNetPipeline.from_pretrained(
            BASE_MODEL_ID, controlnet=controlnet, scheduler=FlowMatchEulerDiscreteScheduler(),
            vae=vae, transformer=transformer, torch_dtype=torch.bfloat16,
            device_map="balanced", max_memory=max_memory, local_files_only=True,
        )
        primary_device = f"cuda:{devices[0]}"
        for proc in transformer.attn_processors.values():
            if isinstance(proc, IPAFluxAttnProcessor2_0):
                proc.to(torch.bfloat16)
        proj_model = proj_model.to(primary_device, dtype=torch.bfloat16)
        logger.info("Models loaded device_map='balanced' across GPUs %s", devices)
    else:
        primary_device = f"cuda:{devices[0]}"
        pipe_base = FluxPipeline.from_pretrained(
            BASE_MODEL_ID, scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae,
            transformer=transformer, torch_dtype=torch.bfloat16, local_files_only=True,
        )
        pipe_base.to(primary_device)
        pipe_cn = FluxControlNetPipeline.from_pretrained(
            BASE_MODEL_ID, controlnet=controlnet, scheduler=FlowMatchEulerDiscreteScheduler(),
            vae=vae, transformer=transformer, torch_dtype=torch.bfloat16, local_files_only=True,
        )
        pipe_cn.to(primary_device)
        for proc in transformer.attn_processors.values():
            if isinstance(proc, IPAFluxAttnProcessor2_0):
                proc.to(primary_device, dtype=torch.bfloat16)
        proj_model = proj_model.to(primary_device, dtype=torch.bfloat16)
        logger.info("Models loaded on %s", primary_device)

    siglip = SiglipVisionModel.from_pretrained(
        SIGLIP_MODEL_ID, torch_dtype=torch.bfloat16, local_files_only=True,
    ).to(primary_device).eval()
    siglip_proc = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID, local_files_only=True)
    logger.info("SigLIP loaded on %s", primary_device)
    return pipe_base, pipe_cn, siglip, siglip_proc, proj_model


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


def _artifact(base64_str="", seed=0, finish="SUCCESS", error=""):
    return {"base64": base64_str, "seed": seed, "finishReason": finish, "errorReason": error}


def _error_response(reason: str, finish: str = "ERROR"):
    return {"artifacts": [_artifact(finish=finish, error=reason)]}


class FluxLitAPI(ls.LitAPI):
    def setup(self, device):
        # Honor the prior NVIDIA_GPU_DEVICE selection for parity. In the pinned
        # container only one GPU is visible, so this resolves to [0]; LitServe's
        # `device` arg points at the same physical GPU.
        raw = os.environ.get("NVIDIA_GPU_DEVICE", "auto")
        if raw.lower() == "auto":
            self.active_devices = [0]
        else:
            self.active_devices = _parse_gpu_devices(raw) or [0]
        logger.info("setup() device=%s active_devices=%s gpu_count=%d",
                    device, self.active_devices, torch.cuda.device_count())
        (self.pipe_base, self.pipe_controlnet, self.siglip_model,
         self.siglip_processor, self.image_proj_model) = _load_model(self.active_devices)
        self.inference_lock = threading.Lock()

    def decode_request(self, request, **kwargs):
        # `request` is the JSON body (dict). Validation errors become artifact
        # errors in predict() to preserve the wire contract.
        return request

    def _generator_device(self):
        return f"cuda:{self.active_devices[0]}" if self.active_devices else "cuda"

    def _compute_image_emb(self, pil_images, device):
        inputs = self.siglip_processor(images=pil_images, return_tensors="pt").pixel_values
        inputs = inputs.to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            siglip_embeds = self.siglip_model(inputs).pooler_output
            image_emb = self.image_proj_model(siglip_embeds)
        return image_emb

    def predict(self, x, **kwargs):
        payload = x
        try:
            req = InferRequest(**payload)
        except ValidationError as e:
            return _error_response(f"Invalid request: {e.errors()}")

        use_controlnet = bool(req.control_image)
        use_ip_adapter = bool(req.ip_adapter_images)
        current_pipe = self.pipe_controlnet if use_controlnet else self.pipe_base
        if current_pipe is None:
            return _error_response("Model is still loading", finish="MODEL_NOT_READY")

        control_image = None
        if use_controlnet:
            if req.control_mode not in CONTROL_MODES:
                return _error_response(
                    f"Invalid control_mode '{req.control_mode}'. Must be one of: {list(CONTROL_MODES.keys())}"
                )
            try:
                control_bytes = base64.b64decode(req.control_image)
                control_image = Image.open(io.BytesIO(control_bytes)).convert("RGB").resize((req.width, req.height))
            except Exception as e:
                return _error_response(f"Failed to decode control_image: {e}")

        ip_images = []
        if use_ip_adapter:
            try:
                for img_b64 in req.ip_adapter_images:
                    ip_images.append(Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB"))
            except Exception as e:
                return _error_response(f"Failed to decode ip_adapter_image: {e}")

        device = self._generator_device()
        generator = torch.Generator(device=device).manual_seed(req.seed)
        try:
            with self.inference_lock:
                scheduler_cls = SCHEDULER_MAP.get(req.sampler)
                if scheduler_cls:
                    current_pipe.scheduler = scheduler_cls.from_config(current_pipe.scheduler.config)
                if use_ip_adapter:
                    image_emb = self._compute_image_emb(ip_images, device)
                    for proc in current_pipe.transformer.attn_processors.values():
                        if isinstance(proc, IPAFluxAttnProcessor2_0):
                            proc.scale = req.adapter_strength
                else:
                    image_emb = None
                jat_kwargs = {"image_emb": image_emb}
                if use_controlnet:
                    result = current_pipe(
                        prompt=req.prompt,
                        negative_prompt=req.negative_prompt or None,
                        control_image=control_image,
                        control_mode=CONTROL_MODES[req.control_mode],
                        controlnet_conditioning_scale=req.controlnet_conditioning_scale,
                        width=req.width, height=req.height,
                        num_inference_steps=req.steps, guidance_scale=req.guidance_scale,
                        generator=generator, joint_attention_kwargs=jat_kwargs,
                    )
                else:
                    result = current_pipe(
                        prompt=req.prompt,
                        negative_prompt=req.negative_prompt or None,
                        width=req.width, height=req.height,
                        num_inference_steps=req.steps, guidance_scale=req.guidance_scale,
                        generator=generator, joint_attention_kwargs=jat_kwargs,
                    )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error("CUDA OOM during inference: resolution=%dx%d steps=%d", req.width, req.height, req.steps)
            return _error_response(
                f"GPU out of memory. Try reducing resolution ({req.width}x{req.height}) or steps ({req.steps})."
            )
        except Exception as e:
            logger.exception("Unhandled error during inference")
            return _error_response(f"Pipeline error: {type(e).__name__}: {e}")

        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"artifacts": [_artifact(base64_str=b64, seed=req.seed)]}

    def encode_response(self, output, **kwargs):
        # predict() already returns the final {"artifacts":[...]} dict.
        return output


# --- Server wiring + custom routes -----------------------------------------
_api = FluxLitAPI(api_path="/v1/infer", max_batch_size=1)
server = ls.LitServer(_api, accelerator="cuda", devices=1, workers_per_device=1,
                      healthcheck_path="/v1/health/ready")
app = server.app  # exposed for `python server.py` / tooling


def _active_devices_from_env():
    raw = os.environ.get("NVIDIA_GPU_DEVICE", "auto")
    if raw.lower() == "auto":
        return [0]
    return _parse_gpu_devices(raw) or [0]


# Remove LitServe's pre-registered "/" route before adding our custom model-summary.
app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/"]


@app.get("/")
async def root():
    return {
        "model": BASE_MODEL_ID,
        "controlnet": CONTROLNET_MODEL_ID,
        "ip_adapter": IP_ADAPTER_MODEL_ID,
        "status": "ready",
        "control_modes": list(CONTROL_MODES.keys()),
        "endpoints": {"health": "/v1/health/ready", "infer": "/v1/infer", "gpu_config": "/v1/gpu/config"},
    }


@app.get("/v1/gpu/config")
async def gpu_config_get():
    return {"devices": _active_devices_from_env()}


@app.post("/v1/gpu/config")
async def gpu_config_set(body: dict):
    devices = body.get("devices") or []
    if not devices:
        raise HTTPException(status_code=400, detail="At least one GPU must be selected")
    gpu_count = torch.cuda.device_count()
    for d in devices:
        if d >= gpu_count:
            raise HTTPException(status_code=400, detail=f"GPU {d} not available (only {gpu_count} visible)")
    active = _active_devices_from_env()
    if sorted(devices) == sorted(active):
        return {"status": "unchanged", "devices": active}
    # Documented deviation: cross-GPU live reload is not supported in the pinned
    # single-GPU LitServe deployment.
    raise HTTPException(
        status_code=409,
        detail="Runtime cross-GPU reload is not supported in this deployment (GPU is pinned). "
               "Un-pin the container and redeploy to change GPUs.",
    )


if __name__ == "__main__":
    server.run(port=8000)
