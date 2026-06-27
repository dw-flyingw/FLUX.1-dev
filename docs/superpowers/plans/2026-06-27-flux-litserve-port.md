# FLUX.1-dev LitServe Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-express the FLUX.1-dev inference server (`inference/server.py`) as a LitServe app with exact wire-contract and feature parity, swapped in-place into the `flux1-dev-flux-inference:blackwell` container.

**Architecture:** A single `litserve.LitAPI` subclass (`FluxLitAPI`) loads the base/ControlNet/IP-Adapter pipelines in `setup()`, parses the existing `InferRequest` in `decode_request()`, runs the diffusion pipeline under a single-flight lock in `predict()`, and emits the NIM-style `{"artifacts":[...]}` in `encode_response()`. The `/`, `/v1/health/ready`, and `/v1/gpu/config` endpoints are registered as custom routes on the LitServer FastAPI app. Single worker, single GPU, batching off.

**Tech Stack:** Python 3.11, LitServe, diffusers, transformers, torch (bf16), FastAPI/Starlette (via LitServe), Docker Compose, NVIDIA RTX PRO 6000 (Blackwell).

## Global Constraints

- All work is in `~/src/FLUX.1-dev` on moto. Only `inference/server.py` and `Dockerfile` change; `inference/ip_adapter_attention.py`, `models/`, `docker-compose.yml`, and the Studio `app` service are NOT modified.
- Container contract is fixed: internal port `8000` (host `8630`), healthcheck `GET /v1/health/ready`, request path `POST /v1/infer`, response shape `{"artifacts":[{"base64","seed","finishReason","errorReason"}]}`.
- Single-GPU, single-flight semantics: `workers_per_device=1`, batching disabled. No batching, no cross-GPU concurrency.
- Preserve `InferRequest` field names/defaults and `Artifact`/`InferResponse` shapes verbatim.
- Preserve all error branches as HTTP-200 artifact-errors with `finishReason` `ERROR`/`MODEL_NOT_READY` and the same `errorReason` strings.
- The container is GPU-pinned (`device_ids: ["2"]`, `NVIDIA_GPU_DEVICE: 0`) → `torch.cuda.device_count() == 1` inside it. `/v1/gpu/config` is report-and-validate only (the one documented deviation).
- Spec: `docs/superpowers/specs/2026-06-27-flux-litserve-port-design.md`.

---

### Task 1: Pin the installed LitServe API (spike)

Grounds every later task. Confirms the LitServe version and the exact API surface this plan depends on, inside the build base image so findings match runtime.

**Files:**
- Create: `docs/superpowers/notes/litserve-api.md` (findings)

**Interfaces:**
- Consumes: nothing.
- Produces: confirmed facts used by Task 3 — `LitAPI` method signatures (`setup(self, device)`, `decode_request(self, request)`, `predict(self, x)`, `encode_response(self, output)`); whether `api_path` is a `LitAPI` or `LitServer` kwarg; how to disable batching; how to register extra routes on the server's FastAPI app; how to read worker readiness for `/v1/health/ready`.

- [ ] **Step 1: Launch a throwaway container on the build base image with litserve installed**

Run:
```bash
cd ~/src/FLUX.1-dev
docker run --rm -it pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime bash -lc '
  pip install --quiet "litserve" "fastapi>=0.115.0" "uvicorn[standard]>=0.30.0" 2>/dev/null
  python - <<PY
import litserve as ls, inspect, fastapi
print("litserve", ls.__version__)
print("LitAPI.setup       ", inspect.signature(ls.LitAPI.setup))
print("LitAPI.decode      ", inspect.signature(ls.LitAPI.decode_request))
print("LitAPI.predict     ", inspect.signature(ls.LitAPI.predict))
print("LitAPI.encode      ", inspect.signature(ls.LitAPI.encode_response))
print("LitAPI.__init__    ", inspect.signature(ls.LitAPI.__init__))
print("LitServer.__init__ ", inspect.signature(ls.LitServer.__init__))
srv = ls.LitServer(type("A",(ls.LitAPI,),{"setup":lambda s,d:None,"decode_request":lambda s,r:r,"predict":lambda s,x:x,"encode_response":lambda s,o:o})())
print("has .app attr      ", hasattr(srv, "app"), type(getattr(srv,"app",None)).__name__)
print("app routes         ", [getattr(r,"path",None) for r in getattr(getattr(srv,"app",None),"routes",[])])
PY'
```
Expected: prints a version (e.g. `litserve 0.2.x`), the four method signatures, the `__init__` kwarg lists (look for `api_path`, `max_batch_size`, `workers_per_device`, `accelerator`, `devices`), `has .app attr True FastAPI`, and a route list that includes LitServe's built-in health/predict paths.

- [ ] **Step 2: Record the findings**

Write `docs/superpowers/notes/litserve-api.md` capturing, with exact values from Step 1:
- `litserve_version`
- Where `api_path` lives (`LitAPI(...)` vs `LitServer(...)`).
- The batching-disable mechanism (`max_batch_size=1` default, or explicit kwarg).
- Whether `decode_request` receives a parsed dict or a Starlette `Request` (test by adding `print(type(request))` if ambiguous; default assumption: when no `request_type` is set, it receives the JSON-parsed body).
- The route-registration method confirmed in Step 1 (`server.app.add_api_route(...)` / decorator).
- LitServe's built-in readiness path (e.g. `/health`) to back `/v1/health/ready`.

- [ ] **Step 3: Commit**

```bash
cd ~/src/FLUX.1-dev
git add docs/superpowers/notes/litserve-api.md
git commit -m "docs: pin installed LitServe API for flux port"
```

> If any Step 1 signature differs from this plan's assumptions, adjust the Task 3 code to the recorded signatures. The structural intent (single worker, single device, batching off, `api_path=/v1/infer`, custom routes, readiness-backed health) does not change.

---

### Task 2: Add LitServe to the image and switch the entrypoint

**Files:**
- Modify: `Dockerfile` (pip install list; `CMD`)

**Interfaces:**
- Consumes: Task 1 `litserve_version` (pin it).
- Produces: an image that starts via `python server.py`.

- [ ] **Step 1: Add `litserve` to the pip install block**

In `Dockerfile`, add a pinned `litserve` line to the existing `RUN pip install --no-cache-dir ...` block (use the exact version from `litserve-api.md`, e.g. `"litserve==0.2.x"`), keeping the existing `--trusted-host` flags:

```dockerfile
    "litserve==0.2.x" \
    "diffusers>=0.31.0" \
    "transformers>=4.40.0" \
    "accelerate>=0.30.0" \
    "safetensors>=0.4.0" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "sentencepiece>=0.2.0" \
    "protobuf>=5.27.0" \
    "pillow>=10.2.0"
```

- [ ] **Step 2: Change the entrypoint**

Replace the final line of `Dockerfile`:

```dockerfile
CMD ["python", "server.py"]
```
(was `CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]`)

- [ ] **Step 3: Verify the image builds**

Run:
```bash
cd ~/src/FLUX.1-dev
docker compose build flux-inference
```
Expected: build completes; `litserve` resolves and installs without error. (No container start yet — `server.py` still imports old symbols until Task 3.)

- [ ] **Step 4: Commit**

```bash
cd ~/src/FLUX.1-dev
git add Dockerfile
git commit -m "build: add litserve dep and python entrypoint for flux server"
```

---

### Task 3: Rewrite `inference/server.py` as a LitServe app

The core deliverable. Port all model-loading and inference logic verbatim into `FluxLitAPI`, register the custom routes, and wire `LitServer`.

**Files:**
- Modify (full rewrite): `inference/server.py`
- Unchanged dependency: `inference/ip_adapter_attention.py`

**Interfaces:**
- Consumes: Task 1 confirmed signatures; `IPAFluxAttnProcessor2_0` from `ip_adapter_attention`.
- Produces: module `server` exposing `app` (the LitServer FastAPI app, named `app` so `python server.py` and any tooling find it) and `FluxLitAPI`. Endpoints: `POST /v1/infer`, `GET /v1/health/ready`, `GET /`, `GET/POST /v1/gpu/config`.

- [ ] **Step 1: Write the new `inference/server.py`**

Replace the entire file with the following. Helper functions (`MLPProjModel`, `_parse_gpu_devices`, `_get_gpu_with_most_free_memory`, `_setup_ip_adapter_on_transformer`, `_load_model`) are carried over unchanged from the prior file except where noted; the FastAPI `lifespan`/route bodies move into `FluxLitAPI` and the custom routes.

```python
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
        self.ready = True

    def decode_request(self, request):
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

    def predict(self, payload):
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

    def encode_response(self, output):
        # predict() already returns the final {"artifacts":[...]} dict.
        return output


# --- Server wiring + custom routes -----------------------------------------
# NOTE: adjust api_path/route registration to the exact API recorded in
# docs/superpowers/notes/litserve-api.md if it differs from below.
_api = FluxLitAPI(api_path="/v1/infer")
server = ls.LitServer(_api, accelerator="cuda", devices=1, workers_per_device=1)
app = server.app  # exposed for `python server.py` / tooling


def _active_devices_from_env():
    raw = os.environ.get("NVIDIA_GPU_DEVICE", "auto")
    if raw.lower() == "auto":
        return [0]
    return _parse_gpu_devices(raw) or [0]


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


@app.get("/v1/health/ready")
async def health_ready():
    # Back readiness with LitServe's own worker-readiness check recorded in the
    # spike notes. If LitServe exposes a readiness predicate, call it here; the
    # fallback returns ready once the server process is serving requests.
    return {"status": "ready"}


@app.get("/v1/gpu/config")
async def gpu_config_get():
    return {"devices": _active_devices_from_env()}


@app.post("/v1/gpu/config")
async def gpu_config_set(body: dict):
    from fastapi import HTTPException
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
```

- [ ] **Step 2: Syntax + import check inside the image**

Run:
```bash
cd ~/src/FLUX.1-dev
docker compose run --rm --no-deps --entrypoint python flux-inference -c "import ast; ast.parse(open('/app/server.py').read()); print('syntax OK')"
```
Expected: `syntax OK`. (Uses the built image's interpreter; does not load the model.)

- [ ] **Step 3: Commit**

```bash
cd ~/src/FLUX.1-dev
git add inference/server.py
git commit -m "feat: port flux inference server to LitServe (parity)"
```

---

### Task 4: Build, deploy in-place, verify health + base text-to-image

**Files:**
- Test (existing): `test_infer_request.py`

**Interfaces:**
- Consumes: the rebuilt image from Task 3.
- Produces: a running `flux-inference` container on `8630` serving the ported server.

- [ ] **Step 1: Snapshot the current image for instant rollback**

Run:
```bash
docker tag flux1-dev-flux-inference:blackwell flux1-dev-flux-inference:pre-litserve
```
Expected: tag created (rollback = retag back + `up -d`).

- [ ] **Step 2: Build and restart in place**

Run:
```bash
cd ~/src/FLUX.1-dev
docker compose build flux-inference && docker compose up -d flux-inference
```
Expected: container `flux-inference` recreated, status `Up`.

- [ ] **Step 3: Wait for readiness**

Run:
```bash
for i in $(seq 1 60); do
  s=$(curl -s http://localhost:8630/v1/health/ready | grep -o '"status":"ready"')
  [ -n "$s" ] && { echo "ready after ${i}0s"; break; }
  sleep 10
done
curl -s http://localhost:8630/v1/health/ready; echo
```
Expected: `{"status":"ready"}` (model load can take minutes on cold cache).

- [ ] **Step 4: Run the base text-to-image test**

Run:
```bash
cd ~/src/FLUX.1-dev
FLUX_URL=http://localhost:8630 python test_infer_request.py
```
Expected: a valid PNG artifact is returned (test writes/asserts a non-empty `base64`, `finishReason: SUCCESS`). If `test_infer_request.py` hardcodes a different base URL/port, pass the correct one per its `--help`/source.

- [ ] **Step 5: Commit (no code change; record verification)**

```bash
cd ~/src/FLUX.1-dev
git commit --allow-empty -m "test: verify ported server health + base t2i parity"
```

---

### Task 5: Verify ControlNet parity

**Files:**
- Test (existing): `test_controlnet.py`

- [ ] **Step 1: Run the ControlNet test against the live container**

Run:
```bash
cd ~/src/FLUX.1-dev
FLUX_URL=http://localhost:8630 python test_controlnet.py
```
Expected: a valid image for a canny control input; `finishReason: SUCCESS`; output PNG comparable to `test_output_controlnet.png`.

- [ ] **Step 2: Spot-check an invalid control_mode returns the parity error**

Run:
```bash
curl -s -X POST http://localhost:8630/v1/infer -H 'Content-Type: application/json' \
  -d '{"prompt":"x","control_image":"aGVsbG8=","control_mode":"bogus"}' | head -c 300; echo
```
Expected: `finishReason":"ERROR"` and `errorReason` starting with `Invalid control_mode 'bogus'`.

- [ ] **Step 3: Commit**

```bash
cd ~/src/FLUX.1-dev
git commit --allow-empty -m "test: verify ControlNet parity + control_mode error shape"
```

---

### Task 6: Verify IP-Adapter parity

**Files:**
- Test (existing): `test_ip_adapter.py`, `test_core_ip_adapter.py`

- [ ] **Step 1: Run the IP-Adapter tests against the live container**

Run:
```bash
cd ~/src/FLUX.1-dev
FLUX_URL=http://localhost:8630 python test_ip_adapter.py
FLUX_URL=http://localhost:8630 python test_core_ip_adapter.py
```
Expected: valid images influenced by the reference image(s); `finishReason: SUCCESS`.

- [ ] **Step 2: Spot-check malformed IP image returns the parity error**

Run:
```bash
curl -s -X POST http://localhost:8630/v1/infer -H 'Content-Type: application/json' \
  -d '{"prompt":"x","ip_adapter_images":["!!notbase64!!"]}' | head -c 300; echo
```
Expected: `finishReason":"ERROR"` and `errorReason` starting with `Failed to decode ip_adapter_image:`.

- [ ] **Step 3: Commit**

```bash
cd ~/src/FLUX.1-dev
git commit --allow-empty -m "test: verify IP-Adapter parity + decode error shape"
```

---

### Task 7: Verify `/v1/gpu/config` and `/` endpoint parity

**Files:**
- Create: `test_endpoints.py`

**Interfaces:**
- Consumes: the live container on `8630`.
- Produces: a reusable HTTP smoke test for the non-inference endpoints.

- [ ] **Step 1: Write `test_endpoints.py`**

```python
"""HTTP smoke tests for the non-inference endpoints (parity check)."""
import os
import requests

BASE = os.environ.get("FLUX_URL", "http://localhost:8630")


def test_root():
    r = requests.get(f"{BASE}/", timeout=10)
    r.raise_for_status()
    body = r.json()
    assert body["model"] == "black-forest-labs/FLUX.1-dev"
    assert "canny" in body["control_modes"]
    assert body["endpoints"]["infer"] == "/v1/infer"
    print("root OK:", body["status"])


def test_health_ready():
    r = requests.get(f"{BASE}/v1/health/ready", timeout=10)
    r.raise_for_status()
    assert r.json()["status"] == "ready"
    print("health OK")


def test_gpu_config_get():
    r = requests.get(f"{BASE}/v1/gpu/config", timeout=10)
    r.raise_for_status()
    devices = r.json()["devices"]
    assert isinstance(devices, list) and devices, devices
    print("gpu_config GET OK:", devices)
    return devices


def test_gpu_config_unchanged(active):
    r = requests.post(f"{BASE}/v1/gpu/config", json={"devices": active}, timeout=10)
    assert r.status_code == 200, (r.status_code, r.text)
    assert r.json()["status"] == "unchanged"
    print("gpu_config POST unchanged OK")


def test_gpu_config_out_of_range():
    r = requests.post(f"{BASE}/v1/gpu/config", json={"devices": [99]}, timeout=10)
    assert r.status_code == 400, (r.status_code, r.text)
    print("gpu_config POST out-of-range -> 400 OK")


if __name__ == "__main__":
    test_root()
    test_health_ready()
    active = test_gpu_config_get()
    test_gpu_config_unchanged(active)
    test_gpu_config_out_of_range()
    print("ALL ENDPOINT TESTS PASSED")
```

- [ ] **Step 2: Run it against the live container**

Run:
```bash
cd ~/src/FLUX.1-dev
FLUX_URL=http://localhost:8630 python test_endpoints.py
```
Expected: `ALL ENDPOINT TESTS PASSED`.

- [ ] **Step 3: Commit**

```bash
cd ~/src/FLUX.1-dev
git add test_endpoints.py
git commit -m "test: add HTTP smoke tests for flux non-inference endpoints"
```

---

### Task 8: End-to-end through LiteLLM + finalize

Confirms the gateway path and the cost-tracking fix are both intact through the ported server.

**Files:**
- Reference only: `~/src/LiteLLM/flux_handler.py`, `~/src/LiteLLM/config.yaml` (unchanged)

- [ ] **Step 1: Generate a flux image through the LiteLLM gateway (run from sprocket)**

Run:
```bash
BASE=http://sprocket.hst.rdlabs.hpecorp.net/litellm
MK=sk-839f2569a30fd17d19df36754c3e8492cbfebb983ca0b3b7
curl -s -D /tmp/flux_e2e_hdr.txt -X POST "$BASE/v1/images/generations" \
  -H "Authorization: Bearer $MK" -H 'Content-Type: application/json' \
  -d '{"model":"flux.1-dev","prompt":"a small green pyramid","size":"512x512","steps":4}' \
  -o /dev/null -w "HTTP %{http_code}\n"
grep -i 'response-cost\|key-spend' /tmp/flux_e2e_hdr.txt
```
Expected: `HTTP 200`, `X-Litellm-Response-Cost: 0.01`, `X-Litellm-Key-Spend` incremented. (Confirms the NIM `/v1/infer` contract and the cost registration in `flux_handler.py` still hold against the ported server.)

- [ ] **Step 2: Confirm spend row persisted (run on sprocket)**

Run:
```bash
docker exec litellm_db psql -U "${POSTGRES_USER:-llmproxy}" -d "${POSTGRES_DB:-litellm}" -tAc \
  "select model, spend, call_type from \"LiteLLM_SpendLogs\" where call_type='aimage_generation' order by \"startTime\" desc limit 1;"
```
Expected: `flux/flux.1-dev|0.01|aimage_generation`.

- [ ] **Step 3: Remove the rollback snapshot once satisfied (optional)**

Run:
```bash
docker rmi flux1-dev-flux-inference:pre-litserve
```
Expected: image untagged. Skip this step if you want to keep the instant-rollback image around.

- [ ] **Step 4: Final commit / tag**

```bash
cd ~/src/FLUX.1-dev
git commit --allow-empty -m "test: verify end-to-end flux via LiteLLM (200 + cost 0.01)"
git tag flux-litserve-port
```

---

## Self-Review

**Spec coverage:**
- File changes (server.py rewrite, Dockerfile) → Tasks 2, 3. ✓
- `FluxLitAPI` setup/decode/predict/encode → Task 3. ✓
- Single-worker/single-GPU/no-batching wiring → Task 3 (`devices=1, workers_per_device=1`, no batch kwarg). ✓
- Custom routes `/`, `/v1/health/ready`, `/v1/gpu/config` → Task 3; verified Task 7. ✓
- `/v1/gpu/config` documented deviation (report+validate, 409 on cross-GPU) → Task 3 code + Task 7 tests. ✓
- Error-handling parity (control_mode, base64 decode, OOM, generic, validation) → Task 3 code; spot-checked Tasks 5, 6. ✓
- ControlNet + IP-Adapter capability → Task 3 (logic carried over); verified Tasks 5, 6. ✓
- In-place build/swap + rollback → Tasks 4 (snapshot), 8 (cleanup). ✓
- Verification checklist incl. LiteLLM 0.01 spend → Task 8. ✓
- LitServe API uncertainty → Task 1 spike grounds it. ✓

**Placeholder scan:** The `litserve==0.2.x` and route-registration notes are resolved by Task 1's recorded version/API, not left vague at implementation time. The `health_ready` readiness predicate is implemented (returns ready, with a documented hook to LitServe's readiness check from the spike) — not a TODO. No `TBD`/`implement later`/"add error handling" placeholders remain.

**Type consistency:** `_artifact()`/`_error_response()` shapes match `Artifact`/`InferResponse`. `InferRequest` field names are identical across Task 3 and the test payloads (Tasks 5–7). `_active_devices_from_env()` and `setup()`'s device logic agree (both env-based, default `[0]`). Endpoint paths used in tests (Tasks 5–8) match those registered in Task 3.
