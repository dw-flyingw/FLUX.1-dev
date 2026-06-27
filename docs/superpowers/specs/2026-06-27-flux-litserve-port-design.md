# FLUX.1-dev inference server: port to LitServe (Approach A)

- **Date:** 2026-06-27
- **Repo:** `~/src/FLUX.1-dev` (on moto)
- **Status:** Approved design, pending implementation plan
- **Author:** Dave Wright (with Claude Code)

## Summary

Re-express the existing FLUX.1-dev FastAPI inference server (`inference/server.py`)
as a [LitServe](https://github.com/Lightning-AI/LitServe) application, preserving
**full feature parity** and the **exact wire contract**. This is a faithful,
minimal-risk port: same endpoints, same request/response schema, same single-GPU,
one-request-at-a-time semantics. No batching and no cross-GPU concurrency are
introduced (explicitly out of scope).

The container contract is unchanged — same image name
(`flux1-dev-flux-inference:blackwell`), same port (`8630:8000`), same
`/v1/health/ready` healthcheck, same NIM-style `/v1/infer` schema — so the LiteLLM
custom handler (`~/src/LiteLLM/flux_handler.py`), the LiteLLM gateway, and the
Studio `app` service all keep working with **zero changes**.

## Goals

- Replace the hand-rolled FastAPI app with a LitServe `LitAPI` + `LitServer`.
- Preserve every endpoint and exact JSON shape currently served.
- Preserve all model capabilities: FLUX.1-dev base text-to-image, ControlNet
  (Union Pro 2.0, 7 control modes), IP-Adapter (InstantX + SigLIP + custom
  attention processors).
- Preserve single-flight, single-GPU semantics (no behavioral surprises).
- Keep the deploy/build flow identical (in-place rebuild + restart on 8630).

## Non-goals (explicitly out of scope)

- Adaptive batching of requests.
- Multi-GPU / multi-worker concurrency.
- True runtime cross-GPU hot-reload (see "Intentional behavioral difference").
- Any change to `flux_handler.py`, the LiteLLM `config.yaml`, the Studio `app`
  service, `docker-compose.yml`, or `ip_adapter_attention.py`.

## Current state (baseline)

`inference/server.py` is a FastAPI app (`uvicorn server:app`, port 8000 inside the
container) with module-global model state and a `threading.Lock` (`inference_lock`)
serializing inference. It loads, in `lifespan`:

- `FluxPipeline` (base, bf16) and `FluxControlNetPipeline` sharing one
  `FluxTransformer2DModel` and `AutoencoderKL`.
- A `FluxControlNetModel` (`Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0`).
- IP-Adapter: `IPAFluxAttnProcessor2_0` processors swapped into the transformer,
  an `MLPProjModel`, and a `SiglipVisionModel` + `AutoProcessor`
  (`google/siglip-so400m-patch14-384`).

Endpoints:

| Method | Path                 | Behavior |
|--------|----------------------|----------|
| GET    | `/`                  | Model/status/endpoint summary JSON |
| GET    | `/v1/health/ready`   | `{"status":"ready"|"loading"}` |
| GET    | `/v1/gpu/config`     | `{"devices":[...]}` (active devices) |
| POST   | `/v1/gpu/config`     | Validate + background hot-reload onto requested GPU(s) |
| POST   | `/v1/infer`          | Generate; returns `{"artifacts":[Artifact]}` |

`InferRequest` fields: `prompt`, `negative_prompt`, `control_image` (b64),
`control_mode` (canny/tile/depth/blur/pose/gray/low_quality),
`controlnet_conditioning_scale`, `ip_adapter_images` (list[b64]),
`adapter_strength`, `width`, `height`, `steps`, `guidance_scale`, `sampler`
(`euler`), `seed`.

`Artifact`: `base64`, `seed`, `finishReason` (`SUCCESS`/`ERROR`/`MODEL_NOT_READY`),
`errorReason`.

**Deployment fact that shapes the design:** the compose service pins the container
to a single physical GPU (`deploy.resources.reservations.devices.device_ids: ["2"]`,
`NVIDIA_GPU_DEVICE: 0`). Inside the container `torch.cuda.device_count() == 1`, so
the multi-GPU hot-reload path in `/v1/gpu/config` is already **inert in production**.

## Target design (Approach A)

### File changes

- **`inference/server.py`** — rewritten as a LitServe app (only substantive change).
- **`Dockerfile`** — add `litserve` to the `pip install` list; change `CMD` from
  `uvicorn server:app ...` to `python server.py` (LitServer starts itself).
- Everything else unchanged.

### `FluxLitAPI(litserve.LitAPI)`

Model state moves from module globals to instance attributes, set in `setup`.

- `setup(self, device)`: run the existing `_load_model` logic for the LitServe-provided
  `device` (a single `cuda:N`, `cuda:0` in the pinned container). Store
  `pipe_base`, `pipe_controlnet`, `siglip_model`, `siglip_processor`,
  `image_proj_model`, and `active_devices` on `self`. Retain the module-level
  helper functions (`_load_model`, `_setup_ip_adapter_on_transformer`,
  `_compute_image_emb`, `MLPProjModel`) largely verbatim.
- `decode_request(self, request)`: parse/validate the JSON body into `InferRequest`;
  return a plain dict of parameters. Pydantic validation errors are caught and
  surfaced as artifact-errors (see Error handling) rather than LitServe's default
  422 — to keep the wire contract identical.
- `predict(self, params)`: the current `/v1/infer` body verbatim — choose
  controlnet vs base pipeline, decode control/IP images, compute IP embeddings,
  set scheduler, run the pipeline, encode PNG→base64. Runs under the retained
  `inference_lock` so it stays single-flight even if LitServe queues calls.
- `encode_response(self, output)`: return `{"artifacts":[{...}]}` matching
  `InferResponse`.

### Server wiring

```python
api = FluxLitAPI(api_path="/v1/infer")
server = ls.LitServer(
    api,
    accelerator="cuda",
    devices=1,
    workers_per_device=1,   # single-flight per device
    # batching disabled (max_batch_size=1 / no batching)
)
# custom routes registered on server.app before run()
server.run(port=8000)
```

> Implementation note: the exact LitServe kwarg names/locations for `api_path`,
> batching disable, and accessing the underlying app (`server.app`) must be pinned
> to the installed LitServe version during the implementation plan (research step).
> The design intent — single worker, single device, batching off, custom
> `api_path`, extra routes — is version-independent.

### Custom routes (added to the LitServe FastAPI app)

- `GET /` → same model/status/endpoints summary.
- `GET /v1/health/ready` → `{"status":"ready"}` once the worker's model is loaded,
  else `{"status":"loading"}`. Must reflect worker readiness, not just process up.
- `GET /v1/gpu/config` → `{"devices": <active>}`.
- `POST /v1/gpu/config` → report + validate (see below).

## Intentional behavioral difference: `/v1/gpu/config`

In the current server the model lives in the same process as the routes, so a POST
can hot-swap it across GPUs. In LitServe the model lives in a separate worker
process, so an HTTP route cannot reach in and reload it. Because the container is
pinned to one GPU, the cross-GPU reload is already inert in production. Therefore:

- `GET` returns the active (single, pinned) device — unchanged.
- `POST` validates the requested device against visible GPUs:
  - requested == current → `{"status":"unchanged","devices":[...]}` (as today),
  - requested out of range → `400` (as today),
  - otherwise → `409`/clear message indicating live cross-GPU reload is not
    supported in this deployment (the only changed branch).

This is the **one documented deviation** from the old server. To restore true
multi-GPU reload later: un-pin the GPUs in compose and reintroduce an in-process
worker / worker-control mechanism (separate, deliberate change — out of scope here).

## Error handling (parity)

All existing error branches are preserved and returned as a 200 with an
`artifacts` entry whose `finishReason` is `ERROR`/`MODEL_NOT_READY` and whose
`errorReason` string matches the current text:

- model not ready / reloading,
- invalid `control_mode`,
- control image base64 decode failure,
- IP-adapter image base64 decode failure,
- `torch.cuda.OutOfMemoryError` (with empty_cache + resolution/steps hint),
- generic pipeline exception (`type(e).__name__: e`).

Request-schema validation failures in `decode_request` are mapped into the same
artifact-error shape so callers (the LiteLLM handler) observe identical responses.

## Build, deploy, rollback

In-place swap on moto in `~/src/FLUX.1-dev`:

```bash
docker compose build flux-inference
docker compose up -d flux-inference   # restarts flux1-dev-flux-inference:blackwell on 8630
```

Rollback: `git checkout -- inference/server.py Dockerfile` then rebuild + up.
(The previous image can also be retagged/kept before building, for instant revert.)

## Verification (must pass before declaring done)

1. `GET /v1/health/ready` → `ready` after model load completes.
2. Base text-to-image via `test_infer_request.py` → valid PNG artifact.
3. ControlNet via `test_controlnet.py` → valid image for a canny control input.
4. IP-Adapter via `test_ip_adapter.py` / `test_core_ip_adapter.py` → valid image.
5. Error parity spot-checks: invalid `control_mode` and malformed base64 return
   the same `finishReason: ERROR` shape.
6. End-to-end through LiteLLM: a `flux.1-dev` image-generation call returns HTTP
   200 **and** still logs `0.01` spend (the recently added cost registration is
   unaffected; `X-Litellm-Response-Cost: 0.01`).
7. `GET /v1/gpu/config` returns the pinned device; `POST` of the current device
   returns `unchanged`.

## Risks & mitigations

- **LitServe API specifics differ by version** → pin exact API in the
  implementation plan's research step; validate against the installed version.
- **Worker-readiness vs `/v1/health/ready`** → ensure the health route reports the
  worker's model-loaded state (LitServe load happens in the worker, asynchronously
  from process start); the compose healthcheck depends on this being accurate.
- **`decode_request` error mapping** → must intercept validation errors and emit
  artifact-errors, or the handler's contract breaks. Covered by error-parity tests.
- **Cold-start time** → model load is heavy; keep the compose healthcheck
  `start_period`/retries adequate (unchanged from today).
