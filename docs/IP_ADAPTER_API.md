# FLUX.1-dev Inference API — IP-Adapter Usage

**Endpoint:** `POST /v1/infer`

**Base URLs:**
- External (via Traefik): `http://sprocket.hst.rdlabs.hpecorp.net/flux.1-dev` — strips `/flux.1-dev` prefix, e.g. `POST http://sprocket.hst.rdlabs.hpecorp.net/flux.1-dev/v1/infer`
- Direct on host: `http://<host>:8630` (host port `8630` → container port `8000`)

## Request body (`InferRequest`)

```json
{
  "prompt": "<required text prompt>",
  "negative_prompt": "",
  "ip_adapter_images": ["<base64 PNG/JPG>", "..."],
  "adapter_strength": 0.8,
  "control_image": "",
  "control_mode": "canny",
  "controlnet_conditioning_scale": 0.5,
  "width": 1024,
  "height": 1024,
  "steps": 30,
  "guidance_scale": 3.5,
  "sampler": "euler",
  "seed": 0
}
```

## IP-Adapter–specific fields

- **`ip_adapter_images`** — list of base64-encoded reference images (no `data:` prefix, just raw base64). IP-Adapter is engaged whenever this list is non-empty; an empty list disables it.
- **`adapter_strength`** — float in `[0.0, 2.0]`, default `0.8`. Controls how strongly the reference image(s) influence the output. Set per-request — the value is written into every `IPAFluxAttnProcessor2_0` on the transformer inside the inference lock.
- Multiple images are allowed; they are encoded by SigLIP (`google/siglip-so400m-patch14-384`) and projected to 128 tokens of cross-attention conditioning.

## Composability

- IP-Adapter and ControlNet can be combined: provide both `ip_adapter_images` *and* a `control_image` + `control_mode`. The server selects `FluxControlNetPipeline` when `control_image` is present, otherwise the base `FluxPipeline`. The IP-Adapter attention processors live on the shared transformer, so they apply to both pipelines.
- IP-Adapter alone (no ControlNet): just leave `control_image` empty.

## Reference image constraints

- Must be decodable by PIL; converted to RGB internally.
- Avoid extreme aspect ratios — there is a known bounds error with very small / unusual references (the test suite uses 128×128 reference images to dodge this). Square images at reasonable sizes (e.g., 256×256+) are safe.

## Response (`InferResponse`)

```json
{ "artifacts": [ { "base64": "<png>", "seed": 0, "finishReason": "SUCCESS", "errorReason": "" } ] }
```

On failure, `finishReason` is `ERROR` or `MODEL_NOT_READY` and `base64` is empty — check this before decoding.

## Other notes

- All requests serialize on a global `inference_lock` (one inference at a time).
- Check `GET /v1/health/ready` first; during a GPU reload (`POST /v1/gpu/config`) inference returns `MODEL_NOT_READY`.
- See `test_infer_request.py` for working example calls.
