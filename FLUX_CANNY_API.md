# FLUX.1-dev + Canny ControlNet — External API

Reference for calling the FLUX.1-dev inference service with the Canny ControlNet, from outside the host.

## Base URL

```
http://sprocket.hst.rdlabs.hpecorp.net/flux.1-dev
```

Routed through Traefik on the HPE corporate network. The `/flux.1-dev` prefix is stripped before reaching the service. Direct (host-local) URL is `http://sprocket.hst.rdlabs.hpecorp.net:8630`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/v1/health/ready` | Returns `{"status": "ready"}` once the model is loaded. |
| GET  | `/v1/gpu/config` | Returns currently active GPU device(s). |
| POST | `/v1/infer` | Generate an image (text-only or with a ControlNet image). |

## Models

- Base: `black-forest-labs/FLUX.1-dev`
- ControlNet: `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` (Union — supports `canny`, `tile`, `depth`, `blur`, `pose`, `gray`, `low_quality`)

## Canny request

`POST /v1/infer` with JSON body:

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | required | Positive prompt. |
| `negative_prompt` | string | `""` | Optional. |
| `control_image` | string (base64 PNG) | `""` | **Required for canny.** Must be a Canny edge map (white edges on black). Empty string = text-only inference, ControlNet ignored. |
| `control_mode` | string | `"canny"` | Use `"canny"`. |
| `controlnet_conditioning_scale` | float ∈ [0.0, 2.0] | `0.5` | 0.4–0.7 is the useful range; higher = more rigid adherence to edges. |
| `width` | int | `1024` | Multiple of 16. Must match `control_image` aspect; the service resizes the control image to `(width, height)`. |
| `height` | int | `1024` | Multiple of 16. |
| `steps` | int | `30` | 20–40 typical. |
| `guidance_scale` | float | `3.5` | FLUX-dev recommends ~3.5. |
| `sampler` | string | `"euler"` | Only `"euler"` supported. |
| `seed` | int | `0` | Reproducibility. |

### Preparing the Canny edge map

The ControlNet expects an edge image, **not** a regular photo. Generate it from a source image with OpenCV:

```python
import cv2, base64
img = cv2.imread("source.jpg")
edges = cv2.Canny(img, 100, 200)            # tune thresholds 50–250
edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
ok, buf = cv2.imencode(".png", edges_rgb)
control_b64 = base64.b64encode(buf.tobytes()).decode()
```

## Response

```json
{
  "artifacts": [
    {
      "base64": "<PNG bytes, base64-encoded>",
      "seed": 42,
      "finishReason": "SUCCESS",
      "errorReason": ""
    }
  ]
}
```

`finishReason` values: `"SUCCESS"`, `"MODEL_NOT_READY"` (model still loading or reloading), `"ERROR"` (see `errorReason`).

## Example — curl

```bash
CONTROL_B64=$(base64 -w0 canny_edges.png)
curl -X POST http://sprocket.hst.rdlabs.hpecorp.net/flux.1-dev/v1/infer \
  -H "Content-Type: application/json" \
  -d @- <<JSON | jq -r '.artifacts[0].base64' | base64 -d > out.png
{
  "prompt": "a futuristic building with glass windows, architectural rendering",
  "control_image": "$CONTROL_B64",
  "control_mode": "canny",
  "controlnet_conditioning_scale": 0.6,
  "width": 1024,
  "height": 1024,
  "steps": 30,
  "guidance_scale": 3.5,
  "seed": 42
}
JSON
```

## Example — Python

```python
import base64, io, cv2, requests
from PIL import Image

BASE = "http://sprocket.hst.rdlabs.hpecorp.net/flux.1-dev"

img = cv2.imread("source.jpg")
edges = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2RGB)
_, png = cv2.imencode(".png", edges)
control_b64 = base64.b64encode(png.tobytes()).decode()

payload = {
    "prompt": "a futuristic building with glass windows, architectural rendering",
    "control_image": control_b64,
    "control_mode": "canny",
    "controlnet_conditioning_scale": 0.6,
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "guidance_scale": 3.5,
    "seed": 42,
}

r = requests.post(f"{BASE}/v1/infer", json=payload, timeout=300)
artifact = r.json()["artifacts"][0]
assert artifact["finishReason"] == "SUCCESS", artifact["errorReason"]
Image.open(io.BytesIO(base64.b64decode(artifact["base64"]))).save("out.png")
```

## Operational notes

- Single-GPU service; requests serialize through an internal lock. Expect ~10–30 s per 1024×1024 image at 30 steps on H100-class hardware.
- Health-check `/v1/health/ready` before posting; on a cold start the model takes several seconds to load.
- A 502 from Traefik usually means the inference container is restarting — check `docker compose logs flux-inference` on the host.
