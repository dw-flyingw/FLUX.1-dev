# IP-Adapter Integration Design for FLUX.1-dev

**Date:** 2026-05-20  
**Status:** Design Approved  
**Goal:** Add IP-Adapter support to enable style/content guidance using reference images, combined with existing ControlNet for structural control.

## 1. Overview

The FLUX.1-dev service currently supports ControlNet for structural image conditioning (edges, depth, etc.). This design adds **IP-Adapter support** to enable style and content guidance using reference images, allowing simultaneous use of both ControlNet and IP-Adapter in a single generation request.

**Primary use case:** Generate images of HPE rack-mount products (DL380a, B10000, etc.) placed within HPE Rack scenes using product reference images for style/content guidance.

## 2. Architecture

### Current State
- **ControlNet:** Structural guidance (canny, tile, depth, blur, pose, gray, low_quality modes)
- **NIM Backend:** NVIDIA Inference Microservice handles image generation
- **API:** `/flux/api/generate` accepts control image + mode + conditioning scale

### Proposed State
Both ControlNet and IP-Adapter can be used simultaneously:

```
FastAPI (/flux/api/generate)
    ↓
FluxEngine.generate_image()
    ↓
NIM Backend (/v1/infer)
    - ControlNet path (if control_image provided)
    - IP-Adapter path (if ip_adapter_images provided)
    ↓
Generated Image
```

**Key principle:** Both conditioning types are optional and independent. Requests can use:
- ControlNet only
- IP-Adapter only  
- Both together
- Neither (text-only)

## 3. API Changes

### New Parameters

**FastAPI endpoint:** `/flux/api/generate` (POST)

```python
# Existing ControlNet parameters (unchanged)
control_image: UploadFile | None = File(None)
control_mode: str = Form("canny")
controlnet_conditioning_scale: float = Form(0.5)

# New IP-Adapter parameters
ip_adapter_images: list[UploadFile] | None = File(None)  # Multiple reference images
adapter_strength: float = Form(0.8)  # Range: 0.0-2.0
```

**NIM backend payload structure:**

```json
{
  "prompt": "HPE DL380a servers in an HPE rack",
  "negative_prompt": "",
  "width": 1024,
  "height": 1024,
  "steps": 28,
  "guidance_scale": 3.5,
  "seed": 42,
  
  "control_image": "base64...",
  "control_mode": "canny",
  "controlnet_conditioning_scale": 0.5,
  
  "ip_adapter_images": ["base64...", "base64..."],
  "adapter_strength": 0.8
}
```

### Processing Flow

1. **FastAPI:** Receive `ip_adapter_images` (file uploads) and `adapter_strength` (float)
2. **Validation:** Check each image size/format; validate `adapter_strength` in range [0.0, 2.0]
3. **Encoding:** Convert each image file to base64 → `ip_adapter_images_b64` list
4. **NIM Request:** Pass both ControlNet and IP-Adapter params in single payload to `/v1/infer`
5. **Response:** Include metadata about both conditioning methods used

### Response Format

```json
{
  "result": {
    "image_b64": "...",
    "prompt": "HPE DL380a servers in an HPE rack",
    "negative_prompt": "",
    "seed_used": 42,
    "generation_time": 45.2,
    "timestamp": "2026-05-20T14:30:00",
    "metadata": {
      "steps": 28,
      "guidance_scale": 3.5,
      "sampler": "euler",
      "aspect_ratio": "1:1",
      "size": "1024x1024"
    },
    "control_mode": "canny",
    "controlnet_conditioning_scale": 0.5,
    "ip_adapter_used": true,
    "adapter_strength": 0.8
  }
}
```

## 4. Backend Implementation

### NIM Backend Changes

**Objective:** Extend NIM inference pipeline to load and apply IP-Adapter alongside ControlNet.

**Changes needed:**

1. **Weight Loading:** Load IP-Adapter weights during model initialization
   - Source: Open-source (InstantX, XLabs-AI) or local checkpoint
   - Timing: Load once at startup, inject into FLUX model layers

2. **Inference Pipeline:** Modify `/v1/infer` to handle IP-Adapter conditioning
   - Encode reference images using IP-Adapter image encoder
   - Inject conditioning at appropriate diffusion steps
   - Apply both ControlNet + IP-Adapter in same forward pass

3. **Validation:** Reject invalid IP-Adapter payloads
   - Empty `ip_adapter_images` list → ignore IP-Adapter path
   - `adapter_strength = 0.0` → no-op (log warning)
   - Invalid image format/size → error response

### Fallback Plan

**If NIM backend cannot be extended with IP-Adapter:** Use open-source Hugging Face implementation as separate local inference path:
- Use [InstantX/FLUX.1-dev-IP-Adapter](https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter) or [XLabs-AI/flux-ip-adapter](https://huggingface.co/XLabs-AI/flux-ip-adapter)
- Run locally in separate inference worker
- Coordinate with NIM backend for consistent results

**Key assumption to verify before implementation:** NIM backend infrastructure supports loading additional model weights/adapters.

## 5. Core Module Updates

**File:** `app/core.py`

### Changes to `FluxEngine.generate_image()`

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
    # ControlNet parameters (existing)
    control_image_b64: str = "",
    control_mode: str = "canny",
    controlnet_conditioning_scale: float = 0.5,
    # IP-Adapter parameters (NEW)
    ip_adapter_images_b64: list[str] = [],
    adapter_strength: float = 0.8,
) -> GenerationResult:
```

### Updates to request payload construction:

```python
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
```

### Updates to `GenerationResult` dataclass:

Add fields to track IP-Adapter usage:
```python
ip_adapter_used: bool = False
adapter_strength: float = 0.0
```

## 6. Error Handling & Validation

### Input Validation

| Parameter | Validation Rule |
|-----------|-----------------|
| `ip_adapter_images` | Each image: valid format (PNG/JPG), size ≥ 64x64 |
| `adapter_strength` | Range [0.0, 2.0]; warn if 0.0 (no-op) |
| Both ControlNet + IP-Adapter | Both can be present; no mutual exclusion |

### Error Scenarios

1. **Invalid IP-Adapter image format**
   ```
   HTTP 422: "Invalid IP-Adapter image: must be PNG or JPEG"
   ```

2. **adapter_strength out of range**
   ```
   HTTP 422: "adapter_strength must be between 0.0 and 2.0"
   ```

3. **NIM backend error during IP-Adapter processing**
   ```
   HTTP 500: "IP-Adapter conditioning failed: {error_reason}"
   ```

4. **Empty ip_adapter_images list**
   - Silently ignore IP-Adapter path (no error)
   - Set `ip_adapter_used = false` in response metadata

## 7. Testing Strategy

### Test Suite: `test_ip_adapter.py`

**Test cases:**

1. **IP-Adapter only** (no ControlNet)
   - Single reference image
   - Multiple reference images
   - Different `adapter_strength` values

2. **ControlNet only** (no IP-Adapter)
   - Regression test: existing functionality unchanged

3. **Both together**
   - ControlNet + single IP-Adapter image
   - ControlNet + multiple IP-Adapter images
   - Verify both conditions applied correctly

4. **Edge cases**
   - `adapter_strength = 0.0` (no-op)
   - Empty `ip_adapter_images` list
   - Invalid image formats
   - Oversized images
   - Very small images (< 64x64)

5. **Integration test**
   - Generate HPE product in rack scene using:
     - ControlNet for structural guidance (canny edge map of rack)
     - IP-Adapter for product style (reference image of DL380a)

### Test Structure

Mirror existing `test_controlnet.py` pattern:
- Health check
- Base inference (text-only)
- IP-Adapter inference
- Combined ControlNet + IP-Adapter inference
- Parameter validation tests

## 8. Files Modified

| File | Changes |
|------|---------|
| `app/app.py` | Add IP-Adapter form parameters to `/flux/api/generate` endpoint |
| `app/core.py` | Update `FluxEngine.generate_image()` signature and implementation; update `GenerationResult` dataclass |
| `test_ip_adapter.py` | New test suite (mirror `test_controlnet.py` structure) |
| **NIM Backend** | Load IP-Adapter weights; modify `/v1/infer` inference pipeline |

## 9. Backward Compatibility

- **Existing ControlNet requests:** No breaking changes. New parameters are optional with sensible defaults.
- **Existing API clients:** Will continue to work without modification.
- **Default behavior:** If neither ControlNet nor IP-Adapter provided, generation is text-only (existing behavior).

## 10. Key Assumptions & Risks

| Item | Status | Mitigation |
|------|--------|-----------|
| NIM backend can load IP-Adapter weights | **TO VERIFY** | Investigate NVIDIA NIM architecture; fallback to Hugging Face implementation |
| IP-Adapter and ControlNet compatible in same forward pass | **ASSUMED** | Validate during NIM backend implementation |
| Shared `adapter_strength` sufficient for multiple images | **ASSUMED** | Per-image strength can be added later if needed |
| Base64 image encoding scales to multiple images | **ASSUMED** | Should be fine; existing ControlNet already uses base64 |

## 11. Success Criteria

- ✅ Generate images with IP-Adapter reference images
- ✅ Generate images with ControlNet + IP-Adapter combined
- ✅ All existing ControlNet tests pass (no regression)
- ✅ New IP-Adapter tests pass (including edge cases)
- ✅ Metadata correctly reflects which conditioning methods were used
- ✅ Error handling for invalid images/parameters
