#!/usr/bin/env python3
"""Test script to verify FLUX.1-dev inference with and without ControlNet."""

import base64
import io
import json
import sys
import time

import requests
from PIL import Image, ImageDraw

BASE_URL = "http://localhost:8630"
TIMEOUT = 300


def make_test_image(width=512, height=512) -> str:
    """Generate a simple test control image (white lines on black) and return as base64."""
    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)
    # Draw some edges to simulate a canny-style control image
    draw.rectangle([100, 100, 400, 400], outline="white", width=3)
    draw.line([100, 100, 400, 400], fill="white", width=3)
    draw.line([400, 100, 100, 400], fill="white", width=3)
    draw.ellipse([150, 150, 350, 350], outline="white", width=3)
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


def test_root_info():
    print("2. Root endpoint (model info)...", end=" ", flush=True)
    resp = requests.get(f"{BASE_URL}/", timeout=10)
    data = resp.json()
    has_controlnet = "controlnet" in data or "control_modes" in data
    if has_controlnet and data.get("status") == "ready":
        print(f"PASS - controlnet: {data.get('controlnet', 'N/A')}")
        return True
    print(f"FAIL - {data}")
    return False


def test_base_inference():
    print("3. Base inference (text-only, 4 steps)...", end=" ", flush=True)
    payload = {
        "prompt": "a red cube on a white table, simple 3d render",
        "width": 512,
        "height": 512,
        "steps": 4,
        "guidance_scale": 3.5,
        "seed": 42,
    }
    start = time.time()
    resp = requests.post(f"{BASE_URL}/v1/infer", json=payload, timeout=TIMEOUT)
    elapsed = time.time() - start
    data = resp.json()

    artifact = data.get("artifacts", [{}])[0]
    if artifact.get("finishReason") == "SUCCESS" and artifact.get("base64"):
        img_bytes = base64.b64decode(artifact["base64"])
        img = Image.open(io.BytesIO(img_bytes))
        print(f"PASS - {img.size[0]}x{img.size[1]}, {elapsed:.1f}s")
        img.save("test_output_base.png")
        print("   Saved: test_output_base.png")
        return True
    print(f"FAIL - {artifact.get('finishReason')}: {artifact.get('errorReason')}")
    return False


def test_controlnet_inference():
    print("4. ControlNet inference (canny, 4 steps)...", end=" ", flush=True)
    control_b64 = make_test_image()
    payload = {
        "prompt": "a futuristic building with glass windows, architectural rendering",
        "control_image": control_b64,
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
    data = resp.json()

    artifact = data.get("artifacts", [{}])[0]
    if artifact.get("finishReason") == "SUCCESS" and artifact.get("base64"):
        img_bytes = base64.b64decode(artifact["base64"])
        img = Image.open(io.BytesIO(img_bytes))
        print(f"PASS - {img.size[0]}x{img.size[1]}, {elapsed:.1f}s")
        img.save("test_output_controlnet.png")
        print("   Saved: test_output_controlnet.png")
        return True
    print(f"FAIL - {artifact.get('finishReason')}: {artifact.get('errorReason')}")
    return False


def test_controlnet_modes():
    print("5. ControlNet mode validation...", end=" ", flush=True)
    payload = {
        "prompt": "test",
        "control_image": make_test_image(64, 64),
        "control_mode": "invalid_mode",
        "width": 512,
        "height": 512,
        "steps": 1,
        "seed": 1,
    }
    resp = requests.post(f"{BASE_URL}/v1/infer", json=payload, timeout=30)
    data = resp.json()
    artifact = data.get("artifacts", [{}])[0]
    if artifact.get("finishReason") == "ERROR" and "invalid" in artifact.get("errorReason", "").lower():
        print("PASS - invalid mode rejected correctly")
        return True
    print(f"FAIL - expected error for invalid mode, got: {artifact.get('finishReason')}")
    return False


def main():
    print(f"Testing FLUX.1-dev + ControlNet at {BASE_URL}\n")

    results = []
    results.append(("Health", test_health()))
    results.append(("Root info", test_root_info()))
    results.append(("Base inference", test_base_inference()))
    results.append(("ControlNet inference", test_controlnet_inference()))
    results.append(("Mode validation", test_controlnet_modes()))

    print(f"\n{'='*40}")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    for name, result in results:
        print(f"  {'PASS' if result else 'FAIL'} - {name}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
