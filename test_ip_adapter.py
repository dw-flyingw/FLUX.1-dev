#!/usr/bin/env python3
"""Integration tests for FLUX.1-dev IP-Adapter support.

Requires both servers running:
  - NIM backend:  cd inference && python server.py   (port 8630)
  - FastAPI app:  cd app && python main.py           (port 8080)
"""
import base64
import io
import os
import sys
import time

import requests
from PIL import Image, ImageDraw

BASE_URL = "http://localhost:8630"
TIMEOUT = 300
OUTPUT_DIR = "test_outputs"


def make_reference_image(width: int = 512, height: int = 512, color: str = "blue") -> str:
    img = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, width - 50, height - 50], outline="white", width=5)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_control_image(width: int = 512, height: int = 512) -> str:
    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 400, 400], outline="white", width=3)
    draw.line([100, 100, 400, 400], fill="white", width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_health():
    print("1. Health check...", end=" ", flush=True)
    start = time.time()
    resp = requests.get(f"{BASE_URL}/v1/health/ready", timeout=10)
    elapsed = time.time() - start
    data = resp.json()
    if data.get("status") == "ready":
        print(f"PASS - {elapsed:.1f}s")
        return True
    print(f"FAIL - {data}")
    return False


def test_ip_adapter_in_root():
    print("2. Root reports ip_adapter...", end=" ", flush=True)
    start = time.time()
    resp = requests.get(f"{BASE_URL}/", timeout=10)
    elapsed = time.time() - start
    data = resp.json()
    if "ip_adapter" in data and data.get("status") == "ready":
        print(f"PASS - {data['ip_adapter']}, {elapsed:.1f}s")
        return True
    print(f"FAIL - {data}")
    return False


def test_ip_adapter_only_single_image():
    print("3. IP-Adapter only (single image, 4 steps)...", end=" ", flush=True)
    payload = {
        "prompt": "HPE DL380a server in an HPE rack, product photography",
        "ip_adapter_images": [make_reference_image(color="silver")],
        "adapter_strength": 0.8,
        "width": 512,
        "height": 512,
        "steps": 4,
        "guidance_scale": 3.5,
        "seed": 42,
    }
    start = time.time()
    resp = requests.post(f"{BASE_URL}/v1/infer", json=payload, timeout=TIMEOUT)
    elapsed = time.time() - start
    artifact = resp.json().get("artifacts", [{}])[0]
    if artifact.get("finishReason") == "SUCCESS" and artifact.get("base64"):
        img = Image.open(io.BytesIO(base64.b64decode(artifact["base64"])))
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        img.save(f"{OUTPUT_DIR}/ip_adapter_single.png")
        print(f"PASS - {img.size[0]}x{img.size[1]}, {elapsed:.1f}s")
        return True
    print(f"FAIL - {artifact.get('finishReason')}: {artifact.get('errorReason')}")
    return False


def test_ip_adapter_only_multiple_images():
    print("4. IP-Adapter only (multiple images, 4 steps)...", end=" ", flush=True)
    payload = {
        "prompt": "HPE rack with multiple servers, datacenter",
        "ip_adapter_images": [
            make_reference_image(color="silver"),
            make_reference_image(color="darkgray"),
        ],
        "adapter_strength": 0.7,
        "width": 512,
        "height": 512,
        "steps": 4,
        "guidance_scale": 3.5,
        "seed": 42,
    }
    start = time.time()
    resp = requests.post(f"{BASE_URL}/v1/infer", json=payload, timeout=TIMEOUT)
    elapsed = time.time() - start
    artifact = resp.json().get("artifacts", [{}])[0]
    if artifact.get("finishReason") == "SUCCESS" and artifact.get("base64"):
        img = Image.open(io.BytesIO(base64.b64decode(artifact["base64"])))
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        img.save(f"{OUTPUT_DIR}/ip_adapter_multi.png")
        print(f"PASS - {img.size[0]}x{img.size[1]}, {elapsed:.1f}s")
        return True
    print(f"FAIL - {artifact.get('finishReason')}: {artifact.get('errorReason')}")
    return False


def test_controlnet_only_regression():
    print("5. ControlNet only (regression)...", end=" ", flush=True)
    payload = {
        "prompt": "a server rack in a clean datacenter",
        "control_image": make_control_image(),
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
    artifact = resp.json().get("artifacts", [{}])[0]
    if artifact.get("finishReason") == "SUCCESS" and artifact.get("base64"):
        print(f"PASS - {elapsed:.1f}s")
        return True
    print(f"FAIL - {artifact.get('finishReason')}: {artifact.get('errorReason')}")
    return False


def test_controlnet_and_ip_adapter_combined():
    print("6. ControlNet + IP-Adapter combined...", end=" ", flush=True)
    payload = {
        "prompt": "HPE DL380a mounted in HPE rack, professional datacenter photography",
        "control_image": make_control_image(),
        "control_mode": "canny",
        "controlnet_conditioning_scale": 0.5,
        "ip_adapter_images": [make_reference_image(color="silver")],
        "adapter_strength": 0.8,
        "width": 512,
        "height": 512,
        "steps": 4,
        "guidance_scale": 3.5,
        "seed": 42,
    }
    start = time.time()
    resp = requests.post(f"{BASE_URL}/v1/infer", json=payload, timeout=TIMEOUT)
    elapsed = time.time() - start
    artifact = resp.json().get("artifacts", [{}])[0]
    if artifact.get("finishReason") == "SUCCESS" and artifact.get("base64"):
        img = Image.open(io.BytesIO(base64.b64decode(artifact["base64"])))
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        img.save(f"{OUTPUT_DIR}/combined.png")
        print(f"PASS - {img.size[0]}x{img.size[1]}, {elapsed:.1f}s")
        return True
    print(f"FAIL - {artifact.get('finishReason')}: {artifact.get('errorReason')}")
    return False


def test_invalid_adapter_strength_rejected():
    print("7. Invalid adapter_strength (5.0) rejected...", end=" ", flush=True)
    payload = {
        "prompt": "test",
        "ip_adapter_images": [make_reference_image(128, 128)],
        "adapter_strength": 5.0,
        "width": 512,
        "height": 512,
        "steps": 1,
        "seed": 1,
    }
    start = time.time()
    resp = requests.post(f"{BASE_URL}/v1/infer", json=payload, timeout=30)
    elapsed = time.time() - start
    # The LitServe port maps request-schema validation failures to a 200
    # artifact-error (finishReason=ERROR), consistent with every other error
    # branch, rather than the original FastAPI server's HTTP 422.
    if resp.status_code == 200:
        artifact = resp.json()["artifacts"][0]
        if artifact.get("finishReason") == "ERROR":
            print(f"PASS - rejected as artifact-error, {elapsed:.1f}s")
            return True
    print(f"FAIL - expected 200 artifact-error, got {resp.status_code}: {resp.text[:200]}")
    return False


def test_empty_ip_adapter_falls_back_to_text_only():
    print("8. Empty ip_adapter_images → text-only fallback...", end=" ", flush=True)
    payload = {
        "prompt": "a server rack, simple render",
        "ip_adapter_images": [],
        "adapter_strength": 0.8,
        "width": 512,
        "height": 512,
        "steps": 4,
        "seed": 42,
    }
    start = time.time()
    resp = requests.post(f"{BASE_URL}/v1/infer", json=payload, timeout=TIMEOUT)
    elapsed = time.time() - start
    artifact = resp.json().get("artifacts", [{}])[0]
    if artifact.get("finishReason") == "SUCCESS" and artifact.get("base64"):
        print(f"PASS - text-only, {elapsed:.1f}s")
        return True
    print(f"FAIL - {artifact.get('finishReason')}: {artifact.get('errorReason')}")
    return False


def main():
    print(f"Testing FLUX.1-dev IP-Adapter at {BASE_URL}\n")
    results = [
        ("Health", test_health()),
        ("IP-Adapter in root", test_ip_adapter_in_root()),
        ("IP-Adapter single image", test_ip_adapter_only_single_image()),
        ("IP-Adapter multiple images", test_ip_adapter_only_multiple_images()),
        ("ControlNet regression", test_controlnet_only_regression()),
        ("ControlNet + IP-Adapter", test_controlnet_and_ip_adapter_combined()),
        ("Invalid adapter_strength", test_invalid_adapter_strength_rejected()),
        ("Empty images fallback", test_empty_ip_adapter_falls_back_to_text_only()),
    ]
    print(f"\n{'='*40}")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    for name, result in results:
        print(f"  {'PASS' if result else 'FAIL'} - {name}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
