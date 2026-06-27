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
