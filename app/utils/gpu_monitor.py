"""GPU monitoring utilities."""

import subprocess
from dataclasses import dataclass
from typing import List


@dataclass
class GPUInfo:
    """GPU telemetry data."""

    index: int
    name: str
    utilization: int
    memory_used: int
    memory_total: int
    memory_percent: float
    temperature: int


def get_gpu_info() -> List[GPUInfo]:
    """Get GPU information using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                index = int(parts[0])
                name = parts[1]
                utilization = int(parts[2])
                memory_used = int(parts[3])
                memory_total = int(parts[4])
                memory_percent = round((memory_used / memory_total) * 100, 1) if memory_total > 0 else 0.0
                temperature = int(parts[5])
                gpus.append(
                    GPUInfo(
                        index=index,
                        name=name,
                        utilization=utilization,
                        memory_used=memory_used,
                        memory_total=memory_total,
                        memory_percent=memory_percent,
                        temperature=temperature,
                    )
                )
        return gpus
    except Exception:
        return []
