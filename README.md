# FLUX.1-dev Studio

Professional AI image generation interface powered by FLUX.1-dev model.

## Architecture

- **Inference Service**: FastAPI server running FLUX.1-dev via HuggingFace Diffusers
- **Frontend App**: FastAPI + TailwindCSS web interface
- **Traefik Integration**: Reverse proxy for external access

## Quick Start

### Prerequisites

1. Docker with NVIDIA Container Toolkit
2. HuggingFace token with access to FLUX.1-dev
3. Pre-downloaded model weights at `/data/huggingface/black-forest-labs/FLUX.1-dev`

### Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# - HF_TOKEN: Your HuggingFace token
# - NVIDIA_GPU_DEVICE: GPU to use (0-3)

# Start services
./start.sh

# Stop services
./stop.sh
```

### Access

- **Web Interface**: http://localhost:8631/flux
- **Inference API**: http://localhost:8630
- **Health Check**: http://localhost:8630/v1/health/ready

## API Endpoints

### Generate Image

```bash
POST /v1/infer
Content-Type: application/json

{
  "prompt": "a beautiful sunset over mountains",
  "negative_prompt": "blurry, low quality",
  "width": 1024,
  "height": 1024,
  "steps": 28,
  "guidance_scale": 3.5,
  "sampler": "euler",
  "seed": 42
}
```

### Health Check

```bash
GET /v1/health/ready
```

### GPU Configuration

```bash
GET /v1/gpu/config
POST /v1/gpu/config
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `FRONTEND_PORT` | Web interface port | 8631 |
| `FLUX_INFERENCE_PORT` | Inference API port | 8630 |
| `HF_TOKEN` | HuggingFace authentication token | (required) |
| `NVIDIA_GPU_DEVICE` | GPU device selection | 0 |
| `HF_HOME` | HuggingFace cache directory | /opt/huggingface |

## Model Details

**FLUX.1-dev** is a high-quality text-to-image model from Black Forest Labs.

- **Base Resolution**: 1024x1024
- **Recommended Steps**: 20-30
- **Guidance Scale**: 2.5-4.0
- **Dtype**: bfloat16
- **VRAM Requirement**: ~24GB (single GPU)

## Troubleshooting

### Model Not Loading

Ensure the model weights are pre-downloaded:
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('black-forest-labs/FLUX.1-dev', local_dir='/data/huggingface/black-forest-labs/FLUX.1-dev')"
```

### GPU Out of Memory

- Reduce image resolution
- Decrease number of steps
- Use single GPU instead of multi-GPU

### Connection Refused

Verify the inference service is running:
```bash
docker ps | grep flux-inference
curl http://localhost:8630/v1/health/ready
```

## License

FLUX.1-dev model is subject to Black Forest Labs license terms.
