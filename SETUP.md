# FLUX.1-dev Setup Summary

## Files Created

```
FLUX.1-dev/
├── Dockerfile                    # Inference service container
├── docker-compose.yml            # Multi-service orchestration
├── .env.example                  # Configuration template
├── .gitignore
├── README.md                     # Full documentation
├── start.sh                      # Startup script
├── stop.sh                       # Shutdown script
├── inference/
│   └── server.py                 # FLUX.1-dev FastAPI inference server
└── app/
    ├── Dockerfile                # Frontend app container
    ├── pyproject.toml            # Python dependencies
    ├── main.py                   # Uvicorn runner
    ├── app.py                    # FastAPI frontend application
    ├── core.py                   # FLUX engine backend
    ├── templates/
    │   └── index.html            # TailwindCSS web UI
    ├── utils/
    │   ├── config.py             # Environment loading
    │   ├── gpu_monitor.py        # NVIDIA GPU telemetry
    │   ├── image_utils.py        # PNG metadata handling
    │   ├── validators.py         # Input validation
    │   └── prompt_extender.py    # LLM prompt enhancement
    └── assets/generated/         # Image gallery storage
```

## Architecture

Based on the existing StableDiffusion setup with these key differences:

### Model Configuration
- **Model**: `black-forest-labs/FLUX.1-dev`
- **Precision**: bfloat16 (vs float16 for SD3.5)
- **Components**: Separate VAE + Transformer loading
- **Recommended Steps**: 20-30 (vs 30 for SD)
- **Guidance Scale**: 2.5-4.0 (vs 7.5 for SD)

### Ports
- **Inference API**: 8630 (vs 8620 for SD)
- **Frontend App**: 8631 (vs 8621 for SD)

### Traefik Labels
- **Path Prefix**: `/flux.1-dev` (vs `/sd-3-5-large`)
- **Router**: `flux-model` (vs `sd-model`)

## Usage

### 1. Configure Environment
```bash
cd /home/users/wrightda/src/FLUX.1-dev
cp .env.example .env
# Edit .env with your HF_TOKEN and GPU selection
```

### 2. Download Model (if not already cached)
```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'black-forest-labs/FLUX.1-dev',
    local_dir='/data/huggingface/black-forest-labs/FLUX.1-dev'
)
"
```

### 3. Start Services
```bash
./start.sh
```

### 4. Access
- **Web UI**: https://moto.hst.rdlabs.hpecorp.net/flux
- **API**: https://moto.hst.rdlabs.hpecorp.net/flux.1-dev/v1/infer
- **Health**: https://moto.hst.rdlabs.hpecorp.net/flux.1-dev/v1/health/ready

## API Example

```bash
curl -X POST http://localhost:8630/v1/infer \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cyberpunk city at night",
    "negative_prompt": "blurry, low quality",
    "width": 1024,
    "height": 1024,
    "steps": 28,
    "guidance_scale": 3.5,
    "seed": 42
  }'
```

## Next Steps

1. **Download model weights** to `/data/huggingface/black-forest-labs/FLUX.1-dev`
2. **Add HF_TOKEN** to `.env` file
3. **Run `./start.sh`** to launch services
4. **Test generation** via web UI or API

## Comparison with StableDiffusion Setup

| Feature | StableDiffusion | FLUX.1-dev |
|---------|----------------|------------|
| Model | SD3.5-Large | FLUX.1-dev |
| Precision | float16 | bfloat16 |
| Steps | 30 | 28 |
| Guidance | 7.5 | 3.5 |
| VRAM | ~20GB | ~24GB |
| Quality | High | Very High |
| Speed | Fast | Medium |
