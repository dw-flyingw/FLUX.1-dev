  Inference API: http://localhost:8630

  Endpoints:
  - POST /v1/infer -- generate images
  - GET /v1/health/ready -- health check
  - GET /v1/gpu/config -- GPU status

  Models loaded:
  - Base: black-forest-labs/FLUX.1-dev
  - ControlNet: Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0

  Web UI: http://localhost:8631/flux

  Via Traefik: http://sprocket.hst.rdlabs.hpecorp.net/flux.1-dev/v1/infer
