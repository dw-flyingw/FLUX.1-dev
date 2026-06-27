# LitServe API Reference (Installed Version)

Captured via live introspection on `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` with
`pip install litserve fastapi uvicorn` (no pinned version — installs latest available).

## Raw Introspection Output

```
litserve 0.2.17
LitAPI.__init__ (self, max_batch_size: int = 1, batch_timeout: float = 0.0, api_path: str = '/predict', stream: bool = False, loop: Union[str, ForwardRef('LitLoop'), NoneType] = 'auto', spec: Optional[litserve.specs.base.LitSpec] = None, mcp: Optional[ForwardRef('MCP')] = None, enable_async: bool = False)
setup (self, device)
decode_request (self, request, **kwargs)
predict (self, x, **kwargs)
encode_response (self, output, **kwargs)
LitServer.__init__ (self, lit_api: Union[litserve.api.LitAPI, list[litserve.api.LitAPI]], accelerator: Literal['cpu', 'cuda', 'mps', 'auto'] = 'auto', devices: Union[int, Literal['auto']] = 'auto', workers_per_device: int = 1, timeout: Union[float, bool] = 30, healthcheck_path: str = '/health', info_path: str = '/info', shutdown_path: str = '/shutdown', enable_shutdown_api: bool = False, model_metadata: Optional[dict] = None, spec: Optional[litserve.specs.base.LitSpec] = None, max_payload_size=None, track_requests: bool = False, callbacks: Union[list[litserve.callbacks.base.Callback], litserve.callbacks.base.Callback, NoneType] = None, middlewares: Optional[list[Union[collections.abc.Callable, tuple[collections.abc.Callable, dict]]]] = None, loggers: Union[litserve.loggers.Logger, list[litserve.loggers.Logger], NoneType] = None, fast_queue: bool = False, disable_openapi_url: bool = False, max_batch_size: Optional[int] = None, batch_timeout: float = 0.0, stream: bool = False, api_path: Optional[str] = None, loop: Union[str, litserve.loops.base.LitLoop, NoneType] = None, restart_workers: bool = False)
has_app True FastAPI
routes ['/openapi.json', '/docs', '/docs/oauth2-redirect', '/redoc', '/', '/health', '/info', '/predict']
```

---

## Recorded Facts

- **litserve_version**: `0.2.17`
  - _Implication_: Pin `litserve==0.2.17` in `requirements.txt` / `Dockerfile` to avoid drift.

- **`api_path` location**: Present on **both** `LitAPI.__init__` (default `'/predict'`) and
  `LitServer.__init__` (default `None`). The `LitAPI` default wins unless overridden at the
  server level.
  - _Implication_: Set `api_path='/predict'` on the `LitAPI` subclass constructor (or leave it
    as the default). Do NOT set it on `LitServer` unless you need to override all APIs at once.
    Example: `class FluxAPI(ls.LitAPI): def __init__(self): super().__init__(api_path='/predict', max_batch_size=1)`

- **Batching — disable mechanism**: `max_batch_size` on `LitAPI.__init__` defaults to `1`
  (i.e., batching is effectively off by default — each request is processed alone).
  Setting it to any value > 1 enables batching.
  `LitServer.__init__` also accepts `max_batch_size: Optional[int] = None` as a server-level
  override.
  - _Implication_: For the FLUX port, batching must stay disabled (image generation is
    memory-bound). Keep `max_batch_size=1` (the default) on `LitAPI`. Do not set it on
    `LitServer`.

- **`workers_per_device`**: Confirmed on `LitServer.__init__` as `workers_per_device: int = 1`.
  - _Implication_: `LitServer(api, workers_per_device=1)` is the correct spelling. Default
    of 1 is correct for FLUX (one large model per GPU).

- **`accelerator`**: Confirmed on `LitServer.__init__` as
  `accelerator: Literal['cpu', 'cuda', 'mps', 'auto'] = 'auto'`.
  - _Implication_: Use `accelerator='cuda'` explicitly in the server constructor to avoid
    auto-detection surprises inside a container.

- **`devices`**: Confirmed on `LitServer.__init__` as `devices: Union[int, Literal['auto']] = 'auto'`.
  - _Implication_: Pass `devices=1` (or the count of GPUs exposed to the container) to be
    explicit. `'auto'` will work but is less deterministic.

- **Method signatures**:
  - `setup(self, device)` — matches assumption exactly.
  - `decode_request(self, request, **kwargs)` — has `**kwargs`; the base signature uses `request`
    as the positional name (LitServe introspects this name to determine request type annotation).
    **Always name the parameter `request`**, not `req` or `r`.
  - `predict(self, x, **kwargs)` — matches assumption (plus `**kwargs`).
  - `encode_response(self, output, **kwargs)` — matches assumption (plus `**kwargs`).
  - _Implication_: Override all four in `FluxLitAPI`. Do not drop `**kwargs` from the override
    signatures — LitServe may pass extra keyword arguments internally.

- **`decode_request` input type** (inferred): When no Pydantic `request_type` annotation is set
  on the `request` parameter, LitServe passes the JSON-parsed body as a plain Python `dict`.
  If the parameter is annotated with a Pydantic model, LitServe validates and deserialises
  automatically.
  - _Implication_: Either (a) annotate `request: GenerateRequest` with a Pydantic model (clean,
    recommended) or (b) annotate `request: Request` with Starlette's `Request` to receive the
    raw request object. Without annotation, expect a `dict`.

- **`server.app` (FastAPI instance)**: `LitServer` exposes `.app` as a live `FastAPI` instance
  (confirmed: `type(srv.app).__name__ == 'FastAPI'`).
  - _Implication_: Custom routes can be added **after** creating the server object and **before**
    calling `server.run()`:
    ```python
    server = ls.LitServer(api, ...)
    @server.app.get("/readyz")
    async def readyz(): return {"status": "ok"}
    server.run(...)
    ```
    Or: `server.app.add_api_route("/readyz", readyz, methods=["GET"])`.

- **Built-in routes**: `/openapi.json`, `/docs`, `/docs/oauth2-redirect`, `/redoc`, `/`,
  `/health`, `/info`, `/predict` (and `/shutdown` when `enable_shutdown_api=True`).
  - _Implication_: The readiness/health path is **`/health`** (configurable via
    `healthcheck_path` kwarg on `LitServer`). No need to add a custom health route; wire
    container/compose healthcheck to `GET /health`.

- **`healthcheck_path` / `shutdown_path` / `info_path`**: All configurable on `LitServer.__init__`.
  Defaults: `healthcheck_path='/health'`, `info_path='/info'`, `shutdown_path='/shutdown'`.
  - _Implication_: Keep defaults unless there's a naming conflict. The existing FastAPI server
    exposed `/health` — LitServe matches that automatically.
