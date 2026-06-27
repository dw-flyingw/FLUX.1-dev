# Task 1 Report — LitServe API Introspection

**Date**: 2026-06-27
**Branch**: flux-litserve-port
**Base image**: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

## Method

Ran a throwaway `docker run --rm` container on moto (no GPUs, no network changes to the live
flux-inference service) and executed a Python introspection script via base64-encoded stdin to
avoid shell quoting issues.

## Raw Output (run 1 — signature dump, partial — errored on route check)

```
litserve 0.2.17
LitAPI.__init__ (self, max_batch_size: int = 1, batch_timeout: float = 0.0, api_path: str = '/predict', stream: bool = False, loop: Union[str, ForwardRef('LitLoop'), NoneType] = 'auto', spec: Optional[litserve.specs.base.LitSpec] = None, mcp: Optional[ForwardRef('MCP')] = None, enable_async: bool = False)
setup (self, device)
decode_request (self, request, **kwargs)
predict (self, x, **kwargs)
encode_response (self, output, **kwargs)
LitServer.__init__ (self, lit_api: Union[litserve.api.LitAPI, list[litserve.api.LitAPI]], accelerator: Literal['cpu', 'cuda', 'mps', 'auto'] = 'auto', devices: Union[int, Literal['auto']] = 'auto', workers_per_device: int = 1, timeout: Union[float, bool] = 30, healthcheck_path: str = '/health', info_path: str = '/info', shutdown_path: str = '/shutdown', enable_shutdown_api: bool = False, model_metadata: Optional[dict] = None, spec: Optional[litserve.specs.base.LitSpec] = None, max_payload_size=None, track_requests: bool = False, callbacks: Union[list[litserve.callbacks.base.Callback], litserve.callbacks.base.Callback, NoneType] = None, middlewares: Optional[list[Union[collections.abc.Callable, tuple[collections.abc.Callable, dict]]]] = None, loggers: Union[litserve.loggers.Logger, list[litserve.loggers.Logger], NoneType] = None, fast_queue: bool = False, disable_openapi_url: bool = False, max_batch_size: Optional[int] = None, batch_timeout: float = 0.0, stream: bool = False, api_path: Optional[str] = None, loop: Union[str, litserve.loops.base.LitLoop, NoneType] = None, restart_workers: bool = False)
ERROR: KeyError 'request' (decode_request lambda used 'r' not 'request'; LitServe introspects param name)
```

## Raw Output (run 2 — route check with corrected lambda)

```
litserve 0.2.17
has_app True FastAPI
routes ['/openapi.json', '/docs', '/docs/oauth2-redirect', '/redoc', '/', '/health', '/info', '/predict']
```

## Recorded Facts Summary

| Fact | Value |
|------|-------|
| litserve_version | 0.2.17 |
| api_path kwarg location | LitAPI.__init__ (default '/predict') AND LitServer.__init__ (default None) |
| batching disabled when | max_batch_size=1 on LitAPI (default — no action needed) |
| workers_per_device | confirmed on LitServer.__init__, default 1 |
| accelerator | confirmed on LitServer.__init__, Literal['cpu','cuda','mps','auto'], default 'auto' |
| devices | confirmed on LitServer.__init__, Union[int, Literal['auto']], default 'auto' |
| setup signature | (self, device) — exact match |
| decode_request signature | (self, request, **kwargs) — **kwargs added; param must be named 'request' |
| predict signature | (self, x, **kwargs) — **kwargs added |
| encode_response signature | (self, output, **kwargs) — **kwargs added |
| server.app type | FastAPI |
| built-in routes | /openapi.json /docs /docs/oauth2-redirect /redoc / /health /info /predict |
| health path | /health (configurable via healthcheck_path kwarg) |
| decode_request input (no annotation) | JSON-parsed dict (assumption; annotate with Pydantic model for typed input) |

## Notes / Gotchas

- LitServe introspects the `decode_request` parameter name: it **must** be called `request`,
  not `req` or `r`. Using another name causes a `KeyError` at `LitServer.__init__` time.
- `**kwargs` must be kept in override signatures (LitServe may inject keyword arguments).
- The `api_path` kwarg exists on both classes. Set it on `LitAPI` (cleaner; self-documents
  the endpoint alongside the API logic).
