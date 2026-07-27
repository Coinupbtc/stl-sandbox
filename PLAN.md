# STL Sandbox — Plan: Prompt → 3D-Printable File

Historical design notes (2026-07). Status: **built** — see `README.md` for the live architecture.

## Goal

Turn a short English prompt into a manifold, printable STL/STEP solid via **LLM-generated CadQuery**, with a generate → execute → validate → retry loop. Organic shapes can use a separate image→3D path when ComfyUI is available.

## Approach chosen

| Approach | When |
|---|---|
| **LLM writes CadQuery** (OpenCascade B-rep) | Functional parts: cases, stands, washers, mugs — manifold by construction |
| **Keyword / parametric templates** | Known product shapes (phone cases, stands) without an LLM call |
| **Image→3D (optional HQ)** | Organic figurines when a local ComfyUI stack is configured |

Dominant research pattern (Text-to-CadQuery / CADSmith-style):

1. LLM emits CadQuery Python from the prompt  
2. Run in an isolated subprocess (timeout + resource limits)  
3. On exception, feed the traceback back for up to ~3 retries  
4. Export STL + STEP from the same solid  

## Architecture (as built)

```
prompt ──┬─► fast path / product templates (no LLM)
         │
         └─► LLM code-gen → sandboxed exec → retry on error
                    │
                    ▼
              STL + STEP + quality / print-assistant checks
```

| Decision | Choice |
|---|---|
| CAD library | CadQuery (OpenCascade) |
| LLM endpoint | OpenAI-compatible HTTP API (`STL_SANDBOX_LLM_URL`, default `http://127.0.0.1:8889/v1`) |
| Sandbox | Subprocess + timeout + `resource` limits; no network in the CAD process |
| Host bind | `127.0.0.1` by default (`BIND_HOST` / `SANDBOX_PORT`) |

## Ops (portable)

- Configure paths with env vars (`COMFY_ROOT`, `STL_SANDBOX_LLM_URL`, …) — no machine-specific hardcodes in the public tree.  
- Prefer a user systemd unit bound to localhost for a long-running service.  
- Failures in a production deploy should page your own alerting path; this repo does not ship host-specific alert scripts.

## Status

- v2/v3 shipped: templates, dimension research, ISO-style quality scoring, print assistant, organic path, tests.  
- Run: see `README.md` and `install.sh`.
