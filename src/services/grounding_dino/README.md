# Model services

Separately run model servers that the router calls over HTTP (svc-1.0 contract in
`src/services/service_contracts.py`). Each server has its own pixi workspace and lock; the
root `pyproject.toml` / `pixi.lock` are not involved, and the `routing` package never imports
server code. Servers bind to `127.0.0.1` by default, so images stay on this machine.

The server environment lives outside the source tree: this workspace's local pixi config
sets `detached-environments = true` (run once per machine:
`pixi config set --local -m src/services/grounding_dino/pixi.toml detached-environments true`).
Without it, pixi would create a multi-GB `.pixi/` env inside `src/services/grounding_dino/`
(gitignored, but walked by anything that scans `src/`).

## grounding_dino (detector)

| | |
|---|---|
| Model | Hugging Face `IDEA-Research/grounding-dino-tiny` (license tag `apache-2.0`), revision `a2bb814dd30d776dcf7e30523b00659f4f141c71` |
| Weights | `model.safetensors`, 689,359,096 bytes, SHA-256 `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`, in the Hugging Face cache (outside git). The server refuses to start on a digest mismatch. |
| Endpoint | `POST http://127.0.0.1:8001/detect` |
| Reported identity | `model_id=grounding-dino-tiny`, `model_revision=a2bb814dd30d+bt0.35+tt0.25` (box/text thresholds are part of the revision, so changing them changes the router's run key) |
| Class rule | One forward pass per requested class, prompted only with that class's positive prompts. Every box from a pass belongs to that class; returned text spans are ignored. |

```bash
pixi run -m src/services/grounding_dino/pixi.toml fetch   # one-time pinned download
pixi run -m src/services/grounding_dino/pixi.toml test    # server tests, fake model (no weights/GPU)
pixi run -m src/services/grounding_dino/pixi.toml smoke   # real model: checks + measurements (JSON)
pixi run -m src/services/grounding_dino/pixi.toml serve   # start the server
```

Then, from the repo root: `pixi run test-gdino` (live client test), or

```bash
pixi run vad-route shadow --config ... --images ... --out runs/gd1 \
  --primary-url http://127.0.0.1:8001 --primary-model-id grounding-dino-tiny \
  --primary-revision a2bb814dd30d+bt0.35+tt0.25 \
  --primary-weights-sha256 1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3
```

### Why one pass per class

With all classes in one prompt, Grounding DINO returns text spans that can merge or overlap.
Probing `"a blue square. a blue circle."` returned spans such as `"blue square a blue circle"`
and `"a a blue circle"`, which map to no single class. Mapping those would silently drop or
misassign boxes. Per-class passes make attribution exact, at the cost of N passes for N classes.

### Measured (2026-11-09, RTX 5070 12 GB, CUDA, transformers 5.22.2, 320×240 synthetic images, 2 classes)

| Measurement | Value |
|---|---|
| Model load | 4.3 s |
| Latency per image (2 classes = 2 passes) | about 390 ms (server `gpu_ms` about 300 ms) |
| Peak torch VRAM in the server | 1,757 MiB |
| Total GPU memory in use, with LM Studio `qwen3-vl-4b-instruct` loaded | 9,590 MiB of 12,227 MiB (7,204 MiB before the server started) |

The CLIP scorer was not loaded at the same time; expect roughly another 0.6–1 GB.

### Known false positives (synthetic images)

Grounding DINO boxes *something* for every prompt. Measured with `smoke`:
- absent class `"a dog"` on an image with only a blue square: a dog box on the square, score 0.71;
- a blank white image: a whole-image `"a blue square"` box, score 0.83.

A joint prompt and added background prompts reduced but did not remove these. Treat scores as
raw candidates: the router's verification and review paths handle them, and per-class
cross-class duplicates are flagged as conflicts. Prompt strategy and thresholds must be
chosen on real trial images, not on these synthetic ones.
