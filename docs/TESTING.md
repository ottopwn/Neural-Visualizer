# Neural Forge — test strategy

Three layers of tests, all run by `.github/workflows/ci.yml` on every push and pull request.

## 1. Backend numerics (pytest, `backend/tests`, 74 tests)

The guiding rule — *a number is shown as a model internal only if a model computed it* — is enforced by comparing what
the API returns with **independent** computations:

| File | What it pins down |
|---|---|
| `test_mlp.py` | The functional MLP equals an `nn.Sequential` for every activation. |
| `test_introspection.py` | Microscope payloads: `z = Σ w·a + b`, gradients vs an independent autograd pass, response maps, provenance. |
| `test_interventions.py` | Exact numeric effect of ablation / weight / bias edits; validation; stored weights untouched. |
| `test_checkpoints.py`, `test_timemachine.py` | Retention policy, frozen snapshots, logged gradient/update norms, frames / histories / A-B comparisons vs independent passes; no endpoint mutates a checkpoint. |
| `test_training_and_api.py` | Real training improves and is deterministic; full HTTP flow; error codes. |
| `test_trace.py` | Pass Explorer trace vs independent autograd (every layer's `z`, `a`, `dL/da`, `dL/dz`, `dL/dW`, `dL/db`, saliency), the chain-rule identities, interventions + historical checkpoints, the SGD preview, the loss landscape (centre = real loss, reproducible). |
| `test_transformer.py` | Transformer Lab trace vs an independent re-implementation (Q, scores, attention, head outputs, logits), causal mask, softmax rows, residual identities, head ablation, truncation, training. |

## 2. Frontend logic (Vitest, `frontend/src/forge/__tests__`, 84 tests)

| File | What it pins down |
|---|---|
| `store.test.ts` | Every request carries probe, what-if list and checkpoint; stale / out-of-order responses (graph, comparison, inspection, **trace**) are dropped; undo/reset. |
| `timeMachine.test.ts`, `timeline.test.ts` | Playhead, playback, debounced sync, caching, cancellation, retraining, A/B selection, Learn narration. |
| `passExplorer.test.ts` | Step lists and every identity the explorer displays, checked on a **real trace produced by the backend** (`fixtures/trace.json`, MLP 2→3→4→2 with a disabled neuron). |
| `scene3d.test.ts` | 3D layout, node/edge values per colour mode taken from the trace, top-k edge selection and its disclosure, selection emphasis. |
| `interventions.test.ts`, `helpers.test.ts` | What-if list semantics, formatting, colour scales. |

## 3. Browser smoke tests (Playwright, `frontend/e2e`, 9 tests)

`playwright.config.ts` starts the real FastAPI backend and the Vite dev server, then drives Chromium. Selectors use roles,
accessible names and a few `data-testid`s; any console error fails a test.

| Test | Flow |
|---|---|
| A (dark + Paper) | Build → train → Time Machine → first checkpoint → select a neuron → Microscope shows `checkpoint · epoch 0` → back to live. |
| B | What-if: disable a neuron → before/after panel, `WHAT-IF ×1` → Undo. |
| C | Forward explorer: linear step equation, activation step, prediction. |
| D | Backward explorer: loss, `Output · dW`, SGD preview, *compare with epoch 0*. |
| E | 3D: canvas renders, edge disclosure, edge limit, layer label selects the layer, backward pass sync. |
| F | First-run welcome → "Break the network" demo (real what-if); welcome not shown again. Presentation: first three steps on the real model. |
| G | Transformer Lab: attention matrix, `q·k·scale` computation, head ablation and restore. |

Run locally (needs the backend Python deps):

```bash
cd frontend
npx playwright install chromium   # once
npm run test:e2e
```

## Manual verification done for M3

Screenshots in `docs/screenshots/` were taken at 1680×1000; the flows above were also run at 1280×800 (dark) and
1440–1920 wide (Paper), with deliberate stress: repeated training, rapid workspace switching, fast Time Machine
scrubbing, what-if on a historical checkpoint followed by retraining, architecture and dataset changes, Learn/Lab and
theme toggling, collapsing panels, an illustrative model type, and the backend stopped (clear "Backend unreachable"
messages, no uncaught errors).
