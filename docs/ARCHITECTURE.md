# Neural Forge — Architecture

This document describes the introspection architecture added on top of the original Neural Visualizer
and how to extend it. For the audit of the original code and the milestone plan see
[AUDIT_AND_ROADMAP.md](AUDIT_AND_ROADMAP.md).

## Guiding rule

> A number is presented as a model internal **only** if a model computed it.

Every introspection payload carries a `Provenance` block (`source: "model"`, the epoch of the parameters
used, whether they are the live weights, and how many interventions were overlaid). The UI shows it as a
green `REAL · …` badge. Graphs that do not come from a model (CNN, RNN, LSTM, GAN, Transformer, Diffuser)
are labelled `ILLUSTRATIVE VALUES`.

## Layers

```
┌────────────────────────── frontend ──────────────────────────┐
│ components/Forge/*        Microscope UI (React, no ML logic)  │
│        │ reads                                                 │
│ forge/store.ts            experiment state: probe, interventions,
│        │                  checkpoint, selection → refetch       │
│ forge/api.ts              HTTP client                          │
│ forge/types.ts            wire contract (mirror of schema.py)  │
│ forge/{interventions,selection,format,explain,activations}.ts │
│                           pure helpers (unit-tested)           │
└────────────────────────────┬──────────────────────────────────┘
                             │ JSON over /api/forge/*
┌────────────────────────────┴──────── backend/forge ───────────┐
│ api.py           FastAPI router, validation, session lookup    │
│ introspect.py    MLPIntrospector: session → schema payloads    │
│ schema.py        pydantic wire contract                        │
│ session.py       ModelSession (params, data, training, history)│
│                  + SessionRegistry (in-process LRU)            │
│ checkpoints.py   bounded snapshot store                        │
│ interventions.py what-if overlay (non-destructive)             │
│ mlp.py           functional MLP + ForwardTrace (pure PyTorch)  │
└────────────────────────────────────────────────────────────────┘
```

Each backend module only imports from modules listed below it. `introspect.py` is the only place that
knows both PyTorch and the wire format; visual components only know `forge/types.ts`.

## Key decisions

### 1. Functional MLP instead of `nn.Module`
`mlp.forward(spec, params, x, neuron_masks)` takes the parameters explicitly and returns a `ForwardTrace`
with every pre-activation `z` and activation `a`. The same code therefore runs on the live weights, any
checkpoint, or the live weights with interventions, without mutating anything and without hooks.
Correctness is pinned by `tests/test_mlp.py`, which compares it against an independent `nn.Sequential`
for every supported activation.

Layer indexing is shared with the UI: graph layer `0` = input features, `1..H` = hidden dense layers,
`H+1` = output (logits → softmax). `params[k]` belongs to graph layer `k+1`.

### 2. The backend is stateless with respect to experiments
A session stores only model state (parameters, dataset, history, checkpoints). The **probe input**, the
**intervention list** and the **checkpoint being viewed** travel with every request
(`ExperimentRequest`). Consequences:
* undo/reset are list operations in the frontend store; nothing can drift out of sync on the server;
* the same request is reproducible and cacheable;
* before/after comparisons are two evaluations of the same request with `[]` vs the list.

### 3. Interventions are an overlay
`compile_interventions` clones the parameters, applies `set_weight` / `set_bias`, and builds per-layer
neuron masks for `ablate_neuron` (the neuron's output is multiplied by 0 *after* its activation, so its
own `z` stays observable and the UI can show the "natural" value it would have sent). Later edits of the
same element win. Output neurons cannot be ablated (it would not be a valid probability distribution);
their bias can be edited.

### 4. Gradients
For the probe, `loss = cross_entropy(logits, target)`; the target is the probe's true label for dataset
rows, the user's choice, or otherwise the predicted class (`ResolvedProbe.target_source` says which).
The introspector runs one autograd pass with `retain_grad` on every `z` and `a`, plus the input
(saliency). For output neurons the reported gradient is `dLoss/dlogit` because the loss is defined on
logits. `tests/test_introspection.py` checks `dL/db`, `dL/dW`, `dL/dx` against an independent autograd
pass and the chain rule `dL/dw_ij = dL/dz_i · a_j`.

### 5. Checkpoints
`CheckpointStore` keeps at most `capacity` (32) snapshots. When full it drops every second snapshot except
the first and last, so spacing becomes roughly geometric and memory is bounded regardless of training
length. Every introspection endpoint accepts `checkpoint_epoch`; the UI exposes this as the **Weights**
selector. Retraining re-records epochs and truncates any newer history.

### 6. Sessions
`SessionRegistry` is an 8-entry LRU in the uvicorn process, guarded by a per-session lock. Sessions are
lost on restart and are not shared across workers — run a single worker. This is intentional for a
locally run lab; a persistent store can be added behind the same `get/add` interface.

### 7. Graph compatibility
`POST /graph` returns the classic `{nodes, edges}` shape (plus `index`, `grad`, `ablated`, `edited`) and
real forward/backward step lists. `App.tsx` subscribes via `onForgeGraph` and pushes it into the legacy
`networkStore`, so the Architecture, Forward, Backprop, Weights, Activations and Pruning tabs display real
values for ANN without modification.

## HTTP API (`/api/forge`)

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/capabilities` | — | supported model types, interventions, limits |
| POST | `/sessions` | `neurons, activations, dataset, noise, custom_dataset?, seed?` | `SessionSummary` (untrained, epoch 0) |
| GET | `/sessions/{id}` | — | `SessionSummary` |
| POST | `/sessions/{id}/train` | `epochs, learning_rate, batch_size, reg_type, reg_rate` | `{summary, new_rows}` |
| POST | `/sessions/{id}/graph` | `ExperimentRequest` | `ForgeGraph` |
| POST | `/sessions/{id}/inspect` | `ExperimentRequest + ref` | `NeuronInspection \| LayerInspection \| ConnectionInspection` |
| POST | `/sessions/{id}/compare` | `ExperimentRequest` | `Comparison` (baseline vs intervened, decision grids) |

`ExperimentRequest = {probe: {x?, sample_index?, target?}, interventions: Intervention[], checkpoint_epoch?}`  
`ref = {kind: "neuron", layer, index} | {kind: "layer", layer} | {kind: "connection", layer, source, target}`  
Errors: `404` unknown/expired session, `422` invalid reference, probe, intervention or config.

## Frontend state flow

```
user action ──► useForgeStore (probe | interventions | checkpoint | selection)
                    │  refresh(): graph + compare in parallel, inspection separately
                    │  each response tagged with a sequence number; stale ones are dropped
                    ▼
            ForgeGraph ──► MicroscopeView (NetworkGraph mode="signal")
                       └─► onForgeGraph listeners ──► networkStore (legacy tabs)
            Comparison ──► ProbePicker (decision regions), WhatIfPanel (before/after)
            Inspection ──► Inspector → NeuronPanel | LayerPanel | ConnectionPanel
```

Learn vs Lab mode only changes presentation: Learn shows sentences generated from the real values
(`forge/explain.ts`) and hides tensors; Lab shows equations, gradients, shapes and statistics.

## How to extend

**Add an intervention type** (e.g. clamp a neuron to a value):
1. add a pydantic model to `interventions.py` and the `Intervention` union; validate it;
2. apply it in `compile_interventions` (a mask/offset pair per layer is enough for clamping);
3. mirror the type in `forge/types.ts`, describe it in `forge/interventions.ts`, add a button in a panel;
4. add a test in `tests/test_interventions.py` that checks the exact numeric effect.

**Add a model family** (e.g. small CNN): implement a functional forward with a trace, a session that
owns it, and an introspector producing the same `schema` payload kinds (add new kinds such as
`feature_map` where needed). Extract a shared adapter protocol from `MLPIntrospector` at that point and
register the model type in `FORGE_MODELS` (`App.tsx`) and `/capabilities`.

## Tests

```bash
cd backend && pip install -r requirements-dev.txt && python -m pytest -q   # 42 tests
cd frontend && npm test                                                    # 21 tests (vitest)
cd frontend && npm run typecheck && npm run build
```
