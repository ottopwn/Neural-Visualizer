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
│ forge/timeMachine.ts      playhead, playback, frame/history/A-B │
│        │ syncs checkpoint  caches (drives forge/store below)     │
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
│ timemachine.py   read-only history: timeline, frames, A/B diff │
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

### 5. Checkpoints (lifecycle)
1. `create_session` records **epoch 0** (the random initialisation).
2. Every training epoch calls `ModelSession.record(stats)`: the epoch's loss/accuracy (evaluated on the whole
   dataset) and its training statistics are appended to `history` (never thinned), and a float32 copy of the
   parameters is offered to the `CheckpointStore`.
3. The store keeps at most `capacity` (**48**) snapshots. When full it evicts the interior snapshot whose removal
   leaves the smallest gap on a *balanced* time axis (half linear, half logarithmic in the epoch). The first and
   the latest snapshot are never evicted, so **the live weights are always also the latest checkpoint**. This
   keeps the early, fast-changing epochs dense without leaving long blind spots later (the M1 policy, "drop every
   second snapshot", left nothing between epoch 0 and 224 after 300 epochs). Memory is O(capacity × params).
4. Snapshots are immutable: `Checkpoint` is a frozen dataclass and every consumer clones parameters before any
   in-place operation (`compile_interventions`, `probe_pass`). `tests/test_timemachine.py` runs every endpoint —
   including what-if interventions on historical checkpoints — and asserts every stored tensor is bit-identical
   afterwards.
5. Re-recording an epoch ≤ the latest truncates newer snapshots (not reachable from the UI today; training
   always appends).

Training statistics logged per epoch (`history` rows): `grad_norm` / `layer_grad_norms` — the L2 norm of the
gradient actually used for each Adam step (loss including regularisation), averaged over the epoch's
mini-batches; `update_norm` / `layer_update_norms` — `‖θ_end − θ_start‖` of the epoch. They cost one norm per
layer per step.

### 6. Training Time Machine
`timemachine.TimeMachine` is a read-only view over a session. It never applies interventions: it shows
history as it happened. Four endpoints:

| Endpoint | Cost | Used for |
|---|---|---|
| `GET /timeline` | one dataset forward per checkpoint | curves (full log), checkpoint health, runs, events — loaded once per training state |
| `POST /frame` | two dataset forwards + 2 boundary grids | the checkpoint under the playhead (metrics, predictions, P(class 1) grid, change vs the previous checkpoint) |
| `POST /component-history` | one probe pass per checkpoint | one neuron / layer / connection across all checkpoints |
| `POST /epoch-compare` | two of everything | A vs B deltas, decision grids, per-layer ‖Δθ‖, the selected component side by side |

`component_metrics()` is the single function that describes a component at one checkpoint; both the history
and the comparison use it, so they cannot disagree (a test checks it). Events (`init`, `run`, `acc_threshold`,
`best_accuracy`, `min_loss`, `largest_drop`) are derived from the training log and carry the nearest stored
checkpoint, which is what the UI jumps to.

**Only stored checkpoints are visitable.** The curves come from the per-epoch log; an epoch without a snapshot is
shown as "log only" on hover and cannot be selected. Nothing is interpolated: chart segments between two
checkpoints are plain connecting lines, and the decision map switches frame by frame without blending.

### 7. Live vs historical vs what-if
| State | Source | Label |
|---|---|---|
| Live | `session.params` (= latest checkpoint) | green `LIVE · EPOCH n` |
| Historical | an immutable stored checkpoint | indigo `HISTORICAL · EPOCH n` |
| What-if | live *or* historical params + the intervention overlay, per request | amber `WHAT-IF ×n` |

What-if on a historical epoch is a **temporary overlay** (decision: allowed rather than disabled — the overlay
architecture already guarantees immutability, and asking "what if this neuron were disabled at epoch 10?" is a
useful experiment). The Microscope, graph and before/after panel show the overlay; the Time Machine's own views
(timeline, frames, component history, A/B comparison) always show the stored checkpoints, and a banner says so
while edits are active. Training drops the intervention list (it referred to older weights) and returns to live.

### 8. Frontend: playhead, sync, caching
`forge/timeMachine.ts` (zustand) owns the playhead (`cursor`: a stored epoch or `null` = live), playback and the
A/B selection. It drives the existing experiment store rather than duplicating it:

```
goTo / step / play ──► cursor ──► frame (cached by session@liveEpoch : epoch : probe)
                         │
                         └─(debounced 120 ms)──► useForgeStore.setCheckpoint(cursor)
                                                   └─► graph + compare + inspection refetch
                                                       (same selection, same probe, same what-if list)
```

* Scrubbing moves the playhead and the frame immediately; the heavier Microscope refresh runs once scrubbing
  pauses for 120 ms.
* Frames, histories and comparisons are LRU-cached (immutable per key); frame requests in flight are shared
  with prefetching (the next frame is prefetched during playback). History and comparison requests are
  cancelled with `AbortController` when superseded; every response is also sequence-checked, so stale or
  out-of-order responses can never overwrite newer state.
* Playback advances only after the current frame is on screen for one interval (700 ms / speed), so a slow
  response slows playback down instead of skipping checkpoints. Manual navigation pauses playback. Reaching
  the end lands on live and stops; Play at the end restarts from epoch 0.
* A new or retrained session resets the playhead to live, clears caches, cancels a pending sync, and snaps the
  A/B epochs to surviving checkpoints.

UI components live in `components/Forge/TimeMachine/`: `TimelineInstrument` (+ `Transport`), `BoundaryStage`,
`ThroughTime` (also embedded compactly in the Inspector), `EpochCompareView`, `HealthPanel`, `StatePill`. Learn
mode sentences come from `forge/explainTime.ts`, templates filled only with payload values (e.g. "not better than
always guessing the most common class" is only said when accuracy ≤ majority-class rate + 2 pp).

### Known limitations
* Checkpoints live in memory with the session (lost on backend restart; single uvicorn worker).
* With more than 48 epochs not every epoch can be visited; the curves still show every epoch.
* Epoch 0 has no training statistics (no step has run). Component gradients in the history/comparison are
  computed on the single probe input, not averaged over the dataset; training gradient norms are per layer only.
* Two checkpoints can be compared, but not more at once; there is no export of the history yet.
* Decision regions, response maps and the A/B "where the class changed" map need 2-D inputs; other inputs get
  metrics only. The Time Machine supports the ANN/MLP binary classifier only.
* The frame payload carries one prediction and confidence per sample (≤ 2000 samples): fine locally, not tuned
  for remote deployments.

### 9. Sessions

`SessionRegistry` is an 8-entry LRU in the uvicorn process, guarded by a per-session lock. Sessions are
lost on restart and are not shared across workers — run a single worker. This is intentional for a
locally run lab; a persistent store can be added behind the same `get/add` interface.

### 10. Graph compatibility
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
| GET | `/sessions/{id}/timeline` | — | `Timeline` (full log, per-checkpoint `LayerHealth`, runs, events, majority rate) |
| POST | `/sessions/{id}/frame` | `{checkpoint_epoch?, probe}` | `Frame` (metrics, predictions, confidence, P(class 1) grid, `previous`) |
| POST | `/sessions/{id}/component-history` | `{ref, probe}` | `ComponentHistory` (`epochs`, `series[]`) |
| POST | `/sessions/{id}/epoch-compare` | `{epoch_a, epoch_b?, probe, ref?}` | `EpochComparison` (deltas, grids, `ParamChange[]`, component rows) |

Time-machine endpoints accept only stored checkpoint epochs (`422` otherwise) and never take interventions.

`ExperimentRequest = {probe: {x?, sample_index?, target?}, interventions: Intervention[], checkpoint_epoch?}`  
`ref = {kind: "neuron", layer, index} | {kind: "layer", layer} | {kind: "connection", layer, source, target}`  
Errors: `404` unknown/expired session, `422` invalid reference, probe, intervention or config.

## Frontend state flow

```
timeline / transport ──► useTimeMachine (cursor, playback, A/B) ──(debounced)──► checkpoint ┐
user action ──► useForgeStore (probe | interventions | checkpoint | selection) ◄─────────────┘
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
cd backend && pip install -r requirements-dev.txt && python -m pytest -q   # 61 tests
cd frontend && npm test                                                    # 59 tests (vitest)
cd frontend && npm run typecheck && npm run build
```

`tests/test_timemachine.py` checks the retention policy, frozen snapshots, the logged gradient/update norms
against independent autograd, frames / histories / A-B comparisons against independent forward passes, that
history and comparison agree, and that no endpoint (with or without what-if) changes any stored tensor.
`src/forge/__tests__/timeMachine.test.ts` covers timeline selection, debounced Microscope sync, playback
(play, pause, speed, end-of-history, waiting for slow frames), manual navigation, caching, stale/out-of-order
and cancelled responses, selection persistence across epochs, what-if/undo on a historical epoch, retraining
and A/B selection; `timeline.test.ts` covers the pure helpers and the Learn-mode narration.
