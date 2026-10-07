# Neural Forge — Audit of the original project & roadmap

_Audit performed at commit `0fd3950` ("Clean up demo section in README"), before any Neural Forge work._

## 1. Original architecture

```
frontend/  React 19 + TypeScript + Vite 8 + Tailwind 3
  App.tsx                 tab shell, Build / Simulate buttons, global status
  store/networkStore.ts   one Zustand store: configs, graph, steps, results, UI state
  api/client.ts           axios client for the FastAPI backend
  components/
    Visualizations/       15 tab components (D3 2D graph, R3F 3D graph, Plotly, canvas)
    Sidebar/              network + training configuration
    CinemaMode, Tutorial, DatasetPreview, DatasetUpload, Layout/Header
backend/   FastAPI, single module-level state
  main.py      REST endpoints, global `_network_graph`
  compute.py   graph building, propagation steps, decision boundary, loss landscape, "training"
  models.py    7 PyTorch nn.Module classes (ANN, CNN, RNN, LSTM, GAN, Transformer, Diffuser)
  datasets.py  4 synthetic 2-D datasets (Circle, Gaussian, XOR, Spiral)
Visualization.py  legacy Dash app (3 173 lines, unused by the new stack)
```

The frontend and backend communicate through plain JSON graphs:
`{nodes: [{id, x, y, name, layer, layer_type, value, ...}], edges: [{source, target, weight, layer}]}`.

## 2. Features that existed

Architecture builder (7 model types, 1–5 layers, activations, losses, regularisation), 2D (D3) and 3D
(Three.js) graph, forward/backward step animation, weight histograms, activation heatmap, pruning view,
attention heatmap, decision boundary, loss landscape, training curves, in-browser TF.js training ("Live
Train"), LR sweep, custom activation designer, architecture comparison, PyTorch/Keras code export with a
template editor, guided tour, cinema mode, 4 themes, dataset preview and CSV upload.

## 3. Important technical findings

| Finding | Where | Consequence |
|---|---|---|
| **Graph values were random**, not computed by any model. ANN values came from a NumPy forward pass over `np.random` weights that were never used anywhere else; all other model types used `np.random.rand()` per node. | `compute.build_network_graph` | Every "value", "weight", "bias" in the tooltips, histograms, activation heatmap and pruning view was decorative. |
| **Backprop gradients were `np.random.rand() * 0.5`.** | `compute.get_backward_propagation_steps` | The vanishing/exploding colouring meant nothing. It was also never displayed: `PropagationView` did not pass `gradients` to the graph when a `NetworkViewComponent` was supplied (always, from `App`). |
| **"Simulate Training" fabricated its curves** with a closed-form decay + noise (e.g. ~95 % accuracy regardless of the data). No model was trained. | `compute.simulate_training` | The Training tab (badge "Simulated") showed invented numbers. |
| **Decision boundary used a freshly initialised, untrained model.** | `compute.compute_decision_boundary` | Unrelated to the "training" that preceded it. |
| **Loss landscape** is a genuine Li et al. filter-normalised slice, but around a random-init model, and silently falls back to a synthetic surface on error. | `compute.compute_loss_landscape` | Real computation, but not of the user's trained network. |
| **Attention heatmap** is synthesised from a formula on the client. | `AttentionHeatmap.tsx` | Decorative. |
| Plotly was loaded from `cdn.plot.ly` via `index.html` (the npm package was a dependency but unused). | `index.html` | Charts needed internet access. |
| Uploaded CSV datasets were stored but **not used by any computation**. | `DatasetUpload`, store | Dead feature. |
| `input_nodes` / `output_nodes` sliders are ignored by the backend (input is always 2, output 1). | `NetworkConfig.tsx`, `main.py` | Misleading controls (still true for the legacy illustrative graphs). |
| Global mutable backend state (`_network_graph`). | `main.py` | Not session-safe. |
| No automated tests (Python or TypeScript); 27 pre-existing ESLint errors. | — | — |
| `NetworkView` was a component created inside `TabContent`'s render. | `App.tsx` | Remounted the graph on every render. |

## 4. Reusable components

* D3 `NetworkGraph` (zoom/pan, tooltip, legend) and R3F `Network3DView` (orbit, node picking) — extended,
  not replaced.
* `PropagationView` step player, `WeightHistogram`, `LayerActivationHeatmap`, `PruningView` — they consume
  the generic graph, so they automatically show real values once the graph is real.
* Synthetic datasets, theme system, tab shell, Zustand store, Tutorial/Cinema.
* TF.js tools (Live Train, LR sweep, Custom Activation) are self-contained and genuinely train in the browser.

## 5. Neural Forge architecture (implemented in this milestone)

See [ARCHITECTURE.md](ARCHITECTURE.md). In short: a new backend package `backend/forge/` owns a **real,
functional PyTorch MLP**, an **intervention overlay**, a **bounded checkpoint store**, and an
**introspector** that turns forward/backward passes into a typed wire contract. The frontend gets a matching
`src/forge/` layer (types, API, store, pure helpers) and `components/Forge/` (Neural Microscope UI). ANN builds
now use this real model everywhere; other model types are explicitly labelled _illustrative_.

## 6. Roadmap

Milestones are ordered by value / dependency. Each should keep the rule: **no number is shown as a model
internal unless a model computed it.**

### M1 — Foundation + Neural Microscope + What-if (this milestone, done)
* Real MLP sessions, real training (Adam + CE), per-epoch checkpoints with thinning.
* Neuron / layer / connection inspection (inputs, weights, contributions, bias, z, activation, gradients,
  dataset statistics, response maps, tensor shapes) with Learn and Lab modes.
* Interventions: ablate neuron, set weight, set bias; undo/reset; before/after prediction, dataset accuracy,
  flip rate and decision regions.
* Provenance badges everywhere (`REAL MODEL VALUES` vs `ILLUSTRATIVE VALUES`).

### M2 — Training Time Machine (done)
* New *Time Machine* tab: timeline instrument (real loss/accuracy log, checkpoint ticks, training events,
  drag-to-scrub, step, first/latest, play/pause at 0.5–4×, back to live, linear/even axis, keyboard).
* Per-checkpoint frames: decision regions with the 0.5 contour, misclassified points, change since the previous
  checkpoint; Learn-mode narration from real numbers.
* The selected neuron/layer/connection followed through time (history charts, historical Microscope).
* Epoch A ↔ B comparison: metric deltas, prediction changes (fixed/broken), decision regions A/B/difference,
  per-layer parameter change, selected component side by side.
* Training health (Lab): per-epoch gradient and update norms logged during real training; dead/saturated
  fractions and norms measured per checkpoint.
* Checkpoint retention replaced: balanced linear/log spacing, capacity 48, immutability enforced and tested.
* The Microscope's Weights select became a compact timeline; LIVE / HISTORICAL / WHAT-IF labels everywhere.
* Not done (candidates): float16 delta storage, more than two epochs compared at once, history export.

### M3 — Real backprop playback and 3D upgrade (next)
* Combine with the Time Machine: replay one probe's forward and backward pass *at any checkpoint*, step by step.
* Drive the Forward/Backprop tabs and Cinema Mode from the real trace (values per step, not just
  highlighting); colour 3D nodes/edges by real activation and gradient magnitude; animate signal intensity
  proportional to `w·a`.
* Instanced meshes for >2k edges in 3D; layer-focus camera moves.

### M4 — Model families beyond MLP
* Introduce an adapter protocol (`structure / run_probe / inspect / compare`) extracted from
  `MLPIntrospector` once a second implementation exists (avoid premature abstraction).
* Small CNN on 8×8 digits (feature maps, kernels, channel ablation).
* Replace the illustrative CNN/RNN/… graphs with real ones, or keep them clearly labelled.

### M5 — Transformer Lab
* A tiny character-level Transformer (≈2 layers, 2–4 heads, d≈64) trained locally on a small corpus.
* Tokens → embeddings → per-head attention matrices → MLP blocks → logits → next-token probabilities, with
  head ablation through the same intervention overlay pattern.

### Cross-cutting
* Persist sessions (optional) and allow multiple uvicorn workers (currently in-process LRU, single worker).
* Make the remaining legacy endpoints honest or remove them (`simulate-training`, random backprop
  gradients for illustrative graphs).
* Fix the 16 remaining pre-existing ESLint errors in untouched files.
* E2E browser tests (Playwright) for the microscope and time-machine flows (M2 was verified manually in a real
  browser with scripted Playwright runs, which are not yet part of the repository's test suite).
* The `NetworkGraph` layer headers overlap when the graph pane is narrow (pre-existing component).
