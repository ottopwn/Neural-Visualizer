<!-- <p align="center">
  <img src="https://github.com/user-attachments/assets/970498f7-008c-487b-b58e-5330f2770ca9" alt="Neural Visualizer" width="100%" />
</p> -->

<h1 align="center">Neural Forge</h1>

<p align="center">
  <strong>A laboratory for looking inside a real neural network — and changing it.</strong><br/>
  Build and train a network, click any neuron, layer or connection to see what it actually computes,
  then disable or rewire it and watch the prediction change.
</p>

<p align="center"><sub>Neural Forge is evolving from <a href="https://github.com/PeakScripter/Neural-Visualizer">Neural Visualizer</a> by PeakScripter (GPL-3.0); all original features are preserved.</sub></p>

<p align="center">
  <img src="https://img.shields.io/badge/React_19-61DAFB?logo=react&logoColor=black&style=flat-square" />
  <img src="https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white&style=flat-square" />
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white&style=flat-square" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=flat-square" />
  <img src="https://img.shields.io/badge/TensorFlow.js-FF6F00?logo=tensorflow&logoColor=white&style=flat-square" />
  <img src="https://img.shields.io/badge/Three.js-000000?logo=threedotjs&logoColor=white&style=flat-square" />
  <img src="https://img.shields.io/badge/License-GPL_v3-blue?style=flat-square" />
</p>

---

## 🎬 Demo

<!-- <p align="center"> -->
https://github.com/user-attachments/assets/fc5b9114-0085-40b6-bdf9-fa3208ce19ff
<!-- </p> -->

---

## 🔬 New in Neural Forge

> Status: first milestone. The Microscope and What-if mode work for **ANN (MLP) models**.
> Other model types are still drawn as illustrative diagrams and are labelled as such.

### Neural Microscope (tab *Forge → Microscope*)
Build an **ANN**, train it, then click any neuron, layer header, or contribution bar. Everything shown is
computed by a PyTorch forward/backward pass of *your* model on a probe input you choose:

- **Neuron** — its inputs, weights, each input's contribution `w·a`, bias, pre-activation `z`, activation
  `f(z)` drawn on the activation curve, gradients (`dL/da`, `dL/dz`, `dL/db`, `dL/dw`), its activation
  histogram over the whole dataset (with "dead neuron" detection), and a **response map** showing where in
  the input plane it fires.
- **Layer** — weight matrix heatmap, bias and activation statistics, never-active neurons, `dL/dW`, tensor shapes.
- **Connection** — weight, carried signal, share of the target's input and rank, gradient, dataset-wide statistics.
- **Probe input** — click a dataset point (true label used for gradients) or any free point on the decision map.
- **Weights selector** — inspect any stored training checkpoint (e.g. epoch 0 vs. epoch 50).
- **Learn / Lab modes** — Learn explains each component in plain language generated from the real values;
  Lab exposes the numbers, equations, gradients and shapes.
- Graph "signal" view: edge thickness/colour = real contribution `w·a` on the current probe.

### What-if mode
From the inspector you can **disable a neuron**, **cut a connection**, or **set a weight / bias**. The
change is applied to the real forward pass (as an overlay — stored weights are never modified) and the app
shows **original vs. after** class probabilities, dataset accuracy, the fraction of predictions that flipped,
and decision regions before and after. **Undo**, **Reset** and per-component **Restore** are available.

### Honest data everywhere
- ANN builds now create a real model session: the Architecture, Forward, Backprop, Weights, Activations
  and Pruning tabs show its real values and real gradients (`REAL MODEL VALUES` badge).
- **Train Model** (ANN) runs real mini-batch training (Adam, cross-entropy) and records checkpoints; the
  Training and Decision tabs show the real results.
- Graphs for other model types carry an `ILLUSTRATIVE VALUES` badge; synthetic charts say so in their badge.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/AUDIT_AND_ROADMAP.md](docs/AUDIT_AND_ROADMAP.md).

---

## ✨ What was new in v3.0 (Neural Visualizer)

The entire application has been **rebuilt from scratch** — migrated from a single-file Dash/Plotly app to a modern **React + FastAPI** architecture with 15+ interactive visualization tabs, in-browser training, and a polished dark-mode UI.

**Recent additions**
- **Guided Tour** — 20-step interactive tutorial with pulsing element highlights and arrow badges; auto-starts on first visit, re-launchable from the header
- **Input / Output node controls** — configure the exact number of input features (1–16) and output neurons (1–10) per architecture
- **Code Template Editor** — write your own code scaffold using `{{variables}}` that auto-fill from current settings in real time (alongside the existing PyTorch / Keras generators)
- **3D layout fix** — Transformer and other architectures with non-contiguous layer indices now render with even spacing in the 3D view

---

## 🧠 Supported Model Types

| Model | Description |
|---|---|
| **ANN** | Fully-connected feedforward network |
| **CNN** | Convolutional neural network with pooling layers |
| **RNN** | Vanilla recurrent network |
| **LSTM** | Long Short-Term Memory network |
| **GAN** | Generator + Discriminator adversarial pair |
| **Transformer** | Multi-head self-attention encoder |
| **Diffuser** | U-Net style encoder–decoder with time embedding |

---

## 🚀 Features

### Network Building & Visualization
- **Interactive Architecture Builder** — Configure hidden layers (1–5), neurons per layer, activation functions (ReLU, Sigmoid, Tanh, LeakyReLU, ELU), loss functions, and regularization (L1 / L2 / L1L2) from the sidebar. *(The input/output node sliders only affect the code export; real ANN models use the dataset's 2 features and 2 classes.)*
- **2D & 3D Network Graphs** — Toggle between a D3-powered 2D layout and a fully interactive Three.js 3D view with orbit controls
- **Forward Propagation** — Step-by-step animation of data flowing through each layer with active node/edge highlighting
- **Backpropagation** — Step through gradient flow in reverse; for ANN the per-neuron gradients are real

### Training & Analysis
- **Train Model (ANN)** — Real server-side PyTorch training of the session model; loss & accuracy per epoch. For other model types the button is *Simulate Training* and the curve is synthetic (badge: *Synthetic curve*)
- **Live In-Browser Training (TF.js)** — Train your configured architecture entirely client-side using TensorFlow.js with real-time loss/accuracy canvas charts
- **Decision Boundaries** — For ANN, the regions of the trained session model; for other types, an untrained model (badge says which)
- **Loss Landscape** — A filter-normalised 2-D slice (Li et al. 2018) of the loss around a *randomly initialised* model of the chosen architecture

### Advanced Visualizations
- **Weight Distribution Histograms** — Inspect per-layer weight distributions (real for ANN)
- **Layer Activation Heatmaps** — Visualize activations across neurons and layers (real for ANN, on the current probe)
- **Attention Heatmaps** — A synthetic attention-like pattern (not computed from a model; labelled *Synthetic pattern*)
- **Network Pruning** — Visually hide neurons whose |activation| is below a threshold (a view filter; use What-if to change predictions)

### Tools
- **Learning Rate Sweep** — Batch-compare 5 learning rates side-by-side, trained with TF.js
- **Custom Activation Designer** — Draw your own activation function and watch it applied in a mini-network
- **Architecture Comparison** — Side-by-side 3D compare of two different network configurations with stat bars (nodes, edges, params)
- **Code Export** — Auto-generate ready-to-use PyTorch or Keras code from your current configuration; copy to clipboard with one click
- **Code Template Editor** — Switch to the Template tab to write your own scaffold using `{{variables}}` (`{{n_layers}}`, `{{input_nodes}}`, `{{loss_fn_code}}`, etc.) that fill in real time from the active config

### UI & Experience
- **Guided Tour** — 20-step interactive tutorial with pulsing highlights and directional arrow badges on every referenced UI element; auto-launches on first visit, re-accessible via the **Tour** button in the header; keyboard-navigable (← →, Esc)
- **4 Themes** — Dark, Cyberpunk, Matrix, Paper (light mode)
- **Cinema Mode** — Full-screen guided walkthrough of forward propagation with layer-by-layer narration, auto-play, and keyboard navigation
- **Dataset Preview** — Live scatter plot of the selected synthetic dataset
- **Custom Dataset Upload** — Load your own CSV data
- **Grouped Tab Bar** — 15 visualization tabs organized into Network · Analysis · Train · Tools groups
- **Framer Motion Animations** — Smooth tab transitions, status toasts, and micro-interactions throughout

### Datasets
Choose from **4 synthetic datasets** with adjustable noise:
- Circle · Gaussian · XOR · Spiral

---

## 🏗️ Architecture

```
Neural-Visualizer/
├── backend/                # Python FastAPI server
│   ├── main.py             # REST API endpoints (legacy + mounts /api/forge)
│   ├── forge/              # Neural Forge: real MLP, introspection, interventions, checkpoints
│   ├── tests/              # pytest suite for forge
│   ├── models.py           # PyTorch model definitions (7 architectures)
│   ├── compute.py          # Illustrative graphs, propagation steps, landscapes
│   ├── datasets.py         # Synthetic dataset generators
│   └── requirements.txt / requirements-dev.txt
├── frontend/               # React 19 + TypeScript + Vite
│   ├── src/
│   │   ├── forge/          # Forge types, API client, store, pure helpers (+ vitest tests)
│   │   ├── components/
│   │   │   ├── Forge/           # Neural Microscope & What-if UI
│   │   │   ├── Layout/          # Header with theme switcher & Tour button
│   │   │   ├── Sidebar/         # NetworkConfig, TrainingConfig panels
│   │   │   ├── Visualizations/  # 15 visualization components
│   │   │   ├── CinemaMode.tsx   # Full-screen guided walkthrough
│   │   │   ├── Tutorial.tsx     # 20-step interactive guided tour
│   │   │   ├── DatasetPreview.tsx
│   │   │   └── DatasetUpload.tsx
│   │   ├── api/            # Axios API client
│   │   ├── store/          # Zustand state management
│   │   ├── contexts/       # Theme context
│   │   └── types/          # TypeScript type definitions
│   └── package.json
├── docs/                   # Architecture, audit and roadmap
├── start.sh                # Launch both servers with one command
├── Visualization.py        # Legacy Dash app (preserved)
└── README.md
```

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 19, TypeScript, Vite, Tailwind CSS |
| **3D Rendering** | Three.js, React Three Fiber, Drei |
| **2D Charts** | D3.js, Plotly.js (bundled, lazy-loaded), HTML Canvas |
| **Animations** | Framer Motion |
| **State** | Zustand |
| **Icons** | Lucide React |
| **In-Browser ML** | TensorFlow.js |
| **Backend** | FastAPI, Uvicorn |
| **ML Engine** | PyTorch, scikit-learn, NumPy, SciPy |

---

## 🛠️ Getting Started

### Prerequisites
- **Python 3.10+** with pip
- **Node.js 18+** with npm

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/PeakScripter/Neural-Visualizer.git
cd Neural-Visualizer
```

**2. Set up the backend**
```bash
cd backend
pip install -r requirements.txt        # add requirements-dev.txt to run the tests
```

**3. Set up the frontend**
```bash
cd frontend
npm install
```

### Running the App

**Option A — Start both servers with one command (Linux/macOS)**
```bash
chmod +x start.sh
./start.sh
```

**Option B — Start each server separately**

Terminal 1 (Backend):
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

Then open your browser at **http://localhost:5173**

> **Note:** The backend runs on port `8000` and the frontend dev server on port `5173`; the frontend calls the backend at `VITE_API_BASE_URL` (default `http://localhost:8000`). Run uvicorn with a **single worker**: model sessions live in the backend process memory and are lost on restart (just click *Build Network* again).

### Running the tests

```bash
cd backend && python -m pytest -q        # numerical correctness of the microscope, interventions, training, API
cd frontend && npm test                  # store and helper unit tests (vitest)
cd frontend && npm run typecheck && npm run lint && npm run build
```

---

## 📖 Usage

1. **Follow the Tour** — a 20-step guided tutorial launches automatically on first visit; click **Tour** in the header to reopen it at any time
2. **Use the Microscope** — choose **ANN**, click **Build Network**, then **Train Model**; open *Forge → Microscope*, pick a probe point, click neurons/layers, and try *Disable* or edit a weight to compare predictions before/after
3. **Select a model type** (ANN, CNN, RNN, LSTM, GAN, Transformer, Diffuser) and configure input nodes, output nodes, hidden layers, neurons, and activations in the sidebar
4. **Click "Build Network"** to generate the architecture graph
5. **Explore tabs** — switch between Architecture, Forward/Backward Propagation, Weights, Activations, Pruning, and more
6. **Toggle 2D/3D** to view the network in an interactive Three.js scene
7. **Configure training parameters** (dataset, noise, learning rate, batch size, epochs) and click **Train Model** (ANN) or **Simulate Training** (other types)
8. **View results** — training curves, decision boundaries, and loss landscapes
9. **Try Live Training** — train in-browser with TensorFlow.js and watch loss/accuracy update in real-time
10. **Launch Cinema Mode** for a narrated, auto-playing walkthrough of forward propagation
11. **Export code** — generate PyTorch or Keras code, or open the **Template** tab to write and preview your own code scaffold with `{{variables}}`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/build-network` | Build network graph from config |
| `GET` | `/api/network-graph` | Get current network graph |
| `POST` | `/api/forward-propagation` | Compute forward propagation steps |
| `POST` | `/api/backward-propagation` | Compute backward propagation steps |
| `POST` | `/api/decision-boundary` | Compute decision boundary |
| `POST` | `/api/loss-landscape` | Compute loss landscape surface |
| `POST` | `/api/simulate-training` | Synthetic training curve (legacy, not a real training run) |
| `POST` | `/api/dataset` | Generate dataset preview |
| `GET` | `/api/forge/capabilities` | What the introspection layer supports |
| `POST` | `/api/forge/sessions` | Create a real MLP session (untrained) |
| `POST` | `/api/forge/sessions/{id}/train` | Real training; appends history + checkpoints |
| `POST` | `/api/forge/sessions/{id}/graph` | Real graph for a probe / interventions / checkpoint |
| `POST` | `/api/forge/sessions/{id}/inspect` | Neuron / layer / connection inspection |
| `POST` | `/api/forge/sessions/{id}/compare` | Before vs after interventions |

Legacy endpoints `build-network`, `forward-propagation`, `backward-propagation` and `simulate-training` produce **illustrative / synthetic** data and are kept for the non-ANN diagrams. Forge request/response schemas are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and at `http://localhost:8000/docs`.

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.
