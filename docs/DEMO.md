# Neural Forge — a 3–5 minute demo

A script for showing Neural Forge to someone (portfolio review, interview, talk). Everything below runs on the real
model; the numbers you see will be close to the ones quoted (training is deterministic for a given setup).

**Before you start:** `./start.sh`, open http://127.0.0.1:5173, use a 1440 px or wider window (1280 works with the
Experiment panel collapsed). The fastest path is **Present** in the top bar — it performs every step below and
narrates the real numbers. The manual path follows.

## 0:00 — The claim (15 s)
> "This is a laboratory for looking inside a real neural network. Everything on screen is computed by a PyTorch model
> running on this laptop — nothing is animated for effect. The green REAL badges say which weights each number came from."

## 0:15 — Build and train (40 s)
1. Experiment panel: ANN, 2 hidden layers × 8 ReLU, Circle, learning rate 0.01, 30 epochs (the demo defaults).
2. **Build** → the Network view shows the untrained model; edge thickness is the real `w·a` on one input.
3. **Train 30** → accuracy ≈ 50 % → 100 %. Every epoch was stored as a checkpoint.

## 0:55 — Rewind with the Time Machine (40 s)
1. **Time Machine** → press **Home**, then **Space** to play. The decision regions are recomputed from each stored
   checkpoint; yellow rings mark mistakes.
2. Pause around epoch 5 (≈ 60 % accuracy): *"this is the network as it really was at epoch 5 — not an interpolation."*
3. Click a neuron in the network pane: the Microscope (right) shows its epoch-5 weights, `z`, activation and gradients.
   **Back to live**: the same neuron, now trained.

## 1:35 — Break it (45 s)
1. **Network** → open **Demos → Break the network** (or select the neuron yourself). Neural Forge disables each hidden
   neuron in turn and keeps the one that hurts most.
2. The Inspector shows **before → after**: e.g. dataset accuracy 100 % → 54 %, 46 % of predictions flipped, decision
   regions before/after. The disabled neuron is crossed out in the graph and ringed in 3D.
3. **Undo**: *"the stored weights were never touched — what-if is an overlay on the forward pass."*

## 2:20 — Follow one input forward (40 s)
1. **Forward / Backward** → *Forward pass* → **Play** (or ← / →).
2. On *Dense 1 · z*: the table lists every `w × a` term for the focused neuron, sorted, with `Σ w·a + b = z`.
3. Continue through the activation (every neuron's `(z, a)` on the ReLU curve), logits, softmax and prediction.

## 3:00 — …and the gradients back (40 s)
1. Switch to *Backward pass*: loss `−log p(target)`, then `dL/dlogits = p − y`.
2. *Output · dW*: `dL/dW = δ ⊗ a`, the update `−η·dL/dw` and the **real one-step SGD preview** (loss before → after).
3. *Dense 2 · δ*: several ReLU neurons typically have slope 0 for this input — *"no gradient flows through a neuron that is off."*
4. Optional: **compare with epoch 2** — the same pass at an earlier checkpoint, with the differences per neuron.

## 3:40 — The real network in 3D (30 s)
1. **3D**: spheres are neurons coloured by their real activation; lines carry `w·a`. Orbit, zoom, click a neuron — the
   Microscope follows. Switch to *Backward pass* and step: pulses travel along the strongest real `w·δ` terms.
2. Point out the disclosure: *"if not every connection is drawn, it says so."*

## 4:10 — Bonus: inside a (tiny) Transformer (40 s)
1. **Transformer Lab** → example *"the cat sat on the mat . then it"*.
2. Attention: pick *Layer 1 · Head 0*, click the row "it": which earlier words it reads from, and the exact
   `q·k × 1/√16 → softmax` computation for one cell. The upper triangle is masked (no peeking at the future).
3. *Next token* → turn a head **off**: the next-word probabilities change for real.
4. Close with honesty: *"a 28k-parameter toy trained here in seconds — not ChatGPT or Claude — but every number is real."*

## If something goes wrong
* "Backend unreachable" → start the backend (`./start.sh`); sessions are in memory, so click **Build** again after a restart.
* The Transformer Lab trains on first use (≈ 5–10 s once per backend start).
