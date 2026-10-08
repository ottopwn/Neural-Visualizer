"""Model introspection: turns a live network into ``schema`` payloads.

``MLPIntrospector`` is the only place that knows both PyTorch internals and
the wire contract.  Every number it emits comes from an actual forward /
backward pass over the session parameters (optionally a checkpoint, with
the intervention overlay applied).  Nothing is synthesised for display.

Gradient convention: ``loss = cross_entropy(logits(probe), target)`` where
``target`` is the probe's true label when it is a dataset row, the user's
choice when given, otherwise the network's own predicted class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from compute import get_forward_propagation_steps

from . import schema as S
from .interventions import CompiledInterventions, Intervention, compile_interventions
from .mlp import ForwardTrace, MLPSpec, Params, activation_fn, clone_params, forward, param_count
from .session import ModelSession

INACTIVE_EPS = 1e-6
MAP_RESOLUTION = 40
BOUNDARY_RESOLUTION = 48
HIST_BINS = 24


class IntrospectionError(ValueError):
    """Bad component reference or probe (maps to HTTP 422)."""


# ── small numeric helpers ───────────────────────────────────────────────────
def _f(v) -> float:
    """Python float from a number or a (possibly grad-tracking) scalar tensor."""
    return float(v.detach()) if isinstance(v, torch.Tensor) else float(v)


def tensor_stats(t: torch.Tensor) -> S.TensorStats:
    t = t.detach().float()
    flat = t.flatten()
    n = flat.numel()
    if n == 0:
        return S.TensorStats(shape=list(t.shape), count=0, mean=0, std=0, min=0, max=0,
                             abs_mean=0, l2_norm=0, frac_zero=0)
    return S.TensorStats(
        shape=list(t.shape),
        count=n,
        mean=_f(flat.mean()),
        std=_f(flat.std(unbiased=False)),
        min=_f(flat.min()),
        max=_f(flat.max()),
        abs_mean=_f(flat.abs().mean()),
        l2_norm=_f(flat.norm()),
        frac_zero=_f((flat.abs() < INACTIVE_EPS).float().mean()),
    )


def histogram(t: torch.Tensor, bins: int = HIST_BINS) -> S.Histogram:
    arr = t.detach().float().flatten().numpy()
    lo, hi = _f(arr.min()), _f(arr.max())
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    counts, edges = np.histogram(arr, bins=bins, range=(lo, hi))
    return S.Histogram(edges=[_f(e) for e in edges], counts=[int(c) for c in counts])


def _floats(t: torch.Tensor) -> List[float]:
    return [_f(v) for v in t.detach().flatten().tolist()]


def _matrix(t: torch.Tensor) -> List[List[float]]:
    return [[_f(v) for v in row] for row in t.detach().tolist()]


@dataclass
class ProbeRun:
    """Result of one instrumented forward + backward pass on the probe."""

    probe: S.ResolvedProbe
    trace: ForwardTrace  # batch of 1, with .grad populated on z / a
    params: Params  # effective parameters (interventions applied), with .grad
    base_params: Params  # parameters before interventions
    compiled: CompiledInterventions
    x: torch.Tensor  # [1, D] with .grad (input saliency)
    provenance: S.Provenance


class MLPIntrospector:
    def __init__(self, session: ModelSession) -> None:
        self.session = session
        self.spec: MLPSpec = session.spec

    # ── naming / structure ───────────────────────────────────────────────
    @property
    def class_names(self) -> List[str]:
        return [f"Class {c}" for c in range(self.spec.n_classes)]

    @property
    def feature_names(self) -> List[str]:
        return [f"x{i + 1}" for i in range(self.spec.input_dim)]

    def role(self, layer: int) -> str:
        if layer == 0:
            return "input"
        return "output" if layer == self.spec.output_layer else "hidden"

    def layer_label(self, layer: int) -> str:
        role = self.role(layer)
        return "Input" if role == "input" else "Output" if role == "output" else f"Dense {layer}"

    def neuron_name(self, layer: int, index: int) -> str:
        role = self.role(layer)
        if role == "input":
            return self.feature_names[index]
        if role == "output":
            return f"C{index}"
        return f"L{layer}N{index + 1}"

    def structure(self) -> S.ModelStructure:
        sizes = self.spec.layer_sizes
        layers = []
        for layer, size in enumerate(sizes):
            fan_in = sizes[layer - 1] if layer > 0 else None
            layers.append(S.LayerInfo(
                layer=layer,
                role=self.role(layer),
                label=self.layer_label(layer),
                size=size,
                activation=self.spec.activation_of(layer),
                fan_in=fan_in,
                param_count=(fan_in * size + size) if fan_in else 0,
                weight_shape=[size, fan_in] if fan_in else None,
                bias_shape=[size] if fan_in else None,
            ))
        return S.ModelStructure(
            input_dim=self.spec.input_dim,
            n_classes=self.spec.n_classes,
            class_names=self.class_names,
            feature_names=self.feature_names,
            layers=layers,
            param_count=param_count(self.spec),
        )

    # ── core instrumented pass ───────────────────────────────────────────
    def _params(self, checkpoint_epoch: Optional[int]) -> Tuple[Params, int]:
        try:
            params = self.session.params_at(checkpoint_epoch)
        except KeyError as exc:
            raise IntrospectionError(str(exc)) from exc
        epoch = self.session.epoch if checkpoint_epoch is None else checkpoint_epoch
        return params, epoch

    def _provenance(self, epoch: int, n_interventions: int) -> S.Provenance:
        return S.Provenance(
            checkpoint_epoch=epoch,
            is_latest=epoch == self.session.epoch,
            interventions_applied=n_interventions,
        )

    def _probe_x(self, probe: S.Probe) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        X, y = self.session.X, self.session.y
        if probe.sample_index is not None:
            if not 0 <= probe.sample_index < X.shape[0]:
                raise IntrospectionError(f"sample_index {probe.sample_index} out of range")
            return X[probe.sample_index].clone(), probe.sample_index, int(y[probe.sample_index])
        if probe.x is None:
            raise IntrospectionError("probe needs x or sample_index")
        if len(probe.x) != self.spec.input_dim:
            raise IntrospectionError(f"probe.x must have {self.spec.input_dim} features")
        x = torch.tensor(probe.x, dtype=torch.float32)
        if not torch.isfinite(x).all():
            raise IntrospectionError("probe.x must be finite")
        return x, None, None

    def compile(self, interventions: Sequence[Intervention], params: Params) -> CompiledInterventions:
        try:
            return compile_interventions(self.spec, params, interventions)
        except ValueError as exc:
            raise IntrospectionError(str(exc)) from exc

    def run_probe(self, req: S.ExperimentRequest) -> ProbeRun:
        base, epoch = self._params(req.checkpoint_epoch)
        compiled = self.compile(req.interventions, base)
        trace, params, x, resolved = self.probe_pass(compiled, req.probe)
        return ProbeRun(resolved, trace, params, base, compiled, x,
                        self._provenance(epoch, len(req.interventions)))

    def probe_pass(
        self, compiled: CompiledInterventions, probe: S.Probe,
    ) -> Tuple[ForwardTrace, Params, torch.Tensor, S.ResolvedProbe]:
        """One instrumented forward + backward pass of ``compiled`` on the probe.

        Works on clones, so the parameters behind ``compiled`` (live weights
        or a stored checkpoint) are never touched.
        """
        x_vec, sample_index, label = self._probe_x(probe)
        params = clone_params(compiled.params, requires_grad=True)
        x = x_vec.unsqueeze(0).requires_grad_(True)
        trace = forward(self.spec, params, x, compiled.neuron_masks, retain_grad=True)

        if probe.target is not None:
            if not 0 <= probe.target < self.spec.n_classes:
                raise IntrospectionError("target out of range")
            target, source = probe.target, "user"
        elif label is not None:
            target, source = label, "label"
        else:
            target, source = int(trace.logits.argmax(-1)), "prediction"

        loss = F.cross_entropy(trace.logits, torch.tensor([target]))
        loss.backward()

        resolved = S.ResolvedProbe(
            x=_floats(x_vec), sample_index=sample_index, label=label,
            target=target, target_source=source,
        )
        return trace, params, x, resolved

    def dataset_trace(self, compiled: CompiledInterventions, X: Optional[torch.Tensor] = None) -> ForwardTrace:
        with torch.no_grad():
            return forward(self.spec, compiled.params, self.session.X if X is None else X, compiled.neuron_masks)

    # ── grid helpers (2-D inputs only) ───────────────────────────────────
    def _grid(self, resolution: int) -> Optional[Tuple[torch.Tensor, List[float], List[float]]]:
        if self.spec.input_dim != 2:
            return None
        X = self.session.X
        lo = X.min(0).values - 0.5
        hi = X.max(0).values + 0.5
        xs = torch.linspace(_f(lo[0]), _f(hi[0]), resolution)
        ys = torch.linspace(_f(lo[1]), _f(hi[1]), resolution)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        pts = torch.stack([gx.flatten(), gy.flatten()], dim=1)
        return pts, [_f(lo[0]), _f(hi[0])], [_f(lo[1]), _f(hi[1])]

    def response_map(self, compiled: CompiledInterventions, layer: int, index: int) -> Optional[S.ResponseMap]:
        if layer == 0:
            return None
        grid = self._grid(MAP_RESOLUTION)
        if grid is None:
            return None
        pts, xr, yr = grid
        values = self.dataset_trace(compiled, pts).activations_of(layer)[:, index]
        values = values.reshape(MAP_RESOLUTION, MAP_RESOLUTION)
        label = f"P({self.class_names[index]})" if self.role(layer) == "output" else "activation"
        return S.ResponseMap(
            label=label, x_range=xr, y_range=yr, resolution=MAP_RESOLUTION,
            values=_matrix(values), value_range=[_f(values.min()), _f(values.max())],
        )

    # ── graph ────────────────────────────────────────────────────────────
    def _output_grad(self, run: ProbeRun, layer: int) -> torch.Tensor:
        """dLoss/d(output) of a graph layer; dLoss/dlogit for the output layer."""
        if layer == 0:
            return run.x.grad[0]
        if layer == self.spec.output_layer:
            return run.trace.pre[layer - 1].grad[0]
        return run.trace.post[layer - 1].grad[0]

    def graph(self, req: S.ExperimentRequest) -> S.ForgeGraph:
        run = self.run_probe(req)
        sizes = self.spec.layer_sizes
        nodes: List[S.GraphNode] = []
        edges: List[S.GraphEdge] = []
        ids: List[List[int]] = []
        nid = 0
        # Every layer spans the same height, so small layers (inputs/outputs)
        # are spread out instead of collapsing in the middle; neuron 0 on top.
        span = (max(sizes) - 1) * 1.5 + 1.5
        for layer, size in enumerate(sizes):
            layer_ids = []
            out = run.trace.activations_of(layer)[0]
            grad_t = self._output_grad(run, layer)
            for i in range(size):
                is_in = layer == 0
                nodes.append(S.GraphNode(
                    id=nid,
                    x=layer * 2.0,
                    y=-((i + 0.5) / size - 0.5) * span,
                    name=self.neuron_name(layer, i),
                    layer=layer,
                    index=i,
                    layer_type=self.role(layer),
                    value=_f(out[i]),
                    z_val=None if is_in else _f(run.trace.pre[layer - 1][0, i]),
                    activation=self.spec.activation_of(layer),
                    bias=None if is_in else _f(run.params[layer - 1].bias[i]),
                    grad=_f(grad_t[i]),
                    ablated=(layer, i) in run.compiled.ablated,
                    edited=(layer, i) in run.compiled.bias_edits,
                ))
                layer_ids.append(nid)
                nid += 1
            ids.append(layer_ids)

        for layer in range(1, len(sizes)):
            p = run.params[layer - 1]
            for t in range(sizes[layer]):
                for s in range(sizes[layer - 1]):
                    edges.append(S.GraphEdge(
                        source=ids[layer - 1][s], target=ids[layer][t],
                        weight=_f(p.weight[t, s]), layer=layer,
                        source_index=s, target_index=t,
                        grad=_f(p.weight.grad[t, s]),
                        edited=(layer, s, t) in run.compiled.weight_edits,
                    ))

        node_dicts = [n.model_dump() for n in nodes]
        edge_dicts = [e.model_dump() for e in edges]
        fwd = get_forward_propagation_steps(node_dicts, edge_dicts)
        # Backward steps: same structure in reverse, with *real* dLoss/d(output) per node.
        bwd = []
        rev = list(reversed(fwd))
        for i, step in enumerate(rev):
            active = set(step["active_nodes"])
            prev_nodes = set(rev[i + 1]["active_nodes"]) if i + 1 < len(rev) else set()
            bwd.append({
                "step": i,
                "label": f"Backprop {step['label']}",
                "active_nodes": step["active_nodes"],
                "active_edges": [j for j, e in enumerate(edge_dicts)
                                 if e["target"] in active and e["source"] in prev_nodes],
                "layer_type": step["layer_type"],
                "gradients": {str(n.id): _f(n.grad or 0.0) for n in nodes if n.id in active},
            })

        return S.ForgeGraph(
            nodes=nodes, edges=edges,
            forward_steps=[S.PropStepOut(**s) for s in fwd],
            backward_steps=[S.PropStepOut(**s) for s in bwd],
            probe=run.probe, provenance=run.provenance,
        )

    # ── inspections ──────────────────────────────────────────────────────
    def inspect(self, req: S.InspectRequest):
        ref = req.ref
        n_layers = len(self.spec.layer_sizes)
        if not 0 <= ref.layer < n_layers:
            raise IntrospectionError(f"layer {ref.layer} out of range 0..{n_layers - 1}")
        if ref.kind == "neuron":
            if ref.index is None or not 0 <= ref.index < self.spec.layer_sizes[ref.layer]:
                raise IntrospectionError("neuron index out of range")
            return self.inspect_neuron(req, ref.layer, ref.index)
        if ref.kind == "layer":
            return self.inspect_layer(req, ref.layer)
        if ref.layer == 0:
            raise IntrospectionError("connections start at layer 1")
        if (ref.source is None or ref.target is None
                or not 0 <= ref.source < self.spec.layer_sizes[ref.layer - 1]
                or not 0 <= ref.target < self.spec.layer_sizes[ref.layer]):
            raise IntrospectionError("connection endpoints out of range")
        return self.inspect_connection(req, ref.layer, ref.source, ref.target)

    def inspect_neuron(self, req: S.ExperimentRequest, layer: int, index: int) -> S.NeuronInspection:
        run = self.run_probe(req)
        trace, role = run.trace, self.role(layer)
        data = self.dataset_trace(run.compiled)
        sizes = self.spec.layer_sizes
        out_all = data.activations_of(layer)[:, index]
        notes: List[str] = []

        kw = {}
        if layer == 0:
            kw.update(value=_f(run.x[0, index]), grad_value=_f(run.x.grad[0, index]))
            notes.append("Input features have no parameters; the gradient is the loss saliency dLoss/dx.")
        else:
            k = layer - 1
            p = run.params[k]
            prev = trace.activations_of(layer - 1)[0]
            w = p.weight[index]
            z = trace.pre[k][0, index]
            kw.update(
                value=_f(trace.activations_of(layer)[0, index]),
                pre_activation=_f(z),
                bias=_f(p.bias[index]),
                input_names=[self.neuron_name(layer - 1, j) for j in range(sizes[layer - 1])],
                inputs=_floats(prev),
                weights=_floats(w),
                contributions=_floats(w * prev),
                grad_value=None if role == "output" else _f(trace.post[k].grad[0, index]),
                grad_pre_activation=_f(trace.pre[k].grad[0, index]),
                grad_bias=_f(p.bias.grad[index]),
                grad_weights=_floats(p.weight.grad[index]),
            )
            kw["shapes"] = {"weights_in": [sizes[layer - 1]], "layer_weight": [sizes[layer], sizes[layer - 1]]}
            if (layer, index) in run.compiled.ablated:
                kw["ablated"] = True
                kw["natural_value"] = _f(activation_fn(self.spec.activations[k])(z))
                notes.append("Ablated: this neuron's output is forced to 0 for everything downstream.")
            if (layer, index) in run.compiled.bias_edits:
                kw["bias_edit"] = S.BiasEdit(value=_f(p.bias[index]), original=_f(run.base_params[k].bias[index]))
            kw["weight_edits"] = [
                S.WeightEdit(source=s, target=t, value=v, original=_f(run.base_params[k].weight[t, s]))
                for (lyr, s, t), v in run.compiled.weight_edits.items() if lyr == layer and t == index
            ]
            if role == "output":
                notes.append("Output value is the softmax probability; the softmax couples all output neurons. "
                             "The loss is computed from the logits, so the gradient shown is dLoss/dlogit.")

        if layer < self.spec.output_layer:
            nxt = run.params[layer].weight[:, index]
            kw.update(
                outgoing_names=[self.neuron_name(layer + 1, j) for j in range(sizes[layer + 1])],
                outgoing_weights=_floats(nxt),
            )

        if self.spec.input_dim != 2 and layer > 0:
            notes.append("Response map is only available for 2-D inputs.")

        return S.NeuronInspection(
            layer=layer, index=index, name=self.neuron_name(layer, index), role=role,
            layer_label=self.layer_label(layer), activation_fn=self.spec.activation_of(layer),
            probe=run.probe, provenance=run.provenance,
            dataset_stats=tensor_stats(out_all), dataset_histogram=histogram(out_all),
            inactive_fraction=_f((out_all.abs() < INACTIVE_EPS).float().mean()),
            response_map=self.response_map(run.compiled, layer, index),
            notes=notes, **kw,
        )

    def inspect_layer(self, req: S.ExperimentRequest, layer: int) -> S.LayerInspection:
        run = self.run_probe(req)
        trace = run.trace
        data = self.dataset_trace(run.compiled)
        sizes = self.spec.layer_sizes
        acts_all = data.activations_of(layer)
        never = (acts_all.abs() < INACTIVE_EPS).all(0)
        kw = {}
        if layer == 0:
            kw.update(grad_activations=_floats(run.x.grad[0]))
        else:
            k = layer - 1
            p = run.params[k]
            kw.update(
                input_names=[self.neuron_name(layer - 1, j) for j in range(sizes[layer - 1])],
                weights=_matrix(p.weight), bias=_floats(p.bias),
                weight_stats=tensor_stats(p.weight), bias_stats=tensor_stats(p.bias),
                weight_histogram=histogram(p.weight),
                pre_activations=_floats(trace.pre[k][0]),
                grad_activations=_floats(self._output_grad(run, layer)),
                grad_weights=_matrix(p.weight.grad),
                grad_weight_norm=_f(p.weight.grad.norm()),
                grad_bias_norm=_f(p.bias.grad.norm()),
                shapes={"weight": [sizes[layer], sizes[layer - 1]], "bias": [sizes[layer]],
                        "activations": [sizes[layer]]},
            )
        fan_in = sizes[layer - 1] if layer > 0 else None
        return S.LayerInspection(
            layer=layer, role=self.role(layer), label=self.layer_label(layer), size=sizes[layer],
            activation_fn=self.spec.activation_of(layer), fan_in=fan_in,
            param_count=(fan_in * sizes[layer] + sizes[layer]) if fan_in else 0,
            probe=run.probe, provenance=run.provenance,
            neuron_names=[self.neuron_name(layer, i) for i in range(sizes[layer])],
            activations=_floats(trace.activations_of(layer)[0]),
            dataset_stats=tensor_stats(acts_all),
            never_active=[int(i) for i in torch.nonzero(never).flatten().tolist()] if layer > 0 else [],
            ablated=sorted(i for (lyr, i) in run.compiled.ablated if lyr == layer),
            **kw,
        )

    def inspect_connection(self, req: S.ExperimentRequest, layer: int, source: int, target: int) -> S.ConnectionInspection:
        run = self.run_probe(req)
        k = layer - 1
        p = run.params[k]
        prev = run.trace.activations_of(layer - 1)[0]
        contribs = p.weight[target] * prev
        contribution = contribs[source]
        abs_total = _f(contribs.abs().sum())
        rank = int((contribs.abs() > contribution.abs()).sum()) + 1
        data = self.dataset_trace(run.compiled)
        data_contrib = p.weight[target, source].detach() * data.activations_of(layer - 1)[:, source]
        edited = (layer, source, target) in run.compiled.weight_edits
        notes = []
        if (layer - 1, source) in run.compiled.ablated:
            notes.append("The source neuron is ablated, so this connection currently carries 0.")
        return S.ConnectionInspection(
            layer=layer, source=source, target=target,
            source_name=self.neuron_name(layer - 1, source), target_name=self.neuron_name(layer, target),
            probe=run.probe, provenance=run.provenance,
            weight=_f(p.weight[target, source]),
            original_weight=_f(run.base_params[k].weight[target, source]) if edited else None,
            source_value=_f(prev[source]),
            contribution=_f(contribution),
            target_pre_activation=_f(run.trace.pre[k][0, target]),
            target_bias=_f(p.bias[target]),
            share_of_input=_f(contribution.abs()) / abs_total if abs_total > 0 else 0.0,
            rank=rank, fan_in=int(prev.numel()),
            grad_weight=_f(p.weight.grad[target, source]),
            dataset_contribution=tensor_stats(data_contrib),
            notes=notes,
        )

    # ── forward / backward pass explorer ─────────────────────────────────
    def trace(self, req: S.TraceRequest) -> S.ComputationTrace:
        """Every tensor of one forward + backward pass on the probe.

        The values come straight from the instrumented autograd pass used by
        the rest of the microscope (``run_probe``); ``local_grad`` (da/dz) is
        obtained by differentiating the activation function itself, with the
        ablation mask applied, so ``grad_z == grad_a * local_grad`` holds.
        """
        run = self.run_probe(req)
        trace, spec = run.trace, self.spec
        sizes = spec.layer_sizes
        layers: List[S.TraceLayer] = []
        for layer in range(1, len(sizes)):
            k = layer - 1
            p = run.params[k]
            z = trace.pre[k][0]
            is_out = layer == spec.output_layer
            kw = {}
            if not is_out:
                zd = z.detach().clone().requires_grad_(True)
                activation_fn(spec.activations[k])(zd).sum().backward()
                local = zd.grad
                mask = run.compiled.neuron_masks[k]
                if mask is not None:
                    local = local * mask
                kw.update(grad_a=_floats(trace.post[k].grad[0]), local_grad=_floats(local))
            layers.append(S.TraceLayer(
                layer=layer, label=self.layer_label(layer), role=self.role(layer),
                activation=spec.activation_of(layer) or "",
                input_names=[self.neuron_name(layer - 1, j) for j in range(sizes[layer - 1])],
                neuron_names=[self.neuron_name(layer, i) for i in range(sizes[layer])],
                weight=_matrix(p.weight), bias=_floats(p.bias),
                input=_floats(trace.activations_of(layer - 1)[0]),
                z=_floats(z), a=_floats(trace.post[k][0]),
                ablated=sorted(i for (lyr, i) in run.compiled.ablated if lyr == layer),
                edited_bias=sorted(i for (lyr, i) in run.compiled.bias_edits if lyr == layer),
                edited_weights=sorted([s, t] for (lyr, s, t) in run.compiled.weight_edits if lyr == layer),
                grad_z=_floats(trace.pre[k].grad[0]),
                grad_weight=_matrix(p.weight.grad), grad_bias=_floats(p.bias.grad),
                grad_input=_floats(self._output_grad(run, layer - 1)),
                **kw,
            ))

        logits = trace.logits[0]
        probs = trace.probabilities[0]
        target = run.probe.target
        loss = _f(F.cross_entropy(trace.logits, torch.tensor([target])))

        preview = None
        if req.learning_rate is not None:
            lr = float(req.learning_rate)
            if not 0 < lr <= 1:
                raise IntrospectionError("learning_rate must be in (0, 1]")
            preview = self._sgd_preview(run, lr, loss, _f(probs[target]))

        return S.ComputationTrace(
            probe=run.probe, provenance=run.provenance,
            feature_names=self.feature_names, class_names=self.class_names,
            input=_floats(run.x[0]), layers=layers,
            logits=_floats(logits), probabilities=_floats(probs),
            predicted_class=int(logits.argmax()), target=target, loss=loss,
            grad_input=_floats(run.x.grad[0]), sgd_preview=preview,
        )

    def _sgd_preview(self, run: ProbeRun, lr: float, loss_before: float, p_before: float) -> S.SgdPreview:
        target = run.probe.target
        with torch.no_grad():
            stepped = clone_params(run.params)
            sq = 0.0
            for new, old in zip(stepped, run.params):
                new.weight -= lr * old.weight.grad
                new.bias -= lr * old.bias.grad
                sq += float((lr * old.weight.grad).pow(2).sum() + (lr * old.bias.grad).pow(2).sum())
            x = run.x.detach()
            after = forward(self.spec, stepped, x, run.compiled.neuron_masks)
            loss_after = _f(F.cross_entropy(after.logits, torch.tensor([target])))
        return S.SgdPreview(
            learning_rate=lr, loss_before=loss_before, loss_after=loss_after,
            target_prob_before=p_before, target_prob_after=_f(after.probabilities[0, target]),
            predicted_after=int(after.logits[0].argmax()), update_norm=sq ** 0.5,
        )

    # ── before / after ───────────────────────────────────────────────────
    def _summary(self, compiled: CompiledInterventions, x: torch.Tensor) -> Tuple[S.PredictionSummary, torch.Tensor]:
        with torch.no_grad():
            probe_trace = forward(self.spec, compiled.params, x.unsqueeze(0), compiled.neuron_masks)
            data = self.dataset_trace(compiled)
            preds = data.logits.argmax(-1)
            summary = S.PredictionSummary(
                logits=_floats(probe_trace.logits[0]),
                probabilities=_floats(probe_trace.probabilities[0]),
                predicted_class=int(probe_trace.logits[0].argmax()),
                dataset_accuracy=_f((preds == self.session.y).float().mean()),
                dataset_loss=_f(F.cross_entropy(data.logits, self.session.y)),
            )
        return summary, preds

    def compare(self, req: S.ExperimentRequest, include_boundary: bool = True) -> S.Comparison:
        base, epoch = self._params(req.checkpoint_epoch)
        baseline_c = self.compile([], base)
        inter_c = self.compile(req.interventions, base)
        x, sample_index, label = self._probe_x(req.probe)
        before, preds_before = self._summary(baseline_c, x)
        after, preds_after = self._summary(inter_c, x)

        if req.probe.target is not None:
            target, source = req.probe.target, "user"
        elif label is not None:
            target, source = label, "label"
        else:
            target, source = before.predicted_class, "prediction"

        boundary = None
        grid = self._grid(BOUNDARY_RESOLUTION) if include_boundary and self.spec.n_classes == 2 else None
        if grid is not None:
            pts, xr, yr = grid
            r = BOUNDARY_RESOLUTION
            b = self.dataset_trace(baseline_c, pts).probabilities[:, 1].reshape(r, r)
            a = self.dataset_trace(inter_c, pts).probabilities[:, 1].reshape(r, r)
            boundary = S.BoundaryPair(x_range=xr, y_range=yr, resolution=r,
                                      baseline=_matrix(b), intervened=_matrix(a))

        return S.Comparison(
            probe=S.ResolvedProbe(x=_floats(x), sample_index=sample_index, label=label,
                                  target=target, target_source=source),
            provenance=self._provenance(epoch, len(req.interventions)),
            baseline=before, intervened=after,
            delta=[a - b for a, b in zip(after.probabilities, before.probabilities)],
            prediction_changed=before.predicted_class != after.predicted_class,
            dataset_flip_fraction=_f((preds_before != preds_after).float().mean()),
            boundary=boundary,
        )
