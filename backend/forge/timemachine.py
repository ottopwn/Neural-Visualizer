"""Training Time Machine: real history of a session's learning process.

Everything here reads *stored* checkpoints (immutable snapshots of the
parameters, see ``checkpoints``) or statistics logged during real training
(``ModelSession.history``).  Nothing is interpolated: an epoch without a
stored checkpoint can only be described by its logged loss / accuracy /
gradient statistics, and the API says so by only accepting stored epochs.

What-if interventions are deliberately *not* applied here.  The time
machine shows history as it happened; the Microscope endpoints remain the
place where a temporary overlay is applied on top of whichever weights
(live or historical) are being viewed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import schema as S
from .introspect import (BOUNDARY_RESOLUTION, INACTIVE_EPS, IntrospectionError, MLPIntrospector,
                         _f, _floats, _matrix)
from .mlp import ForwardTrace, Params, forward
from .session import ModelSession

SATURATING = {"Sigmoid", "Tanh"}
SATURATION_MARGIN = 0.01  # within 1% of an asymptote
ACC_THRESHOLD = 0.9
TOP_NEURONS = 3


def _norm(t: torch.Tensor) -> float:
    return _f(t.detach().float().norm())


class DatasetEval:
    """One no-grad forward pass of a parameter set over the whole dataset."""

    def __init__(self, session: ModelSession, params: Params) -> None:
        with torch.no_grad():
            self.trace: ForwardTrace = forward(session.spec, params, session.X)
            self.probs = self.trace.probabilities
            self.preds = self.trace.logits.argmax(-1)
            self.loss = _f(F.cross_entropy(self.trace.logits, session.y))
            self.correct = self.preds == session.y
            self.accuracy = _f(self.correct.float().mean())
            self.confidence = self.probs.max(-1).values
            self.p_true = self.probs.gather(1, session.y.unsqueeze(1)).squeeze(1)


class TimeMachine:
    def __init__(self, session: ModelSession) -> None:
        self.session = session
        self.spec = session.spec
        self.intro = MLPIntrospector(session)

    # ── helpers ──────────────────────────────────────────────────────────
    def _params(self, epoch: Optional[int]) -> Tuple[Params, int]:
        """Stored parameters for a checkpoint epoch (``None`` = live)."""
        return self.intro._params(epoch)

    def _boundary(self, params: Params) -> Optional[S.ResponseMap]:
        grid = self.intro._grid(BOUNDARY_RESOLUTION) if self.spec.n_classes == 2 else None
        if grid is None:
            return None
        pts, xr, yr = grid
        r = BOUNDARY_RESOLUTION
        with torch.no_grad():
            p1 = forward(self.spec, params, pts).probabilities[:, 1].reshape(r, r)
        return S.ResponseMap(label="P(Class 1)", x_range=xr, y_range=yr, resolution=r,
                             values=_matrix(p1), value_range=[_f(p1.min()), _f(p1.max())])

    @staticmethod
    def _flip_fraction(a: Optional[S.ResponseMap], b: Optional[S.ResponseMap]) -> Optional[float]:
        """Fraction of the input-plane grid whose predicted class differs."""
        if a is None or b is None:
            return None
        ta, tb = torch.tensor(a.values) > 0.5, torch.tensor(b.values) > 0.5
        return _f((ta != tb).float().mean())

    def _history_row(self, epoch: int) -> Optional[Dict]:
        for row in self.session.history:
            if row["epoch"] == epoch:
                return row
        return None

    def _clean_probe(self, params: Params, probe: S.Probe) -> Tuple[ForwardTrace, Params, torch.Tensor, S.ResolvedProbe]:
        return self.intro.probe_pass(self.intro.compile([], params), probe)

    # ── timeline ─────────────────────────────────────────────────────────
    def layer_health(self, params: Params, data: DatasetEval, row: Optional[Dict]) -> List[S.LayerHealth]:
        out = []
        for k, p in enumerate(params):
            layer = k + 1
            hidden = layer < self.spec.output_layer
            act = self.spec.activation_of(layer)
            w_norm, b_norm = _norm(p.weight), _norm(p.bias)
            p_norm = (w_norm ** 2 + b_norm ** 2) ** 0.5
            grad = upd = None
            if row is not None and row.get("layer_grad_norms") is not None:
                grad = row["layer_grad_norms"][k]
                upd = row["layer_update_norms"][k]
            kw = {}
            if hidden:
                a = data.trace.post[k]
                zero = a.abs() < INACTIVE_EPS
                kw.update(
                    mean_abs_activation=_f(a.abs().mean()),
                    zero_fraction=_f(zero.float().mean()),
                    dead_fraction=_f(zero.all(0).float().mean()),
                )
                if act == "Sigmoid":
                    kw["saturated_fraction"] = _f(((a < SATURATION_MARGIN) | (a > 1 - SATURATION_MARGIN)).float().mean())
                elif act == "Tanh":
                    kw["saturated_fraction"] = _f((a.abs() > 1 - SATURATION_MARGIN).float().mean())
            out.append(S.LayerHealth(
                layer=layer, label=self.intro.layer_label(layer), activation=act,
                weight_norm=w_norm, bias_norm=b_norm, param_norm=p_norm,
                train_grad_norm=grad, update_norm=upd,
                update_ratio=(upd / p_norm) if upd is not None and p_norm > 0 else None,
                **kw,
            ))
        return out

    def _events(self) -> List[S.TrainingEvent]:
        hist = self.session.history
        store = self.session.checkpoints
        events: List[S.TrainingEvent] = []

        def add(epoch: int, kind: str, label: str) -> None:
            events.append(S.TrainingEvent(epoch=epoch, checkpoint_epoch=store.nearest(epoch),
                                          kind=kind, label=label))

        add(0, "init", "Initialisation (random weights)")
        for i, run in enumerate(self.session.runs):
            add(run["start_epoch"], "run",
                f"Run {i + 1}: {run['end_epoch'] - run['start_epoch']} epochs · lr {run['learning_rate']:g} · "
                f"batch {run['batch_size']}" + (f" · {run['reg_type']} {run['reg_rate']:g}" if run["reg_type"] != "None" and run["reg_rate"] > 0 else ""))
        trained = [r for r in hist if r["epoch"] > 0]
        if not trained:
            return events
        first_good = next((r for r in hist if r["accuracy"] >= ACC_THRESHOLD), None)
        if first_good is not None and first_good["epoch"] > 0:
            add(first_good["epoch"], "acc_threshold", f"Accuracy first ≥ {ACC_THRESHOLD:.0%}")
        best = max(trained, key=lambda r: (r["accuracy"], -r["epoch"]))
        add(best["epoch"], "best_accuracy", f"Best accuracy {best['accuracy']:.1%}")
        low = min(trained, key=lambda r: (r["loss"], r["epoch"]))
        add(low["epoch"], "min_loss", f"Lowest loss {low['loss']:.4f}")
        drops = [(hist[i - 1]["loss"] - hist[i]["loss"], hist[i]["epoch"]) for i in range(1, len(hist))]
        drop, epoch = max(drops)
        if drop > 0:
            add(epoch, "largest_drop", f"Largest one-epoch loss drop (−{drop:.4f})")
        return events

    def timeline(self) -> S.Timeline:
        s = self.session
        checkpoints = []
        for c in s.checkpoints.items:
            data = DatasetEval(s, c.params)
            checkpoints.append(S.CheckpointHealth(
                epoch=c.epoch, loss=data.loss, accuracy=data.accuracy,
                layers=self.layer_health(c.params, data, self._history_row(c.epoch)),
            ))
        counts = torch.bincount(s.y, minlength=self.spec.n_classes)
        return S.Timeline(
            live_epoch=s.epoch,
            capacity=s.checkpoints.capacity,
            history=[S.TimelineRow(**r) for r in s.history],
            checkpoints=checkpoints,
            runs=[S.TrainingRun(**r) for r in s.runs],
            events=self._events(),
            majority_rate=_f(counts.max().float() / counts.sum()),
        )

    # ── one frame ────────────────────────────────────────────────────────
    def frame(self, req: S.FrameRequest) -> S.Frame:
        params, epoch = self._params(req.checkpoint_epoch)
        data = DatasetEval(self.session, params)
        trace, _, _, probe = self._clean_probe(params, req.probe)
        probs = trace.probabilities[0].detach()
        boundary = self._boundary(params)

        previous = None
        prev = self.session.checkpoints.previous(epoch)
        if prev is not None:
            pd = DatasetEval(self.session, prev.params)
            previous = S.FramePrevious(
                epoch=prev.epoch, loss=pd.loss, accuracy=pd.accuracy,
                changed=int((pd.preds != data.preds).sum()),
                fixed=int((~pd.correct & data.correct).sum()),
                broken=int((pd.correct & ~data.correct).sum()),
                boundary_flip_fraction=self._flip_fraction(self._boundary(prev.params), boundary),
            )
        return S.Frame(
            epoch=epoch, is_latest=epoch == self.session.epoch,
            loss=data.loss, accuracy=data.accuracy,
            probe=probe, probe_probabilities=_floats(probs), probe_predicted=int(probs.argmax()),
            predictions=[int(v) for v in data.preds.tolist()],
            confidence=_floats(data.confidence),
            boundary=boundary, previous=previous,
        )

    # ── one component across time ────────────────────────────────────────
    def _validate_ref(self, ref: S.ComponentRef) -> None:
        sizes = self.spec.layer_sizes
        if not 0 <= ref.layer < len(sizes):
            raise IntrospectionError(f"layer {ref.layer} out of range 0..{len(sizes) - 1}")
        if ref.kind == "neuron" and (ref.index is None or not 0 <= ref.index < sizes[ref.layer]):
            raise IntrospectionError("neuron index out of range")
        if ref.kind == "connection":
            if ref.layer == 0:
                raise IntrospectionError("connections start at layer 1")
            if (ref.source is None or ref.target is None
                    or not 0 <= ref.source < sizes[ref.layer - 1] or not 0 <= ref.target < sizes[ref.layer]):
                raise IntrospectionError("connection endpoints out of range")

    def ref_name(self, ref: S.ComponentRef) -> str:
        if ref.kind == "neuron":
            return self.intro.neuron_name(ref.layer, ref.index)
        if ref.kind == "layer":
            return self.intro.layer_label(ref.layer)
        return f"{self.intro.neuron_name(ref.layer - 1, ref.source)} → {self.intro.neuron_name(ref.layer, ref.target)}"

    def component_metrics(
        self, params: Params, epoch: int, ref: S.ComponentRef, probe: S.Probe,
    ) -> List[Tuple[str, str, str, Optional[float]]]:
        """(key, label, group, value) rows describing one component at one epoch.

        Shared by the history and the A/B comparison, so both always agree.
        """
        trace, gparams, x, _ = self._clean_probe(params, probe)
        data = DatasetEval(self.session, params)
        layer, out_layer = ref.layer, self.spec.output_layer
        rows: List[Tuple[str, str, str, Optional[float]]] = []

        if ref.kind == "neuron":
            i = ref.index
            if layer == 0:
                rows.append(("grad_input", "dL/dx on probe", "gradient", _f(x.grad[0, i])))
            else:
                k = layer - 1
                w = params[k].weight[i]
                a_all = data.trace.post[k][:, i]
                rows += [
                    ("bias", "bias b", "parameter", _f(params[k].bias[i])),
                    ("w_in_mean", "mean incoming weight", "parameter", _f(w.mean())),
                    ("w_in_norm", "‖incoming weights‖", "parameter", _norm(w)),
                    ("z_probe", "z on probe", "probe", _f(trace.pre[k][0, i])),
                    ("a_probe", "P(class) on probe" if layer == out_layer else "activation on probe", "probe",
                     _f(trace.post[k][0, i])),
                    ("act_mean", "mean output (dataset)", "dataset", _f(a_all.mean())),
                    ("act_std", "std of output (dataset)", "dataset", _f(a_all.std(unbiased=False))),
                ]
                if layer < out_layer:
                    rows.append(("active_frac", "active on dataset", "dataset",
                                 _f((a_all.abs() >= INACTIVE_EPS).float().mean())))
                rows.append(("grad_bias", "dL/db on probe", "gradient", _f(gparams[k].bias.grad[i])))
            if layer < out_layer:
                rows.insert(0 if layer == 0 else 3, ("w_out_norm", "‖outgoing weights‖", "parameter",
                                                     _norm(params[layer].weight[:, i])))
        elif ref.kind == "connection":
            k, s_, t = layer - 1, ref.source, ref.target
            w = params[k].weight[t, s_]
            src = data.trace.inputs if layer == 1 else data.trace.post[k - 1]
            src_probe = trace.activations_of(layer - 1)[0, s_]
            rows += [
                ("weight", "weight w", "parameter", _f(w)),
                ("contribution", "w·a on probe", "probe", _f(w * src_probe)),
                ("contrib_mean", "mean w·a (dataset)", "dataset", _f((w * src[:, s_]).mean())),
                ("grad_w", "dL/dw on probe", "gradient", _f(gparams[k].weight.grad[t, s_])),
            ]
        else:  # layer
            if layer > 0:
                k = layer - 1
                p = params[k]
                rows += [
                    ("w_norm", "‖W‖", "parameter", _norm(p.weight)),
                    ("w_mean", "mean weight", "parameter", _f(p.weight.mean())),
                    ("w_std", "weight std", "parameter", _f(p.weight.std(unbiased=False))),
                    ("b_norm", "‖b‖", "parameter", _norm(p.bias)),
                ]
                if layer < out_layer:
                    a = data.trace.post[k]
                    zero = a.abs() < INACTIVE_EPS
                    rows += [
                        ("act_mean_abs", "mean |a| (dataset)", "dataset", _f(a.abs().mean())),
                        ("dead_frac", "dead neurons", "dataset", _f(zero.all(0).float().mean())),
                    ]
                rows.append(("grad_w_norm", "‖dL/dW‖ on probe", "gradient", _norm(gparams[k].weight.grad)))
                row = self._history_row(epoch)
                if row is not None and row.get("layer_grad_norms") is not None:
                    rows.append(("train_grad_norm", "training ‖grad‖ (epoch mean)", "gradient",
                                 float(row["layer_grad_norms"][k])))
        return rows

    def component_history(self, req: S.ComponentHistoryRequest) -> S.ComponentHistory:
        self._validate_ref(req.ref)
        epochs: List[int] = []
        table: Dict[str, Tuple[str, str, List[Optional[float]]]] = {}
        order: List[str] = []
        items = self.session.checkpoints.items
        for n, c in enumerate(items):
            epochs.append(c.epoch)
            for key, label, group, value in self.component_metrics(c.params, c.epoch, req.ref, req.probe):
                if key not in table:
                    table[key] = (label, group, [None] * n)
                    order.append(key)
                table[key][2].append(value)
            for key in order:  # a metric missing at this epoch (e.g. no training log yet)
                if len(table[key][2]) < n + 1:
                    table[key][2].append(None)
        notes = []
        if req.ref.kind == "layer" and req.ref.layer == 0:
            notes.append("The input layer has no parameters; select a hidden layer, neuron or connection.")
        return S.ComponentHistory(
            ref=req.ref, name=self.ref_name(req.ref), epochs=epochs,
            series=[S.Series(key=k, label=table[k][0], group=table[k][1], values=table[k][2]) for k in order],
            notes=notes,
        )

    # ── A vs B ───────────────────────────────────────────────────────────
    def _epoch_metrics(self, params: Params, epoch: int, data: DatasetEval, probe: S.Probe) -> Tuple[S.EpochMetrics, S.ResolvedProbe]:
        trace, _, _, resolved = self._clean_probe(params, probe)
        probs = trace.probabilities[0].detach()
        return S.EpochMetrics(
            epoch=epoch, loss=data.loss, accuracy=data.accuracy,
            mean_confidence=_f(data.confidence.mean()),
            probe_probabilities=_floats(probs), probe_predicted=int(probs.argmax()),
        ), resolved

    def _param_changes(self, pa: Params, pb: Params) -> Tuple[List[S.ParamChange], float, float]:
        out = []
        total_sq = base_sq = 0.0
        for k, (a, b) in enumerate(zip(pa, pb)):
            dw, db = b.weight - a.weight, b.bias - a.bias
            per_neuron = (dw.pow(2).sum(1) + db.pow(2)).sqrt()
            top = torch.argsort(per_neuron, descending=True)[:TOP_NEURONS]
            delta_sq = _f(dw.pow(2).sum() + db.pow(2).sum())
            a_sq = _f(a.weight.pow(2).sum() + a.bias.pow(2).sum())
            total_sq += delta_sq
            base_sq += a_sq
            out.append(S.ParamChange(
                layer=k + 1, label=self.intro.layer_label(k + 1),
                weight_norm_a=_norm(a.weight), weight_norm_b=_norm(b.weight),
                bias_norm_a=_norm(a.bias), bias_norm_b=_norm(b.bias),
                weight_delta_norm=_norm(dw), bias_delta_norm=_norm(db),
                relative_change=(delta_sq ** 0.5) / (a_sq ** 0.5) if a_sq > 0 else 0.0,
                mean_abs_weight_delta=_f(dw.abs().mean()), max_abs_weight_delta=_f(dw.abs().max()),
                top_neurons=[int(i) for i in top.tolist()],
                top_neuron_deltas=[_f(per_neuron[i]) for i in top.tolist()],
            ))
        total = total_sq ** 0.5
        return out, total, total / (base_sq ** 0.5) if base_sq > 0 else 0.0

    def _component_compare(self, pa: Params, ea: int, pb: Params, eb: int,
                           ref: S.ComponentRef, probe: S.Probe) -> S.ComponentCompare:
        self._validate_ref(ref)
        ma = self.component_metrics(pa, ea, ref, probe)
        mb = {key: value for key, _, _, value in self.component_metrics(pb, eb, ref, probe)}
        rows = []
        for key, label, group, va in ma:
            vb = mb.get(key)
            rows.append(S.ComponentDeltaRow(
                key=key, label=label, group=group, a=va, b=vb,
                delta=(vb - va) if va is not None and vb is not None else None,
            ))
        response_a = response_b = None
        if ref.kind == "neuron" and ref.layer > 0:
            response_a = self.intro.response_map(self.intro.compile([], pa), ref.layer, ref.index)
            response_b = self.intro.response_map(self.intro.compile([], pb), ref.layer, ref.index)
        return S.ComponentCompare(ref=ref, name=self.ref_name(ref), rows=rows,
                                  response_a=response_a, response_b=response_b)

    def compare(self, req: S.EpochCompareRequest) -> S.EpochComparison:
        pa, ea = self._params(req.epoch_a)
        pb, eb = self._params(req.epoch_b)
        da, db = DatasetEval(self.session, pa), DatasetEval(self.session, pb)
        ma, probe = self._epoch_metrics(pa, ea, da, req.probe)
        mb, _ = self._epoch_metrics(pb, eb, db, req.probe)
        changed = da.preds != db.preds
        n = int(changed.numel())
        ba, bb = self._boundary(pa), self._boundary(pb)
        layers, total, rel = self._param_changes(pa, pb)
        return S.EpochComparison(
            a=ma, b=mb, probe=probe,
            loss_delta=mb.loss - ma.loss, accuracy_delta=mb.accuracy - ma.accuracy,
            changed=int(changed.sum()), changed_fraction=int(changed.sum()) / n if n else 0.0,
            fixed=int((~da.correct & db.correct).sum()), broken=int((da.correct & ~db.correct).sum()),
            changed_indices=[int(i) for i in torch.nonzero(changed).flatten().tolist()],
            predictions_a=[int(v) for v in da.preds.tolist()],
            predictions_b=[int(v) for v in db.preds.tolist()],
            confidence_delta=_floats(db.p_true - da.p_true),
            boundary_a=ba, boundary_b=bb, boundary_flip_fraction=self._flip_fraction(ba, bb),
            layers=layers, total_delta_norm=total, total_relative_change=rel,
            component=self._component_compare(pa, ea, pb, eb, req.ref, req.probe) if req.ref else None,
        )

