import type { ReactNode } from 'react';
import { sampleCurve } from '../../../forge/activations';
import { fmt, fmtSigned, maxAbs, pct } from '../../../forge/format';
import {
  backpropTerms, contributionTerms, denseOf, diff, softmaxParts, type PassStep,
} from '../../../forge/passExplorer';
import type { ComputationTrace, TraceLayer } from '../../../forge/types';

// ── primitives ──────────────────────────────────────────────────────────────

function Eq({ children }: { children: ReactNode }) {
  return (
    <div className="font-mono text-[12.5px] px-3 py-2 rounded-md border overflow-x-auto whitespace-nowrap tnum"
      style={{ borderColor: 'var(--border)', background: 'var(--bg-base)', color: 'var(--text-primary)' }}>{children}</div>
  );
}

function Learn({ children }: { children: ReactNode }) {
  return <p className="text-[13px] leading-relaxed" style={{ color: 'var(--text-primary)' }}>{children}</p>;
}

function H({ children, hint }: { children: ReactNode; hint?: ReactNode }) {
  return (
    <div className="flex items-baseline gap-2 mt-1">
      <h4 className="m-0 text-[12.5px] font-semibold" style={{ color: 'var(--text-primary)' }}>{children}</h4>
      {hint && <span className="text-[11.5px]" style={{ color: 'var(--text-faint)' }}>{hint}</span>}
    </div>
  );
}

function Num({ v, signed = false, digits = 4, tone }: { v: number | null | undefined; signed?: boolean; digits?: number; tone?: 'sign' }) {
  const color = tone === 'sign' && v !== null && v !== undefined
    ? (v > 0 ? 'var(--text-pos)' : v < 0 ? 'var(--text-neg)' : 'var(--text-muted)') : undefined;
  return <span className="font-mono tnum" style={{ color }}>{signed ? fmtSigned(v, digits) : fmt(v, digits)}</span>;
}

/** Bar centred at zero, scaled to `max`. */
function Bar({ v, max }: { v: number; max: number }) {
  const w = max > 0 ? (Math.abs(v) / max) * 50 : 0;
  return (
    <span className="relative block h-2.5 w-full" aria-hidden="true">
      <span className="absolute top-0 bottom-0 left-1/2 w-px" style={{ background: 'var(--border-soft)' }} />
      <span className="absolute top-0 bottom-0 rounded-sm" style={{ left: v >= 0 ? '50%' : `${50 - w}%`, width: `${w}%`, background: v >= 0 ? 'var(--pos)' : 'var(--neg)', opacity: 0.85 }} />
    </span>
  );
}

interface Col { label: string; values: (number | null)[]; signed?: boolean; tone?: 'sign' }

/** A table of per-neuron values; one column can carry bars. */
function VecTable({ names, cols, barCol = 0, focus, onPick, flagged, maxRows = 32 }: {
  names: string[]; cols: Col[]; barCol?: number | null; focus?: number; onPick?: (i: number) => void;
  flagged?: Record<number, string>; maxRows?: number;
}) {
  const barValues = barCol !== null ? (cols[barCol].values.filter((v) => v !== null) as number[]) : [];
  const m = maxAbs(barValues);
  const rows = names.map((_, i) => i);
  const shown = rows.length > maxRows ? rows.slice(0, maxRows) : rows;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[12px] tnum border-collapse">
        <thead>
          <tr style={{ color: 'var(--text-faint)' }}>
            <th className="text-left font-medium py-1 pr-2">neuron</th>
            {cols.map((c) => <th key={c.label} className="text-right font-medium py-1 px-2 font-mono">{c.label}</th>)}
            {barCol !== null && <th className="w-[28%] py-1" aria-hidden="true" />}
          </tr>
        </thead>
        <tbody>
          {shown.map((i) => {
            const isFocus = focus === i;
            return (
              <tr key={i} onClick={() => onPick?.(i)} className={onPick ? 'cursor-pointer hover:bg-[var(--bg-hover)]' : ''}
                style={{ background: isFocus ? 'var(--bg-active)' : undefined }}>
                <td className="py-[3px] pr-2 font-mono whitespace-nowrap" style={{ color: isFocus ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                  {names[i]}{flagged?.[i] && <span className="ml-1.5 text-[10.5px]" style={{ color: 'var(--text-warn)' }}>{flagged[i]}</span>}
                </td>
                {cols.map((c) => (
                  <td key={c.label} className="py-[3px] px-2 text-right"><Num v={c.values[i]} signed={c.signed} tone={c.tone} /></td>
                ))}
                {barCol !== null && <td className="py-[3px] pl-2"><Bar v={(cols[barCol].values[i] ?? 0) as number} max={m} /></td>}
              </tr>
            );
          })}
        </tbody>
      </table>
      {rows.length > shown.length && (
        <p className="text-[11px] mt-1" style={{ color: 'var(--text-faint)' }}>Showing {shown.length} of {rows.length} neurons (click a cell in the map to focus any neuron).</p>
      )}
    </div>
  );
}

/** Plot of an elementwise activation with every neuron of the layer as a point. */
function ActivationPlot({ d, focus }: { d: TraceLayer; focus: number }) {
  const span = Math.max(3, ...d.z.map((z) => Math.abs(z) * 1.15));
  const pts = sampleCurve(d.activation, -span, span, 120);
  if (!pts.length) return null;
  const W = 340, Hh = 150, P = 18;
  const ys = pts.map((p) => p[1]).concat(d.a);
  const lo = Math.min(...ys), hi = Math.max(...ys);
  const sx = (v: number) => P + ((v + span) / (2 * span)) * (W - 2 * P);
  const sy = (v: number) => Hh - P - ((v - lo) / (hi - lo || 1)) * (Hh - 2 * P);
  const path = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join('');
  return (
    <svg viewBox={`0 0 ${W} ${Hh}`} className="w-full max-w-[420px]" role="img" aria-label={`${d.activation} with every neuron's (z, a) on this probe`}>
      <line x1={P} x2={W - P} y1={sy(0)} y2={sy(0)} style={{ stroke: 'var(--border-soft)' }} />
      <line x1={sx(0)} x2={sx(0)} y1={P / 2} y2={Hh - P} style={{ stroke: 'var(--border-soft)' }} />
      <path d={path} fill="none" style={{ stroke: 'var(--accent)' }} strokeWidth={2} />
      {d.z.map((z, i) => {
        const ablated = d.ablated.includes(i);
        return (
          <circle key={i} cx={sx(z)} cy={sy(d.a[i])} r={i === focus ? 5 : 3.2}
            style={{ fill: ablated ? 'var(--warn)' : i === focus ? 'var(--select)' : 'var(--text-muted)', stroke: 'var(--bg-card)' }} strokeWidth={1.2}>
            <title>{`${d.neuron_names[i]}: z = ${z.toFixed(4)} → a = ${d.a[i].toFixed(4)}${ablated ? ' (disabled by what-if)' : ''}`}</title>
          </circle>
        );
      })}
      <text x={W - P} y={Hh - 3} fontSize={10} textAnchor="end" style={{ fill: 'var(--text-faint)' }}>z →</text>
      <text x={sx(0) + 4} y={P} fontSize={10} style={{ fill: 'var(--text-faint)' }}>a</text>
    </svg>
  );
}

/** Small numeric matrix (or a summary for big ones). */
function Matrix({ m, rowNames, colNames, focusRow, label }: { m: number[][]; rowNames: string[]; colNames: string[]; focusRow?: number; label: string }) {
  const big = m.length * (m[0]?.length ?? 0) > 96;
  const mx = maxAbs(m.flat()) || 1;
  if (big) {
    return (
      <p className="text-[12px]" style={{ color: 'var(--text-muted)' }}>
        {label} is {m.length}×{m[0].length} — too large to print; the focused row is shown below and the full matrix is in the Microscope (layer view).
      </p>
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="text-[11px] font-mono tnum border-collapse" aria-label={label}>
        <thead>
          <tr><th />{colNames.map((c) => <th key={c} className="px-1.5 py-0.5 font-medium" style={{ color: 'var(--text-faint)' }}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {m.map((row, i) => (
            <tr key={i} style={{ background: i === focusRow ? 'var(--bg-active)' : undefined }}>
              <th className="pr-2 text-left font-medium" style={{ color: i === focusRow ? 'var(--text-primary)' : 'var(--text-faint)' }}>{rowNames[i]}</th>
              {row.map((v, j) => (
                <td key={j} className="px-1.5 py-0.5 text-right"
                  style={{ color: v >= 0 ? 'var(--text-pos)' : 'var(--text-neg)', opacity: 0.45 + 0.55 * Math.min(1, Math.abs(v) / mx) }}>{fmt(v, 3)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── step bodies ─────────────────────────────────────────────────────────────

interface Ctx {
  t: ComputationTrace;
  step: PassStep;
  focus: number;
  lab: boolean;
  onPick: (layer: number, index: number) => void;
  cmp: ComputationTrace | null; // same experiment at another epoch
  cmpEpoch: number | null;
  lr: number;
}

function compareCol(cmp: ComputationTrace | null, cmpEpoch: number | null, pick: (t: ComputationTrace) => number[] | null | undefined, current: number[]): Col[] {
  if (!cmp || cmpEpoch === null) return [];
  const other = pick(cmp);
  const d = diff(other ?? null, current);
  if (!other || !d) return [];
  return [{ label: `@ep ${cmpEpoch}`, values: other }, { label: 'Δ now−then', values: d, signed: true, tone: 'sign' }];
}

function InputStep({ t, lab, cmp, cmpEpoch }: Ctx) {
  const p = t.probe;
  return (
    <>
      <Learn>
        The network receives one point: {t.feature_names.map((f, i) => <span key={f}><b className="font-mono">{f} = {fmt(t.input[i], 3)}</b>{i < t.input.length - 1 ? ', ' : ''}</span>)}.
        {p.sample_index !== null ? <> It is sample #{p.sample_index} of the dataset, labelled <b>{t.class_names[p.label ?? 0]}</b>.</> : <> It is a free point you placed on the map.</>}
      </Learn>
      {lab && <Eq>x ∈ ℝ^{t.input.length}  ·  x = [{t.input.map((v) => fmt(v, 4)).join(', ')}]</Eq>}
      <VecTable names={t.feature_names} cols={[{ label: 'x', values: t.input }, ...compareCol(cmp, cmpEpoch, (c) => c.input, t.input)]} barCol={0} />
    </>
  );
}

function LinearStep({ t, step, focus, lab, onPick, cmp, cmpEpoch }: Ctx) {
  const d = denseOf(t, step.layer)!;
  const c = contributionTerms(d, focus);
  const name = d.neuron_names[focus];
  const out = step.kind === 'logits';
  const top = c.terms[0];
  const editedW = d.edited_weights.filter(([, tg]) => tg === focus).map(([s]) => s);
  const cmpD = cmp ? denseOf(cmp, step.layer) : null;
  return (
    <>
      <Learn>
        {out ? 'Each output neuron' : `Each neuron of ${d.label}`} multiplies every incoming value by its weight, adds them up and adds its bias.
        For <b className="font-mono">{name}</b> the strongest push is from <b className="font-mono">{top.name}</b> ({fmt(top.a, 3)} × {fmt(top.w, 3)} = <Num v={top.product} signed digits={3} tone="sign" />).
        All {c.terms.length} terms sum to <Num v={c.sum} signed digits={3} />; with the bias <Num v={c.bias} signed digits={3} /> the result is
        <b className="font-mono"> z = {fmt(c.z, 4)}</b>{out ? ' — the logit of this class.' : '.'}
      </Learn>
      {lab && (
        <Eq>
          z = W·a + b   W ∈ ℝ^{d.weight.length}×{d.input.length}, a ∈ ℝ^{d.input.length}, b ∈ ℝ^{d.bias.length} → z ∈ ℝ^{d.z.length}
        </Eq>
      )}
      <H hint={`${c.terms.length} inputs · sorted by |w·a|`}>{name}: z = Σ w·a + b</H>
      <div className="overflow-x-auto">
        <table className="w-full text-[12px] tnum border-collapse">
          <thead>
            <tr style={{ color: 'var(--text-faint)' }}>
              <th className="text-left font-medium py-1">input</th><th className="text-right font-medium font-mono">a</th>
              <th className="text-right font-medium font-mono">w</th><th className="text-right font-medium font-mono">w·a</th><th className="w-[30%]" />
            </tr>
          </thead>
          <tbody>
            {c.terms.slice(0, lab ? 24 : 8).map((x) => (
              <tr key={x.index}>
                <td className="py-[3px] font-mono" style={{ color: 'var(--text-muted)' }}>
                  {x.name}{editedW.includes(x.index) && <span className="ml-1 text-[10.5px]" style={{ color: 'var(--text-warn)' }}>edited</span>}
                </td>
                <td className="text-right"><Num v={x.a} /></td>
                <td className="text-right"><Num v={x.w} /></td>
                <td className="text-right"><Num v={x.product} signed tone="sign" /></td>
                <td className="pl-2"><Bar v={x.product} max={Math.abs(top.product)} /></td>
              </tr>
            ))}
          </tbody>
        </table>
        {c.terms.length > (lab ? 24 : 8) && (
          <p className="text-[11px] mt-1" style={{ color: 'var(--text-faint)' }}>
            + {c.terms.length - (lab ? 24 : 8)} smaller terms, together <Num v={c.terms.slice(lab ? 24 : 8).reduce((s, x) => s + x.product, 0)} signed digits={4} />
          </p>
        )}
      </div>
      <Eq>
        Σ w·a = <Num v={c.sum} signed />  +  b = <Num v={c.bias} signed />{d.edited_bias.includes(focus) ? ' (edited)' : ''}  =  z = <b><Num v={c.z} /></b>
        {lab && <span style={{ color: 'var(--text-faint)' }}>   (check: {fmt(c.sum + c.bias - c.z, 2)})</span>}
      </Eq>
      <H hint="every neuron of the layer · click to focus">All pre-activations</H>
      <VecTable names={d.neuron_names} focus={focus} onPick={(i) => onPick(step.layer, i)}
        cols={[{ label: 'z', values: d.z, signed: true }, ...(lab ? [{ label: 'b', values: d.bias, signed: true }] : []),
          ...compareCol(cmp, cmpEpoch, () => cmpD?.z, d.z)]} barCol={0} />
      {lab && <><H hint="rows = neurons of this layer">Weight matrix W</H><Matrix m={d.weight} rowNames={d.neuron_names} colNames={d.input_names} focusRow={focus} label="W" /></>}
    </>
  );
}

function ActivationStep({ t, step, focus, lab, onPick, cmp, cmpEpoch }: Ctx) {
  const d = denseOf(t, step.layer)!;
  const n = d.z.length;
  const zeroes = d.a.filter((v) => Math.abs(v) < 1e-6).length;
  const flagged: Record<number, string> = {};
  d.ablated.forEach((i) => { flagged[i] = 'disabled'; });
  const cmpD = cmp ? denseOf(cmp, step.layer) : null;
  return (
    <>
      <Learn>
        Each neuron passes its sum <span className="font-mono">z</span> through <b>{d.activation}</b>.
        <b className="font-mono"> {d.neuron_names[focus]}</b>: {d.activation}({fmt(d.z[focus], 3)}) = <b className="font-mono">{fmt(d.a[focus], 4)}</b>.
        {d.activation === 'ReLU' || d.activation === 'LeakyReLU'
          ? <> Negative sums are {d.activation === 'ReLU' ? 'cut to 0' : 'scaled by 0.2'}; {zeroes} of {n} neurons output 0 on this input.</>
          : <> Large sums saturate towards the flat ends of the curve.</>}
        {d.ablated.length > 0 && <> {d.ablated.length} neuron{d.ablated.length > 1 ? 's are' : ' is'} disabled by your what-if edits: the output is forced to 0.</>}
      </Learn>
      {lab && <Eq>a = {d.activation}(z){d.ablated.length ? ' ⊙ mask' : ''}   applied elementwise, z, a ∈ ℝ^{n}</Eq>}
      <ActivationPlot d={d} focus={focus} />
      <VecTable names={d.neuron_names} focus={focus} onPick={(i) => onPick(step.layer, i)} flagged={flagged}
        cols={[{ label: 'z', values: d.z, signed: true }, { label: 'a', values: d.a }, ...compareCol(cmp, cmpEpoch, () => cmpD?.a, d.a)]} barCol={1} />
    </>
  );
}

function SoftmaxStep({ t, lab, cmp, cmpEpoch }: Ctx) {
  const sm = softmaxParts(t.logits);
  return (
    <>
      <Learn>
        Softmax turns the logits into probabilities that sum to 1: each class gets e^logit divided by the sum over classes.
        The bigger a logit is relative to the others, the larger its share.
      </Learn>
      {lab && <Eq>p_k = exp(z_k − max z) / Σ_j exp(z_j − max z)   ·   Σ exp = {fmt(sm.total, 5)}</Eq>}
      <VecTable names={t.class_names} cols={[
        { label: 'logit z', values: t.logits, signed: true },
        ...(lab ? [{ label: 'exp(z−max)', values: sm.exps }] : []),
        { label: 'p', values: t.probabilities },
        ...compareCol(cmp, cmpEpoch, (c) => c.probabilities, t.probabilities),
      ]} barCol={null} />
      <div className="space-y-1">
        {t.probabilities.map((p, k) => (
          <div key={k} className="grid grid-cols-[70px_1fr_56px] items-center gap-2 text-[12px]">
            <span style={{ color: k === 1 ? '#3b82f6' : '#ef4444' }}>{t.class_names[k]}</span>
            <span className="h-2.5 rounded-sm" style={{ background: 'var(--border)' }}>
              <span className="block h-full rounded-sm" style={{ width: `${p * 100}%`, background: k === 1 ? '#3b82f6' : '#ef4444' }} />
            </span>
            <span className="font-mono tnum text-right">{pct(p)}</span>
          </div>
        ))}
      </div>
    </>
  );
}

function PredictionStep({ t, cmp, cmpEpoch }: Ctx) {
  const k = t.predicted_class;
  const label = t.probe.label;
  return (
    <>
      <Learn>
        The prediction is the class with the highest probability: <b>{t.class_names[k]}</b> at {pct(t.probabilities[k])}.
        {label !== null
          ? (label === k ? <> That matches the true label — correct.</> : <> The true label is <b>{t.class_names[label]}</b> — the network is wrong on this sample.</>)
          : <> This is a free point, so there is no true label.</>}
        {cmp && cmpEpoch !== null && <> At epoch {cmpEpoch} it predicted <b>{t.class_names[cmp.predicted_class]}</b> at {pct(cmp.probabilities[cmp.predicted_class])}.</>}
      </Learn>
      <Eq>ŷ = argmax_k p_k = {k} ({t.class_names[k]})</Eq>
    </>
  );
}

function LossStep({ t, lab, cmp, cmpEpoch }: Ctx) {
  const src = t.probe.target_source;
  return (
    <>
      <Learn>
        Training needs one number that says how wrong the network is. Cross-entropy uses the probability given to the target class
        <b> {t.class_names[t.target]}</b> ({src === 'label' ? 'the true label' : src === 'user' ? 'chosen by you' : 'the predicted class, since this point has no label'}):
        L = −log({fmt(t.probabilities[t.target], 4)}) = <b className="font-mono">{fmt(t.loss, 4)}</b>.
        {t.probabilities[t.target] > 0.9 ? ' Confident and right: the loss is small.' : t.probabilities[t.target] < 0.5 ? ' The target gets less than half the probability: the loss is large.' : ''}
        {cmp && cmpEpoch !== null && <> At epoch {cmpEpoch} the same input had loss {fmt(cmp.loss, 4)}.</>}
      </Learn>
      {lab && <Eq>L = cross_entropy(z, y) = −log softmax(z)_y = {fmt(t.loss, 6)}</Eq>}
      <p className="text-[12px]" style={{ color: 'var(--text-muted)' }}>
        Backpropagation now asks: how would L change if each number in the network changed a little? The next steps follow those derivatives backwards.
      </p>
    </>
  );
}

function GradLogitsStep({ t, lab, cmp, cmpEpoch }: Ctx) {
  const out = t.layers[t.layers.length - 1];
  return (
    <>
      <Learn>
        For softmax + cross-entropy the gradient at the logits is simply <b>probability minus target</b>: the target logit
        ({t.class_names[t.target]}) gets <Num v={out.grad_z[t.target]} signed digits={3} tone="sign" /> — increasing it would lower the loss —
        and the other class gets the opposite push.
      </Learn>
      {lab && <Eq>δ_out = dL/dz = p − onehot(y) = [{out.grad_z.map((g) => fmtSigned(g, 4)).join(', ')}]</Eq>}
      <VecTable names={t.class_names} cols={[
        { label: 'p', values: t.probabilities },
        { label: 'y', values: t.probabilities.map((_, k) => (k === t.target ? 1 : 0)) },
        { label: 'dL/dz', values: out.grad_z, signed: true, tone: 'sign' },
        ...compareCol(cmp, cmpEpoch, (c) => c.layers[c.layers.length - 1].grad_z, out.grad_z),
      ]} barCol={2} />
    </>
  );
}

function GradParamsStep({ t, step, focus, lab, onPick, lr }: Ctx) {
  const d = denseOf(t, step.layer)!;
  const name = d.neuron_names[focus];
  const delta = d.grad_z[focus];
  const terms = d.input.map((a, j) => ({ j, a, g: d.grad_weight[focus][j] })).sort((x, y) => Math.abs(y.g) - Math.abs(x.g));
  const top = terms[0];
  const pv = t.sgd_preview;
  return (
    <>
      <Learn>
        Each weight's gradient is the neuron's error signal <span className="font-mono">δ</span> times the input it received.
        <b className="font-mono"> {name}</b> has δ = <Num v={delta} signed digits={4} tone="sign" />; its largest weight gradient is for
        <b className="font-mono"> {top.j < d.input_names.length ? d.input_names[top.j] : ''}</b> ({fmt(delta, 3)} × {fmt(top.a, 3)} = <Num v={top.g} signed digits={4} tone="sign" />).
        Gradient descent moves each weight <i>against</i> its gradient: with learning rate {lr} this weight would change by
        <b className="font-mono"> {fmtSigned(-lr * top.g, 4)}</b>.
        {Math.abs(delta) < 1e-6 && ' δ is 0 here (inactive or disabled neuron), so none of its weights would change for this input.'}
      </Learn>
      {lab && <Eq>dL/dW = δ ⊗ a_prev ∈ ℝ^{d.weight.length}×{d.input.length}   ·   dL/db = δ   ·   Δθ_SGD = −η·∇θ, η = {lr}</Eq>}
      <H hint="sorted by |dL/dw|">{name}: incoming weight gradients</H>
      <div className="overflow-x-auto">
        <table className="w-full text-[12px] tnum border-collapse">
          <thead>
            <tr style={{ color: 'var(--text-faint)' }}>
              <th className="text-left font-medium py-1">from</th><th className="text-right font-medium font-mono">a</th>
              <th className="text-right font-medium font-mono">dL/dw = δ·a</th><th className="text-right font-medium font-mono">w now</th>
              <th className="text-right font-medium font-mono">−η·dL/dw</th>
            </tr>
          </thead>
          <tbody>
            {terms.slice(0, lab ? 24 : 6).map((x) => (
              <tr key={x.j}>
                <td className="py-[3px] font-mono" style={{ color: 'var(--text-muted)' }}>{d.input_names[x.j]}</td>
                <td className="text-right"><Num v={x.a} /></td>
                <td className="text-right"><Num v={x.g} signed tone="sign" /></td>
                <td className="text-right"><Num v={d.weight[focus][x.j]} /></td>
                <td className="text-right"><Num v={-lr * x.g} signed tone="sign" /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Eq>dL/db({name}) = δ = <Num v={d.grad_bias[focus]} signed />   →   Δb = <Num v={-lr * d.grad_bias[focus]} signed /></Eq>
      <H hint="click to focus">Error signal δ of every neuron in {d.label}</H>
      <VecTable names={d.neuron_names} focus={focus} onPick={(i) => onPick(step.layer, i)}
        cols={[{ label: 'δ = dL/dz', values: d.grad_z, signed: true, tone: 'sign' }, { label: '‖dL/dW row‖', values: d.grad_weight.map((r) => Math.hypot(...r)) }]} barCol={0} />
      {pv && (
        <div className="rounded-md border p-3 space-y-1" style={{ borderColor: 'var(--border)' }}>
          <H>One real SGD step on this sample (preview)</H>
          <p className="text-[12.5px] tnum" style={{ color: 'var(--text-primary)' }}>
            Applying θ ← θ − {pv.learning_rate}·∇θ to <i>all</i> layers at once (on a copy) changes this sample's loss from
            <b className="font-mono"> {fmt(pv.loss_before, 4)}</b> to <b className="font-mono">{fmt(pv.loss_after, 4)}</b> and P({t.class_names[t.target]}) from
            {' '}{pct(pv.target_prob_before)} to {pct(pv.target_prob_after)}. Update size ‖Δθ‖ = {fmt(pv.update_norm, 4)}.
          </p>
          <p className="text-[11.5px]" style={{ color: 'var(--text-faint)' }}>{pv.note}</p>
        </div>
      )}
    </>
  );
}

function GradInputStep({ t, step, focus, lab, onPick, cmp, cmpEpoch }: Ctx) {
  // Gradient flows from layer step.layer + 1 back to step.layer (or to the input for saliency).
  const src = denseOf(t, step.edgeLayer!)!;
  const names = step.layer === 0 ? t.feature_names : t.layers[step.layer - 1].neuron_names;
  const b = backpropTerms(src, focus);
  const target = names[focus];
  const top = b.terms[0];
  const values = src.grad_input;
  const sal = step.kind === 'saliency';
  return (
    <>
      <Learn>
        {sal
          ? <>The gradient reaches the input. <b className="font-mono">dL/d{target} = {fmtSigned(b.reported, 4)}</b>: moving {target} slightly {b.reported > 0 ? 'up would increase' : 'up would decrease'} the loss. Inputs are not trained — this is a sensitivity (saliency) map.</>
          : <><b className="font-mono">{target}</b> fed every neuron of {src.label}, so its blame is the sum of their error signals weighted by the connecting weights:
            largest term from <b className="font-mono">{top.name}</b> (w = {fmt(top.w, 3)} × δ = {fmt(top.delta, 3)} = <Num v={top.product} signed digits={4} tone="sign" />);
            total <b className="font-mono">dL/da = {fmtSigned(b.reported, 4)}</b>.</>}
      </Learn>
      {lab && <Eq>dL/da_prev = Wᵀ·δ   ·   ({src.weight[0].length}×{src.weight.length})·({src.grad_z.length}) → ({src.input.length})   ·   Σ terms = {fmtSigned(b.sum, 5)}</Eq>}
      <H hint="sorted by |w·δ|">{target}: dL/da = Σ_i w_i·δ_i</H>
      <VecTable names={b.terms.slice(0, lab ? 24 : 8).map((x) => x.name)}
        cols={[
          { label: 'w', values: b.terms.slice(0, lab ? 24 : 8).map((x) => x.w) },
          { label: 'δ', values: b.terms.slice(0, lab ? 24 : 8).map((x) => x.delta), signed: true },
          { label: 'w·δ', values: b.terms.slice(0, lab ? 24 : 8).map((x) => x.product), signed: true, tone: 'sign' },
        ]} barCol={2} />
      <H hint="click to focus">{sal ? 'Input saliency dL/dx' : `dL/da for every neuron of ${t.layers[step.layer - 1].label}`}</H>
      <VecTable names={names} focus={focus} onPick={step.layer > 0 ? (i) => onPick(step.layer, i) : undefined}
        cols={[{ label: 'dL/da', values, signed: true, tone: 'sign' }, ...compareCol(cmp, cmpEpoch, (c) => c.layers[step.edgeLayer! - 1].grad_input, values)]} barCol={0} />
    </>
  );
}

function GradActivationStep({ t, step, focus, lab, onPick }: Ctx) {
  const d = denseOf(t, step.layer)!;
  const name = d.neuron_names[focus];
  const blocked = d.local_grad!.filter((v) => Math.abs(v) < 1e-9).length;
  const flagged: Record<number, string> = {};
  d.ablated.forEach((i) => { flagged[i] = 'disabled'; });
  return (
    <>
      <Learn>
        To reach the weights of {d.label}, the gradient crosses its activation: it is multiplied by the slope f′(z) at each neuron's operating point.
        <b className="font-mono"> {name}</b>: {fmtSigned(d.grad_a![focus], 4)} × f′ = {fmt(d.local_grad![focus], 4)} → δ = <Num v={d.grad_z[focus]} signed digits={4} tone="sign" />.
        {blocked > 0 && <> {blocked} of {d.z.length} neurons have slope 0 here {d.activation === 'ReLU' ? '(ReLU is flat for z < 0)' : ''}{d.ablated.length ? ' or are disabled' : ''}: no gradient passes through them for this input.</>}
      </Learn>
      {lab && <Eq>δ = dL/dz = dL/da ⊙ f′(z){d.ablated.length ? ' ⊙ mask' : ''}   ·   f = {d.activation}</Eq>}
      <VecTable names={d.neuron_names} focus={focus} onPick={(i) => onPick(step.layer, i)} flagged={flagged}
        cols={[
          { label: 'z', values: d.z, signed: true },
          { label: 'dL/da', values: d.grad_a!, signed: true },
          { label: "f′(z)", values: d.local_grad! },
          { label: 'δ', values: d.grad_z, signed: true, tone: 'sign' },
        ]} barCol={3} />
    </>
  );
}

export function StepDetail(ctx: Ctx) {
  switch (ctx.step.kind) {
    case 'input': return <InputStep {...ctx} />;
    case 'linear': case 'logits': return <LinearStep {...ctx} />;
    case 'activation': return <ActivationStep {...ctx} />;
    case 'softmax': return <SoftmaxStep {...ctx} />;
    case 'prediction': return <PredictionStep {...ctx} />;
    case 'loss': return <LossStep {...ctx} />;
    case 'grad_logits': return <GradLogitsStep {...ctx} />;
    case 'grad_params': return <GradParamsStep {...ctx} />;
    case 'grad_input': case 'saliency': return <GradInputStep {...ctx} />;
    case 'grad_activation': return <GradActivationStep {...ctx} />;
  }
}
