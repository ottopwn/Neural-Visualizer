// Plain-language narration for the Training Time Machine (Learn mode).
// Every sentence is a template filled with numbers from a real payload;
// a claim is only made when the data supports it.

import { fmt, fmtSigned, pct } from './format';
import type { ComponentHistory, EpochComparison, Frame } from './types';

/** Accuracy within this margin of the majority-class rate counts as "not better than guessing". */
const GUESS_MARGIN = 0.02;

function trend(a: number, b: number, up: string, down: string, same: string, eps = 1e-9): string {
  if (Math.abs(b - a) <= eps) return same;
  return b > a ? up : down;
}

/** Fraction of samples predicted as each class. */
function classShares(predictions: number[], nClasses: number): number[] {
  const counts = new Array(nClasses).fill(0);
  predictions.forEach((p) => { counts[p] += 1; });
  return counts.map((c) => (predictions.length ? c / predictions.length : 0));
}

export function describeFrame(frame: Frame, classNames: string[], majorityRate: number): string[] {
  const out: string[] = [];
  if (frame.epoch === 0) {
    out.push('Epoch 0 is the network before any training: every weight and bias still has its random initial value.');
  }

  const shares = classShares(frame.predictions, classNames.length);
  const only = shares.findIndex((s) => s === 1);
  if (only >= 0) {
    out.push(`It currently predicts ${classNames[only]} for every point in the dataset, so it has not separated the classes at all.`);
  } else if (frame.accuracy <= majorityRate + GUESS_MARGIN) {
    out.push(`It classifies ${pct(frame.accuracy)} of the points correctly — no better than always guessing the most common class (${pct(majorityRate)}), so it has not yet separated the classes.`);
  } else if (frame.accuracy >= 0.995) {
    out.push(`It classifies every point of the dataset correctly (${pct(frame.accuracy)}); the loss is ${fmt(frame.loss, 4)}.`);
  } else {
    out.push(`It classifies ${pct(frame.accuracy)} of the points correctly, compared with ${pct(majorityRate)} for always guessing the most common class. Loss: ${fmt(frame.loss, 4)}.`);
  }

  const p = frame.previous;
  if (p) {
    out.push(`Since the previous stored checkpoint (epoch ${p.epoch}), accuracy ${trend(p.accuracy, frame.accuracy, 'rose', 'fell', 'stayed')} ${p.accuracy === frame.accuracy ? `at ${pct(frame.accuracy)}` : `from ${pct(p.accuracy)} to ${pct(frame.accuracy)}`} and the loss ${trend(p.loss, frame.loss, 'went up', 'went down', 'did not change')} from ${fmt(p.loss, 4)} to ${fmt(frame.loss, 4)}.`);
    if (p.changed === 0) {
      out.push('No point changed its predicted class in that interval.');
    } else {
      out.push(`${p.changed} point${p.changed > 1 ? 's' : ''} changed predicted class: ${p.fixed} became correct and ${p.broken} became wrong.`);
    }
    if (p.boundary_flip_fraction !== null && p.boundary_flip_fraction > 0) {
      out.push(`The decision boundary moved: ${pct(p.boundary_flip_fraction)} of the plotted input plane switched class.`);
    }
  }

  const probeClass = classNames[frame.probe_predicted];
  const conf = frame.probe_probabilities[frame.probe_predicted];
  const truth = frame.probe.label;
  out.push(`For the probe point it predicts ${probeClass} with ${pct(conf)} probability${truth !== null ? ` (true label: ${classNames[truth]}${truth === frame.probe_predicted ? ', correct' : ', wrong'})` : ''}.`);
  return out;
}

export function describeComparison(c: EpochComparison, classNames: string[]): string[] {
  const { a, b } = c;
  const out: string[] = [];
  out.push(`Between epoch ${a.epoch} and epoch ${b.epoch}, accuracy ${trend(a.accuracy, b.accuracy, 'increased', 'decreased', 'stayed')} ${a.accuracy === b.accuracy ? `at ${pct(a.accuracy)}` : `from ${pct(a.accuracy)} to ${pct(b.accuracy)}`} while the loss ${trend(a.loss, b.loss, 'increased', 'decreased', 'stayed')} from ${fmt(a.loss, 4)} to ${fmt(b.loss, 4)}.`);
  if (c.changed === 0) {
    out.push('Every point keeps the same predicted class.');
  } else {
    out.push(`${c.changed} point${c.changed > 1 ? 's' : ''} (${pct(c.changed_fraction)}) changed predicted class: ${c.fixed} became correct, ${c.broken} became wrong.`);
  }
  if (c.boundary_flip_fraction !== null && c.boundary_flip_fraction > 0) {
    out.push(`${pct(c.boundary_flip_fraction)} of the plotted input plane is assigned to a different class at epoch ${b.epoch}.`);
  }
  if (a.probe_predicted !== b.probe_predicted) {
    out.push(`The probe point's prediction flipped from ${classNames[a.probe_predicted]} to ${classNames[b.probe_predicted]}.`);
  }
  const biggest = [...c.layers].sort((x, y) => y.relative_change - x.relative_change)[0];
  if (biggest && c.total_delta_norm > 0) {
    out.push(`The parameters that changed most, relative to their size at epoch ${a.epoch}, are in ${biggest.label}: they moved by ${pct(biggest.relative_change, 0)} (‖Δθ‖ = ${fmt(Math.hypot(biggest.weight_delta_norm, biggest.bias_delta_norm), 3)}).`);
  } else if (c.total_delta_norm === 0) {
    out.push('The two checkpoints have identical parameters.');
  }
  const comp = c.component;
  if (comp) {
    const row = (key: string) => comp.rows.find((r) => r.key === key);
    const act = row('act_mean');
    const active = row('active_frac');
    const bias = row('bias');
    const w = row('weight');
    if (act && act.a !== null && act.b !== null) {
      out.push(`${comp.name}'s average output over the dataset went from ${fmt(act.a, 3)} to ${fmt(act.b, 3)} (${fmtSigned(act.delta, 3)}).`);
    }
    if (active && active.a !== null && active.b !== null && active.a !== active.b) {
      out.push(`It is active (non-zero) for ${pct(active.b, 0)} of the points at epoch ${b.epoch}, versus ${pct(active.a, 0)} at epoch ${a.epoch}.`);
    }
    if (bias && bias.a !== null && bias.b !== null) {
      out.push(`Its bias moved from ${fmt(bias.a, 3)} to ${fmt(bias.b, 3)}.`);
    }
    if (w && w.a !== null && w.b !== null) {
      out.push(`This connection's weight moved from ${fmt(w.a, 3)} to ${fmt(w.b, 3)}${Math.sign(w.a) !== Math.sign(w.b) && w.a !== 0 && w.b !== 0 ? ', changing sign' : ''}.`);
    }
  }
  return out;
}

/** One sentence about how a series evolved up to the selected epoch. */
export function describeSeries(h: ComponentHistory, key: string, epoch: number): string | null {
  const s = h.series.find((x) => x.key === key);
  if (!s) return null;
  const i = h.epochs.indexOf(epoch);
  const first = s.values.findIndex((v) => v !== null);
  if (i < 0 || first < 0 || s.values[i] === null) return null;
  const v0 = s.values[first]!;
  const v = s.values[i]!;
  if (i === first) return `${h.name}: ${s.label} = ${fmt(v, 3)} at epoch ${epoch}.`;
  if (Math.abs(v - v0) < 1e-9) return `${h.name}: ${s.label} has stayed at ${fmt(v, 3)} since epoch ${h.epochs[first]}.`;
  return `${h.name}: ${s.label} went from ${fmt(v0, 3)} at epoch ${h.epochs[first]} to ${fmt(v, 3)} at epoch ${epoch} (${fmtSigned(v - v0, 3)}).`;
}

/** Dead-neuron note from the real `active_frac` series (hidden neurons only). */
export function describeSilence(h: ComponentHistory, epoch: number, activation: string | null = null): string | null {
  const s = h.series.find((x) => x.key === 'active_frac');
  const i = h.epochs.indexOf(epoch);
  if (!s || i < 0 || s.values[i] !== 0) return null;
  const everActive = s.values.slice(0, i + 1).some((v) => v !== null && v > 0);
  return everActive
    ? `At epoch ${epoch}, ${h.name} outputs 0 for every point in the dataset: it has become a "dead" neuron.`
    : `${h.name} has output 0 for every point in the dataset at every stored checkpoint up to epoch ${epoch}: a "dead" neuron.${
      activation === 'ReLU' ? ' ReLU has zero slope there, so the data sends no gradient to its incoming weights and bias.' : ''}`;
}
