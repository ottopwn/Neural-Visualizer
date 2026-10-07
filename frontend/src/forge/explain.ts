// Plain-language explanations for Learn mode.  Every sentence is derived
// from real inspection values; nothing here invents numbers.

import { fmt, fmtSigned, pct, rankByMagnitude } from './format';
import type { ConnectionInspection, LayerInspection, NeuronInspection } from './types';

export function explainNeuron(n: NeuronInspection, classNames: string[]): string[] {
  const out: string[] = [];
  if (n.role === 'input') {
    out.push(`${n.name} is one coordinate of the selected point: its value is ${fmt(n.value, 3)}. Inputs have no weights — they just pass the number on.`);
    if (n.grad_value !== null) {
      out.push(n.grad_value < 0
        ? `Moving this coordinate up would make the network more confident in ${classNames[n.probe.target]}.`
        : `Moving this coordinate down would make the network more confident in ${classNames[n.probe.target]}.`);
    }
    return out;
  }

  const k = n.inputs?.length ?? 0;
  if (n.role === 'output') {
    out.push(`${n.name} is the network's raw score (logit) for ${classNames[n.index]}: ${fmt(n.pre_activation, 3)}. Softmax turns the scores of all classes into probabilities — this one gets ${pct(n.value)}.`);
  } else {
    out.push(`${n.name} receives ${k} numbers from the previous layer, multiplies each by a weight, adds them up together with its bias, and gets z = ${fmt(n.pre_activation, 3)}. Then ${n.activation_fn} turns z into its output: ${fmt(n.value, 3)}.`);
  }

  if (n.contributions && n.input_names && n.contributions.length) {
    const [top] = rankByMagnitude(n.contributions);
    const c = n.contributions[top];
    out.push(`The biggest single influence right now is ${n.input_names[top]} (${fmtSigned(c, 3)}), which ${c >= 0 ? 'pushes z up' : 'pulls z down'}.`);
  }

  if (n.ablated) {
    out.push(`You disabled this neuron, so it sends 0 to the next layer (it would have sent ${fmt(n.natural_value, 3)}).`);
  } else if (n.role === 'hidden' && Math.abs(n.value) < 1e-6 && (n.activation_fn === 'ReLU')) {
    out.push('z is negative, so ReLU outputs exactly 0: this neuron is silent for this input.');
  }

  if (n.role === 'hidden') {
    const active = 1 - n.inactive_fraction;
    if (n.inactive_fraction >= 0.999) {
      out.push('It outputs 0 for every point in the dataset — a "dead" neuron that currently contributes nothing.');
    } else {
      out.push(`Across the whole dataset it is active (non-zero) for ${pct(active, 0)} of the points.`);
    }
  }
  return out;
}

export function explainLayer(l: LayerInspection): string[] {
  if (l.role === 'input') {
    return [`The input layer just holds the ${l.size} features of the selected point.`];
  }
  const out = [
    `${l.label} has ${l.size} neurons. Each one is connected to all ${l.fan_in} outputs of the previous layer, so it learns a ${l.size}×${l.fan_in} grid of weights plus ${l.size} biases — ${l.param_count} numbers in total.`,
  ];
  if (l.role === 'hidden') {
    const silent = l.activations.filter((a) => Math.abs(a) < 1e-6).length;
    out.push(`For the selected point, ${l.size - silent} of ${l.size} neurons are active.`);
    if (l.never_active.length) {
      out.push(`${l.never_active.length} neuron(s) never activate on any point of the dataset.`);
    }
  }
  return out;
}

export function explainConnection(c: ConnectionInspection): string[] {
  return [
    `This connection carries ${c.source_name}'s output (${fmt(c.source_value, 3)}) into ${c.target_name}, multiplied by the weight ${fmt(c.weight, 3)}.`,
    `So it adds ${fmtSigned(c.contribution, 3)} to ${c.target_name}'s sum — ${pct(c.share_of_input, 0)} of all the incoming signal (rank ${c.rank} of ${c.fan_in}).`,
  ];
}

/** Sign convention: a negative dLoss/dθ means increasing θ lowers the loss. */
export function gradientHint(grad: number | null, what: string, target: string): string | null {
  if (grad === null || Math.abs(grad) < 1e-9) return null;
  return grad < 0
    ? `Increasing ${what} would make the network more confident in ${target}.`
    : `Decreasing ${what} would make the network more confident in ${target}.`;
}
