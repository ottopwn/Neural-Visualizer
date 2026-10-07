import { Ban, RotateCcw } from 'lucide-react';
import { explainNeuron, gradientHint } from '../../forge/explain';
import { fmt, fmtSigned, pct, rankByMagnitude } from '../../forge/format';
import * as ivs from '../../forge/interventions';
import { useForgeStore } from '../../forge/store';
import type { ModelStructure, NeuronInspection } from '../../forge/types';
import { ActivationCurve, ContributionBars, HeatmapCanvas, KV, MiniHistogram, Section } from './charts';
import { responseColor, useDatasetPoints } from './hooks';
import { ValueEditor } from './shared';

export function NeuronPanel({ n, structure }: { n: NeuronInspection; structure: ModelStructure }) {
  const mode = useForgeStore((s) => s.mode);
  const interventions = useForgeStore((s) => s.interventions);
  const select = useForgeStore((s) => s.select);
  const setInterventions = useForgeStore((s) => s.setInterventions);
  const addIntervention = useForgeStore((s) => s.addIntervention);
  const points = useDatasetPoints();
  const lab = mode === 'lab';
  const classNames = structure.class_names;
  const target = classNames[n.probe.target];
  const ref = { kind: 'neuron' as const, layer: n.layer, index: n.index };

  return (
    <div className="space-y-2.5">
      {/* Learn-mode narrative, derived from the numbers below */}
      {!lab && (
        <Section title="What is happening">
          <ul className="space-y-1.5 text-[12px] leading-relaxed" style={{ color: 'var(--text-primary)' }}>
            {explainNeuron(n, classNames).map((t, i) => <li key={i}>{t}</li>)}
          </ul>
        </Section>
      )}

      {/* The computation itself */}
      {n.role !== 'input' && n.contributions && n.weights && n.inputs && n.input_names && (
        <Section
          title="Weighted sum"
          hint={lab ? `z = Σ wᵢ·aᵢ + b · ${n.inputs.length} inputs` : 'each input × its weight'}
        >
          <ContributionBars
            names={n.input_names}
            contributions={n.contributions}
            weights={n.weights}
            inputs={n.inputs}
            maxRows={lab ? 12 : 6}
            onSelect={(i) => select({ kind: 'connection', layer: n.layer, source: i, target: n.index })}
          />
          <div className="mt-2 pt-2 border-t space-y-0.5" style={{ borderColor: 'var(--border)' }}>
            <KV k="Σ contributions" v={fmtSigned(n.contributions.reduce((a, b) => a + b, 0), 4)} />
            <KV k={n.bias_edit ? 'bias (edited)' : 'bias b'} v={fmtSigned(n.bias, 4)} color={n.bias_edit ? '#f59e0b' : undefined} />
            <KV k="pre-activation z" v={fmt(n.pre_activation, 4)} color="var(--text-info)" />
            <KV
              k={n.role === 'output' ? 'softmax probability' : `${n.activation_fn}(z)`}
              v={n.role === 'output' ? pct(n.value, 2) : fmt(n.value, 4)}
              color="#fde047"
            />
          </div>
          <p className="text-[10px] mt-1.5" style={{ color: 'var(--text-faint)' }}>Click a bar to inspect that connection.</p>
        </Section>
      )}

      {n.role === 'hidden' && n.activation_fn && n.pre_activation !== null && (
        <Section title="Activation function" hint={n.activation_fn}>
          <ActivationCurve fn={n.activation_fn} z={n.pre_activation} a={n.value} natural={n.natural_value} />
          {n.ablated && (
            <p className="text-[11px] mt-1" style={{ color: 'var(--text-neg)' }}>
              Disabled: the curve gives {fmt(n.natural_value, 3)}, but 0 is sent downstream.
            </p>
          )}
        </Section>
      )}

      {n.role === 'input' && (
        <Section title="Value">
          <KV k={`${n.name} on this probe`} v={fmt(n.value, 4)} color="#fde047" />
        </Section>
      )}

      {/* What does this neuron respond to? */}
      {n.response_map && (
        <Section
          title={n.role === 'output' ? 'Response over input space' : 'Where it fires'}
          hint={n.response_map.label}
        >
          <HeatmapCanvas
            ariaLabel={`${n.name} response over the input plane`}
            values={n.response_map.values}
            color={responseColor(n.response_map, n.role === 'output' ? n.index : null)}
            xRange={n.response_map.x_range}
            yRange={n.response_map.y_range}
            points={points}
            marker={n.probe.x.length === 2 ? [n.probe.x[0], n.probe.x[1]] : null}
            height={150}
          />
          <p className="text-[10px] mt-1" style={{ color: 'var(--text-faint)' }}>
            Computed by running the network on a {n.response_map.resolution}×{n.response_map.resolution} grid of inputs
            {lab && <> · range [{fmt(n.response_map.value_range[0], 3)}, {fmt(n.response_map.value_range[1], 3)}]</>}.
          </p>
        </Section>
      )}

      {/* Dataset-wide behaviour */}
      <Section title="Across the dataset" hint={`${n.dataset_stats.count} samples`}>
        <MiniHistogram hist={n.dataset_histogram} marker={n.value} color={n.role === 'input' ? '#60a5fa' : '#34d399'} />
        <div className="grid grid-cols-2 gap-x-4 mt-1">
          <KV k="mean" v={fmt(n.dataset_stats.mean)} />
          <KV k="std" v={fmt(n.dataset_stats.std)} />
          <KV k="min" v={fmt(n.dataset_stats.min)} />
          <KV k="max" v={fmt(n.dataset_stats.max)} />
          {n.role === 'hidden' && <KV k="active" v={pct(1 - n.inactive_fraction, 0)} color={n.inactive_fraction > 0.99 ? 'var(--text-neg)' : undefined} />}
        </div>
      </Section>

      {/* Gradients */}
      {lab ? (
        <Section title="Gradients" hint={`dLoss/d· · loss = CE(target = ${target}, ${n.probe.target_source})`}>
          {n.role === 'input' && <KV k="dL/dx (saliency)" v={fmt(n.grad_value, 5)} />}
          {n.role === 'hidden' && <KV k="dL/da" v={fmt(n.grad_value, 5)} />}
          {n.role !== 'input' && (
            <>
              <KV k={n.role === 'output' ? 'dL/dlogit' : 'dL/dz'} v={fmt(n.grad_pre_activation, 5)} />
              <KV k="dL/db" v={fmt(n.grad_bias, 5)} />
              {n.grad_weights && n.input_names && (
                <div className="mt-1">
                  <div className="text-[10px] mb-0.5" style={{ color: 'var(--text-faint)' }}>largest dL/dw (= dL/dz · aᵢ)</div>
                  {rankByMagnitude(n.grad_weights).slice(0, 5).map((i) => (
                    <KV key={i} k={`w[${n.input_names![i]}]`} v={fmt(n.grad_weights![i], 5)} />
                  ))}
                </div>
              )}
            </>
          )}
        </Section>
      ) : (
        (() => {
          const hint = n.role === 'input'
            ? null
            : gradientHint(n.grad_bias, `${n.name}'s bias`, target);
          return hint ? (
            <Section title="Learning signal">
              <p className="text-[12px]" style={{ color: 'var(--text-primary)' }}>{hint}</p>
              <p className="text-[10px] mt-1" style={{ color: 'var(--text-faint)' }}>From the real gradient of the loss for {target} ({n.probe.target_source === 'label' ? 'the true label' : n.probe.target_source === 'user' ? 'chosen target' : 'the predicted class'}).</p>
            </Section>
          ) : null;
        })()
      )}

      {/* Outgoing connections */}
      {lab && n.outgoing_weights && n.outgoing_names && (
        <Section title="Outgoing weights" hint={`to ${n.outgoing_weights.length} neurons`}>
          {rankByMagnitude(n.outgoing_weights).slice(0, 6).map((j) => (
            <button key={j} type="button" className="w-full hover:bg-[var(--bg-hover)] rounded px-1"
              onClick={() => select({ kind: 'connection', layer: n.layer + 1, source: n.index, target: j })}>
              <KV k={`→ ${n.outgoing_names![j]}`} v={fmtSigned(n.outgoing_weights![j], 4)}
                color={n.outgoing_weights![j] >= 0 ? 'var(--text-pos)' : 'var(--text-neg)'} />
            </button>
          ))}
        </Section>
      )}

      {lab && Object.keys(n.shapes).length > 0 && (
        <Section title="Tensor shapes">
          {Object.entries(n.shapes).map(([k, v]) => <KV key={k} k={k} v={`[${v.join(', ')}]`} />)}
        </Section>
      )}

      {/* What-if */}
      {n.role !== 'input' && (
        <Section title="What-if" hint="changes the real forward pass">
          <div className="space-y-2">
            {n.role === 'hidden' && (
              <button
                type="button"
                onClick={() => setInterventions(ivs.toggleAblation(interventions, n.layer, n.index))}
                className="w-full flex items-center justify-center gap-1.5 text-xs py-1.5 rounded-lg border transition-colors"
                style={n.ablated
                  ? { borderColor: '#10b981', color: 'var(--text-pos)', background: 'rgba(16,185,129,0.08)' }
                  : { borderColor: 'rgba(239,68,68,0.5)', color: 'var(--text-neg)', background: 'rgba(239,68,68,0.08)' }}
              >
                {n.ablated ? <><RotateCcw size={12} /> Re-enable {n.name}</> : <><Ban size={12} /> Disable {n.name}</>}
              </button>
            )}
            {n.bias !== null && (
              <ValueEditor
                key={`${ivs.refKey(ref)}:${n.bias}`}
                label="bias"
                current={n.bias}
                original={n.bias_edit?.original ?? null}
                onApply={(value) => addIntervention({ type: 'set_bias', layer: n.layer, index: n.index, value })}
                onRestore={() => setInterventions(ivs.restoreBias(interventions, n.layer, n.index))}
              />
            )}
          </div>
        </Section>
      )}

      {n.notes.length > 0 && (
        <ul className="text-[10px] space-y-0.5 px-1" style={{ color: 'var(--text-faint)' }}>
          {n.notes.map((t, i) => <li key={i}>• {t}</li>)}
        </ul>
      )}
    </div>
  );
}
