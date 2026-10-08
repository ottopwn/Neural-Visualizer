import { AlertCircle, CheckCircle2, ChevronDown, Loader2, Minus, Play, Plus, RefreshCw, Zap } from 'lucide-react';
import { useState, type ReactNode } from 'react';
import { configSignature, FORGE_MODELS, useExperiment, type Status } from '../../app/experiment';
import { architectureString } from '../../forge/selection';
import { useForgeStore } from '../../forge/store';
import { useNetworkStore } from '../../store/networkStore';
import type { ActivationType, DatasetType, ModelType, RegType } from '../../types';
import { DatasetPreview } from '../DatasetPreview';
import { DatasetUpload } from '../DatasetUpload';
import { useT } from '../../i18n';

const MODEL_TYPES: ModelType[] = ['ANN', 'CNN', 'RNN', 'LSTM', 'GAN', 'Transformer', 'Diffuser'];
const ACTIVATIONS: ActivationType[] = ['ReLU', 'Tanh', 'Sigmoid', 'LeakyReLU', 'ELU'];
const NEURON_OPTIONS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128];
const DATASETS: DatasetType[] = ['Circle', 'Gaussian', 'XOR', 'Spiral'];
const BATCH_SIZES = [1, 8, 16, 32, 64, 128];
const LR_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1];
const EPOCH_MARKS = [1, 5, 10, 20, 50, 100];
const REG_TYPES: RegType[] = ['None', 'L1', 'L2', 'L1L2'];

/** Total hidden neurons the UI allows (keeps the graph readable and the API fast). */
const NEURON_BUDGET = 192;

function Section({ title, children, defaultOpen = true, aside }: { title: string; children: ReactNode; defaultOpen?: boolean; aside?: ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="border-b" style={{ borderColor: 'var(--border)' }}>
      <button type="button" className="w-full flex items-center gap-2 px-4 py-2.5 text-left" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span className="eyebrow flex-1">{title}</span>
        {aside}
        <ChevronDown size={14} style={{ color: 'var(--text-faint)', transform: open ? 'none' : 'rotate(-90deg)', transition: 'transform 0.15s' }} />
      </button>
      {open && <div className="px-4 pb-4 space-y-3">{children}</div>}
    </section>
  );
}

function Field({ label, value, children }: { label: string; value?: ReactNode; children: ReactNode }) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="text-[12px]" style={{ color: 'var(--text-muted)' }}>{label}</span>
        {value !== undefined && <span className="text-[12px] font-mono tnum" style={{ color: 'var(--text-primary)' }}>{value}</span>}
      </div>
      {children}
    </div>
  );
}

function Choices<T extends string | number>({ options, value, onChange, format, cols, label }: {
  options: T[]; value: T; onChange: (v: T) => void; format?: (v: T) => string; cols: number; label: string;
}) {
  return (
    <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }} role="group" aria-label={label}>
      {options.map((o) => (
        <button key={String(o)} type="button" className="choice tnum" aria-pressed={value === o} onClick={() => onChange(o)}>
          {format ? format(o) : String(o)}
        </button>
      ))}
    </div>
  );
}

function StatusLine({ status }: { status: Status }) {
  if (status.type === 'idle' || status.type === 'loading') return null;
  const err = status.type === 'error';
  return (
    <div role={err ? 'alert' : 'status'} className="flex items-start gap-2 text-[12px] px-2.5 py-2 rounded-md border"
      style={{ borderColor: err ? 'var(--neg)' : 'var(--border)', color: err ? 'var(--text-neg)' : 'var(--text-muted)', background: err ? 'color-mix(in srgb, var(--neg) 8%, transparent)' : 'transparent' }}>
      {err ? <AlertCircle size={14} className="flex-shrink-0 mt-px" /> : <CheckCircle2 size={14} className="flex-shrink-0 mt-px" style={{ color: 'var(--pos)' }} />}
      <span>{status.message}</span>
    </div>
  );
}

export function ExperimentPanel() {
  const net = useNetworkStore((s) => s.networkConfig);
  const train = useNetworkStore((s) => s.trainingConfig);
  const setNet = useNetworkStore((s) => s.setNetworkConfig);
  const setTrain = useNetworkStore((s) => s.setTrainingConfig);
  const customDataset = useNetworkStore((s) => s.customDataset);
  const setCustomDataset = useNetworkStore((s) => s.setCustomDataset);
  const graphSource = useNetworkStore((s) => s.graphSource);
  const networkBuilt = useNetworkStore((s) => s.networkBuilt);
  const session = useForgeStore((s) => s.session);
  const { buildStatus, trainStatus, builtSignature, build, train: runTraining } = useExperiment();
  const t = useT();
  const dirty = networkBuilt && builtSignature !== null && builtSignature !== configSignature();

  const real = FORGE_MODELS.has(net.model_type);
  const hidden = net.neurons.slice(0, net.n_layers);
  const budgetLeft = NEURON_BUDGET - hidden.reduce((a, b) => a + b, 0);

  const setLayers = (n: number) => {
    const neurons = [...net.neurons];
    const activations = [...net.activations];
    while (neurons.length < n) { neurons.push(8); activations.push('ReLU'); }
    setNet({ n_layers: n, neurons, activations });
  };
  const setNeurons = (i: number, v: number) => {
    const neurons = [...net.neurons];
    neurons[i] = v;
    setNet({ neurons });
  };
  const setActivation = (i: number, v: ActivationType) => {
    const activations = [...net.activations];
    activations[i] = v;
    setNet({ activations });
  };

  const building = buildStatus.type === 'loading';
  const training = trainStatus.type === 'loading';

  return (
    <div className="h-full flex flex-col min-h-0">
      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="px-4 pt-3 pb-2">
          <div className="eyebrow">{t.panel.experiment}</div>
          {session ? (
            <div className="mt-1.5 text-[12px] leading-relaxed" style={{ color: 'var(--text-muted)' }}>
              <span className="font-mono" style={{ color: 'var(--text-primary)' }}>MLP {architectureString(session.structure)}</span>
              <br />{t.panel.summary(session.structure.param_count.toLocaleString(), session.dataset_name, session.epoch)}
            </div>
          ) : (
            <p className="mt-1.5 text-[12px] leading-relaxed" style={{ color: 'var(--text-muted)' }}>
              {t.panel.configure} <b style={{ color: 'var(--text-primary)' }}>{t.panel.build}</b> {t.panel.configureTail}
            </p>
          )}
        </div>

        <Section title={t.panel.model}>
          <Field label={t.panel.modelType}>
            <select className="select-base" value={net.model_type} aria-label={t.panel.modelType}
              onChange={(e) => setNet({ model_type: e.target.value as ModelType })}>
              {MODEL_TYPES.map((m) => (
                <option key={m} value={m}>{m}{FORGE_MODELS.has(m) ? t.panel.realModel : t.panel.illustrative}</option>
              ))}
            </select>
          </Field>
          {!real && (
            <p className="text-[11.5px] leading-relaxed px-2.5 py-2 rounded-md border" style={{ borderColor: 'var(--tm-whatif)', color: 'var(--text-muted)' }}>
              <b style={{ color: 'var(--tm-whatif-text)' }}>{t.panel.illustrativeOnly}</b> {t.panel.illustrativeBody(net.model_type)} <b>ANN</b>.
            </p>
          )}

          <Field label={t.panel.hiddenLayers} value={net.n_layers}>
            <div className="flex items-center gap-1.5">
              <button type="button" className="btn-secondary !p-1.5" aria-label={t.panel.removeLayer} disabled={net.n_layers <= 1}
                onClick={() => setLayers(net.n_layers - 1)}><Minus size={13} /></button>
              <div className="flex-1 grid grid-cols-5 gap-1" role="group" aria-label={t.panel.hiddenLayers}>
                {[1, 2, 3, 4, 5].map((n) => (
                  <button key={n} type="button" className="choice" aria-pressed={net.n_layers === n} onClick={() => setLayers(n)}>{n}</button>
                ))}
              </div>
              <button type="button" className="btn-secondary !p-1.5" aria-label={t.panel.addLayer} disabled={net.n_layers >= 5}
                onClick={() => setLayers(net.n_layers + 1)}><Plus size={13} /></button>
            </div>
          </Field>

          <div className="space-y-1.5">
            <div className="grid grid-cols-[34px_1fr_1fr] gap-1.5 text-[10.5px] uppercase tracking-wider" style={{ color: 'var(--text-faint)' }}>
              <span>{t.panel.layer}</span><span>{t.panel.neurons}</span><span>{t.panel.activation}</span>
            </div>
            {hidden.map((n, i) => (
              <div key={i} className="grid grid-cols-[34px_1fr_1fr] gap-1.5 items-center">
                <span className="text-[12px] font-mono" style={{ color: 'var(--text-muted)' }}>L{i + 1}</span>
                <select className="select-base !py-1 tnum" aria-label={t.panel.neuronsIn(i + 1)} value={n}
                  onChange={(e) => setNeurons(i, +e.target.value)}>
                  {NEURON_OPTIONS.filter((o) => o === n || o - n <= budgetLeft).map((o) => <option key={o} value={o}>{o}</option>)}
                </select>
                <select className="select-base !py-1" aria-label={t.panel.activationOf(i + 1)} value={net.activations[i]}
                  onChange={(e) => setActivation(i, e.target.value as ActivationType)}>
                  {ACTIVATIONS.map((a) => <option key={a} value={a}>{a}</option>)}
                </select>
              </div>
            ))}
            <p className="text-[11px]" style={{ color: 'var(--text-faint)' }}>
              {real ? t.panel.archReal(hidden.join(' → ')) : t.panel.archDiagram(hidden.join(' → '))}
              {' · '}{t.panel.budget} {NEURON_BUDGET - budgetLeft}/{NEURON_BUDGET}
            </p>
          </div>
        </Section>

        <Section title={t.panel.data}>
          <Field label={t.panel.dataset}>
            <Choices label={t.panel.dataset} options={DATASETS} value={train.dataset} onChange={(d) => setTrain({ dataset: d })} cols={4} />
          </Field>
          <Field label={t.panel.noise} value={`${train.noise}%`}>
            <input type="range" min={0} max={20} step={5} value={train.noise} aria-label={t.panel.noiseLabel}
              onChange={(e) => setTrain({ noise: +e.target.value })} />
          </Field>
          <DatasetPreview />
          <details className="text-[12px]" open={!!customDataset}>
            <summary className="cursor-pointer select-none" style={{ color: 'var(--text-muted)' }}>{t.panel.customCsv}</summary>
            <div className="mt-2">
              <DatasetUpload loaded={!!customDataset} onLoad={(ds) => setCustomDataset(ds)} onClear={() => setCustomDataset(null)} />
              <p className="mt-1.5 text-[11px] leading-relaxed" style={{ color: 'var(--text-faint)' }}>
                {t.panel.csvHelp}
              </p>
            </div>
          </details>
        </Section>

        <Section title={t.panel.training}>
          <Field label={t.panel.lr} value={train.learning_rate}>
            <Choices label={t.panel.lrLabel} options={LR_VALUES} value={train.learning_rate} onChange={(v) => setTrain({ learning_rate: v })} cols={5} />
          </Field>
          <Field label={t.panel.batch} value={train.batch_size}>
            <Choices label={t.panel.batch} options={BATCH_SIZES} value={train.batch_size} onChange={(v) => setTrain({ batch_size: v })} cols={6} />
          </Field>
          <Field label={t.panel.epochs} value={train.epochs}>
            <Choices label={t.panel.epochs} options={EPOCH_MARKS} value={train.epochs} onChange={(v) => setTrain({ epochs: v })} cols={6} />
          </Field>
          <Field label={t.panel.reg}>
            <div className="grid grid-cols-[1fr_88px] gap-1.5">
              <select className="select-base" aria-label={t.panel.regType} value={train.reg_type} onChange={(e) => setTrain({ reg_type: e.target.value as RegType })}>
                {REG_TYPES.map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
              <select className="select-base tnum" aria-label={t.panel.regRate} value={train.reg_rate} disabled={train.reg_type === 'None'}
                onChange={(e) => setTrain({ reg_rate: +e.target.value })}>
                {[0.0001, 0.001, 0.01].map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
            </div>
          </Field>
        </Section>
      </div>

      {/* Actions: always visible */}
      <div className="flex-shrink-0 border-t p-3 space-y-2" style={{ borderColor: 'var(--border)', background: 'var(--bg-sidebar)' }}>
        {dirty && (
          <p className="text-[11.5px] flex items-center gap-1.5" style={{ color: 'var(--text-warn)' }}>
            <RefreshCw size={12} />{t.panel.dirty}
          </p>
        )}
        <div className="grid grid-cols-2 gap-2">
          <button type="button" className="btn-primary !px-2" onClick={() => void build()} disabled={building} data-tour="build-btn">
            {building ? <Loader2 size={14} className="animate-spin" /> : <Zap size={14} />}
            {networkBuilt ? t.panel.rebuild : t.panel.build}
          </button>
          <button type="button" className="btn-secondary !px-2" onClick={() => void runTraining()} disabled={training || !networkBuilt}
            title={graphSource === 'model' ? t.panel.trainReal : t.panel.trainSynthetic}>
            {training ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
            {graphSource === 'model' || !networkBuilt ? t.panel.train(train.epochs) : t.panel.simulate}
          </button>
        </div>
        <StatusLine status={buildStatus} />
        <StatusLine status={trainStatus} />
      </div>
    </div>
  );
}
