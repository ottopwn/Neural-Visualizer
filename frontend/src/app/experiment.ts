// Experiment actions: build a model, train it.  Shared by the Experiment
// panel, the guided demos and presentation mode, so a demo runs exactly the
// same code path as a click in the sidebar.

import { create } from 'zustand';
import * as api from '../api/client';
import * as forgeApi from '../forge/api';
import { useForgeStore } from '../forge/store';
import { useNetworkStore } from '../store/networkStore';
import { tr } from '../i18n';
import type { BoundaryPair, SessionSummary } from '../forge/types';
import type { DecisionBoundaryData, TrainingResult } from '../types';

/** Model types whose graph is backed by a real Forge model session. */
export const FORGE_MODELS = new Set(['ANN']);

export type Status = { type: 'idle' | 'loading' | 'success' | 'error'; message?: string };

export function historyToResult(summary: SessionSummary): TrainingResult {
  return {
    epochs: summary.history.map((r) => r.epoch),
    loss_history: summary.history.map((r) => r.loss),
    accuracy_history: summary.history.map((r) => r.accuracy),
    source: 'pytorch',
  };
}

export function boundaryToData(b: BoundaryPair, summary: SessionSummary): DecisionBoundaryData {
  const r = b.resolution;
  const xs = Array.from({ length: r }, (_, i) => b.x_range[0] + ((b.x_range[1] - b.x_range[0]) * i) / (r - 1));
  const ys = Array.from({ length: r }, (_, i) => b.y_range[0] + ((b.y_range[1] - b.y_range[0]) * i) / (r - 1));
  return {
    xx: ys.map(() => xs),
    yy: ys.map((y) => xs.map(() => y)),
    zz: b.baseline,
    X: summary.dataset_X,
    y: summary.dataset_y,
    source: 'trained-session',
  };
}

/** Architecture the sidebar describes, trimmed to the active layer count. */
export function configuredArchitecture() {
  const c = useNetworkStore.getState().networkConfig;
  return {
    ...c,
    neurons: c.neurons.slice(0, c.n_layers),
    activations: c.activations.slice(0, c.n_layers),
  };
}

/**
 * Signature of what a Build would create, so the UI can tell when the
 * sidebar no longer matches the model on screen.
 */
export function configSignature(): string {
  const s = useNetworkStore.getState();
  const a = configuredArchitecture();
  return JSON.stringify([a.model_type, a.neurons, a.activations, s.trainingConfig.dataset, s.trainingConfig.noise, !!s.customDataset]);
}

interface ExperimentState {
  buildStatus: Status;
  trainStatus: Status;
  builtSignature: string | null;
  build: () => Promise<boolean>;
  train: (epochs?: number) => Promise<boolean>;
}

let statusTimer: ReturnType<typeof setTimeout> | null = null;

export const useExperiment = create<ExperimentState>((set, get) => {
  const settle = (key: 'buildStatus' | 'trainStatus', status: Status) => {
    set({ [key]: status } as Partial<ExperimentState>);
    if (status.type === 'success') {
      if (statusTimer) clearTimeout(statusTimer);
      statusTimer = setTimeout(() => set({ [key]: { type: 'idle' } } as Partial<ExperimentState>), 4000);
    }
  };

  return {
    buildStatus: { type: 'idle' },
    trainStatus: { type: 'idle' },
    builtSignature: null,

    build: async () => {
      if (get().buildStatus.type === 'loading') return false;
      const store = useNetworkStore.getState();
      set({ buildStatus: { type: 'loading' }, trainStatus: { type: 'idle' } });
      try {
        const config = configuredArchitecture();
        store.setTrainingResult(null);
        store.setDecisionBoundary(null);
        store.setPropStep(0);
        let message: string;

        if (FORGE_MODELS.has(config.model_type)) {
          // Real model: a PyTorch MLP every Forge instrument reads from.
          const custom = store.customDataset;
          const summary = await forgeApi.createSession({
            neurons: config.neurons,
            activations: config.activations,
            dataset: store.trainingConfig.dataset,
            noise: store.trainingConfig.noise,
            custom_dataset: custom ? { X: custom.X, y: custom.y } : null,
          });
          await useForgeStore.getState().setSession(summary);
          const err = useForgeStore.getState().error;
          if (err) throw new Error(err);
          store.setGraphSource('model');
          store.setTrainingResult(historyToResult(summary));
          message = tr().status.builtReal(summary.structure.param_count);
        } else {
          await useForgeStore.getState().setSession(null);
          const graph = await api.buildNetwork(config);
          store.setGraph(graph);
          const [fwd, bwd] = await Promise.all([api.getForwardProp(config), api.getBackwardProp(config)]);
          store.setForwardSteps(fwd.steps);
          store.setBackwardSteps(bwd.steps);
          store.setGraphSource('illustrative');
          message = tr().status.builtDiagram(config.model_type, graph.nodes.length);
        }
        store.setNetworkBuilt(true);
        set({ builtSignature: configSignature() });
        settle('buildStatus', { type: 'success', message });

        // Legacy illustrative landscape (random-init model) only for non-ANN diagrams;
        // the ANN landscape is computed from the session model in the Analysis view.
        if (!FORGE_MODELS.has(config.model_type)) {
          api.getLossLandscape(config).then(store.setLossLandscape).catch(() => undefined);
        }
        return true;
      } catch (err) {
        settle('buildStatus', { type: 'error', message: forgeApi.errorMessage(err) });
        return false;
      }
    },

    train: async (epochsOverride) => {
      if (get().trainStatus.type === 'loading') return false;
      const store = useNetworkStore.getState();
      const forge = useForgeStore.getState();
      set({ trainStatus: { type: 'loading' }, buildStatus: { type: 'idle' } });
      if (forge.session && store.graphSource === 'model') {
        try {
          const t = store.trainingConfig;
          const { summary, new_rows } = await forgeApi.trainSession(forge.session.session_id, {
            epochs: epochsOverride ?? t.epochs, learning_rate: t.learning_rate, batch_size: t.batch_size,
            reg_type: t.reg_type, reg_rate: t.reg_rate,
          });
          await useForgeStore.getState().updateSession(summary);
          store.setTrainingResult(historyToResult(summary));
          const boundary = useForgeStore.getState().comparison?.boundary;
          if (boundary) store.setDecisionBoundary(boundaryToData(boundary, summary));
          const last = new_rows[new_rows.length - 1];
          settle('trainStatus', { type: 'success', message: tr().status.trained(summary.epoch, last.accuracy) });
          return true;
        } catch (err) {
          settle('trainStatus', { type: 'error', message: forgeApi.errorMessage(err) });
          return false;
        }
      }
      try {
        const netConfig = configuredArchitecture();
        const [result, boundary] = await Promise.all([
          api.simulateTraining(netConfig, store.trainingConfig),
          api.getDecisionBoundary(netConfig, store.trainingConfig),
        ]);
        store.setTrainingResult(result);
        store.setDecisionBoundary(boundary);
        settle('trainStatus', { type: 'success', message: tr().status.simulated(result.epochs.length) });
        return true;
      } catch (err) {
        settle('trainStatus', { type: 'error', message: forgeApi.errorMessage(err) });
        return false;
      }
    },
  };
});
