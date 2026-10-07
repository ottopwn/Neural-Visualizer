// Scalar activation functions, used only to *draw* the activation curve in
// the microscope.  They must match backend/forge/mlp.py exactly; the values
// shown as data always come from the backend.

export const LEAKY_RELU_SLOPE = 0.2;
const SELU_ALPHA = 1.6732632423543772;
const SELU_SCALE = 1.0507009873554805;

export function applyActivation(name: string, z: number): number | null {
  switch (name) {
    case 'ReLU': return Math.max(0, z);
    case 'Sigmoid': return 1 / (1 + Math.exp(-z));
    case 'Tanh': return Math.tanh(z);
    case 'LeakyReLU': return z > 0 ? z : LEAKY_RELU_SLOPE * z;
    case 'ELU': return z > 0 ? z : Math.exp(z) - 1;
    case 'SELU': return SELU_SCALE * (z > 0 ? z : SELU_ALPHA * (Math.exp(z) - 1));
    default: return null; // Softmax depends on the whole layer
  }
}

export function sampleCurve(name: string, lo: number, hi: number, n = 80): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const z = lo + ((hi - lo) * i) / (n - 1);
    const a = applyActivation(name, z);
    if (a !== null) pts.push([z, a]);
  }
  return pts;
}
