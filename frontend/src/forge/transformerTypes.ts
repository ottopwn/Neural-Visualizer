// Mirror of backend/forge/transformer.py (Transformer Lab wire contract).

export interface TransformerInfo {
  description: string;
  d_model: number;
  n_heads: number;
  n_layers: number;
  d_head: number;
  d_mlp: number;
  max_len: number;
  vocab: string[];
  param_count: number;
  corpus_size: number;
  corpus_examples: string[];
  initial_loss: number;
  final_loss: number;
  train_steps: number;
  train_seconds: number;
  seed: number;
  examples: string[];
}

export interface TokenOut { id: number; text: string; unknown: boolean }

export interface HeadTrace {
  head: number;
  q: number[][];
  k: number[][];
  v: number[][];
  scores: number[][]; // QKᵀ·scale, before the causal mask
  attn: number[][]; // rows = query (destination) positions, cols = key (source) positions
  out: number[][];
  ablated: boolean;
}

export interface LayerTrace {
  layer: number;
  resid_pre: number[][];
  ln1: number[][];
  heads: HeadTrace[];
  attn_out: number[][];
  resid_mid: number[][];
  ln2: number[][];
  mlp_hidden_norm: number[];
  mlp_active_fraction: number[];
  mlp_out: number[][];
  resid_post: number[][];
  norms: Record<'resid_pre' | 'attn_out' | 'mlp_out' | 'resid_post', number[]>;
}

export interface Candidate { token: string; prob: number; logit: number }

export interface TransformerTrace {
  tokens: TokenOut[];
  truncated: boolean;
  scale: number;
  tok_emb: number[][];
  pos_emb: number[][];
  embed: number[][];
  layers: LayerTrace[];
  final_ln: number[][];
  logits_last: number[];
  next_token: Candidate[];
  next_token_baseline: Candidate[] | null;
  per_position_top: Candidate[];
  ablate_heads: [number, number][];
}
