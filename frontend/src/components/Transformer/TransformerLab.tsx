import { AlertCircle, Brain, Loader2, Play, RotateCcw } from 'lucide-react';
import { useEffect, useMemo, type ReactNode } from 'react';
import { diverging, fmt, maxAbs, pct } from '../../forge/format';
import { useForgeStore } from '../../forge/store';
import { useTransformerLab, type LabStage } from '../../forge/transformerLab';
import type { Candidate, HeadTrace, TransformerTrace } from '../../forge/transformerTypes';
import { HeatmapCanvas } from '../Forge/charts';
import { useNeutral } from '../Forge/TimeMachine/hooks';

const STAGES: { id: LabStage; label: string }[] = [
  { id: 'tokens', label: 'Tokens' },
  { id: 'embeddings', label: 'Embeddings' },
  { id: 'attention', label: 'Attention' },
  { id: 'block', label: 'Block' },
  { id: 'next', label: 'Next token' },
];

function P({ children }: { children: ReactNode }) {
  return <p className="text-[13px] leading-relaxed m-0" style={{ color: 'var(--text-primary)' }}>{children}</p>;
}

function Eq({ children }: { children: ReactNode }) {
  return (
    <div className="font-mono text-[12.5px] px-3 py-2 rounded-md border overflow-x-auto whitespace-nowrap tnum"
      style={{ borderColor: 'var(--border)', background: 'var(--bg-base)', color: 'var(--text-primary)' }}>{children}</div>
  );
}

function Title({ children, hint }: { children: ReactNode; hint?: ReactNode }) {
  return (
    <div className="flex items-baseline gap-2 flex-wrap">
      <h3 className="text-[13px] font-semibold m-0" style={{ color: 'var(--text-primary)' }}>{children}</h3>
      {hint && <span className="text-[11.5px]" style={{ color: 'var(--text-faint)' }}>{hint}</span>}
    </div>
  );
}

/** A small token-indexed matrix as a heatmap with row labels. */
function TokenMatrix({ m, tokens, label, height }: { m: number[][]; tokens: string[]; label: string; height?: number }) {
  const neutral = useNeutral();
  const mx = maxAbs(m.flat()) || 1;
  const color = useMemo(() => (v: number) => diverging(v, mx, neutral), [mx, neutral]);
  const h = height ?? Math.max(60, Math.min(260, tokens.length * 16));
  return (
    <div className="grid gap-1.5" style={{ gridTemplateColumns: '72px 1fr' }}>
      <div className="flex flex-col justify-around text-[11px] font-mono text-right pr-1" style={{ height: h, color: 'var(--text-muted)' }}>
        {tokens.map((t, i) => <span key={i} className="truncate leading-none">{t}</span>)}
      </div>
      <HeatmapCanvas values={m} color={color} height={h} flipY ariaLabel={`${label}: one row per token, one column per dimension`} />
    </div>
  );
}

/** Attention matrix: rows = destination (query) tokens, columns = source (key) tokens. */
function AttentionMatrix({ h, tokens, query, keyIdx, onPick }: {
  h: HeadTrace; tokens: string[]; query: number | null; keyIdx: number | null; onPick: (q: number, k: number) => void;
}) {
  const n = tokens.length;
  const cell = Math.max(16, Math.min(34, Math.floor(420 / n)));
  const lab = 74;
  const W = lab + n * cell;
  const H = lab + n * cell;
  return (
    <svg width={W} height={H} role="img" aria-label="Attention weights: rows are destination tokens, columns are source tokens" className="max-w-full">
      {tokens.map((t, j) => (
        <text key={`c${j}`} x={lab + j * cell + cell / 2} y={lab - 6} fontSize={10.5} textAnchor="start"
          transform={`rotate(-50 ${lab + j * cell + cell / 2} ${lab - 6})`} className="font-mono"
          style={{ fill: keyIdx === j ? 'var(--text-primary)' : 'var(--text-muted)' }}>{t}</text>
      ))}
      {tokens.map((t, i) => (
        <text key={`r${i}`} x={lab - 6} y={lab + i * cell + cell / 2 + 3.5} fontSize={10.5} textAnchor="end" className="font-mono"
          style={{ fill: query === i ? 'var(--text-primary)' : 'var(--text-muted)' }}>{t}</text>
      ))}
      {h.attn.map((row, i) => row.map((a, j) => {
        const masked = j > i;
        const sel = query === i && keyIdx === j;
        return (
          <g key={`${i}-${j}`} onClick={() => !masked && onPick(i, j)} style={{ cursor: masked ? 'not-allowed' : 'pointer' }}>
            <rect x={lab + j * cell} y={lab + i * cell} width={cell - 1} height={cell - 1} rx={2}
              style={{ fill: masked ? 'var(--bg-raised)' : 'var(--accent)', fillOpacity: masked ? 1 : 0.06 + 0.94 * a, stroke: sel ? 'var(--select)' : query === i ? 'var(--border-soft)' : 'none' }}
              strokeWidth={sel ? 2 : 1} />
            {masked && <line x1={lab + j * cell + 3} y1={lab + i * cell + cell - 4} x2={lab + j * cell + cell - 4} y2={lab + i * cell + 3} style={{ stroke: 'var(--border-soft)' }} />}
            {!masked && cell >= 26 && a >= 0.05 && (
              <text x={lab + j * cell + cell / 2} y={lab + i * cell + cell / 2 + 3.5} fontSize={9} textAnchor="middle" className="tnum"
                style={{ fill: a > 0.55 ? '#fff' : 'var(--text-primary)', pointerEvents: 'none' }}>{a.toFixed(2)}</text>
            )}
            <title>{masked ? `${tokens[i]} cannot attend to the future token ${tokens[j]} (causal mask)` : `${tokens[i]} → ${tokens[j]}: weight ${a.toFixed(4)}, score ${h.scores[i][j].toFixed(4)}`}</title>
          </g>
        );
      }))}
    </svg>
  );
}

function Bars({ items, compare, label }: { items: Candidate[]; compare?: Candidate[] | null; label: string }) {
  const cmp = new Map((compare ?? []).map((c) => [c.token, c.prob]));
  return (
    <div className="space-y-1" role="list" aria-label={label}>
      {items.map((c) => (
        <div key={c.token} role="listitem" className="grid items-center gap-2 text-[12.5px]" style={{ gridTemplateColumns: '84px 1fr 56px 64px' }}>
          <span className="font-mono truncate" style={{ color: 'var(--text-primary)' }}>{c.token}</span>
          <span className="h-3 rounded-sm relative" style={{ background: 'var(--border)' }}>
            <span className="absolute inset-y-0 left-0 rounded-sm" style={{ width: `${c.prob * 100}%`, background: 'var(--accent)' }} />
            {cmp.has(c.token) && <span className="absolute -inset-y-0.5 w-0.5" style={{ left: `${cmp.get(c.token)! * 100}%`, background: 'var(--text-warn)' }} title={`before: ${pct(cmp.get(c.token)!)}`} />}
          </span>
          <span className="font-mono tnum text-right">{pct(c.prob)}</span>
          <span className="font-mono tnum text-right text-[11px]" style={{ color: 'var(--text-faint)' }}>{fmt(c.logit, 2)}</span>
        </div>
      ))}
    </div>
  );
}

// ── stages ──────────────────────────────────────────────────────────────────

function TokensStage({ t, lab }: { t: TransformerTrace; lab: boolean }) {
  const unknown = t.tokens.filter((x) => x.unknown).length;
  return (
    <div className="space-y-3">
      <P>
        The text is split into words from a fixed vocabulary of the training corpus; each word becomes an integer id. A <span className="font-mono">&lt;bos&gt;</span> token
        marks the start. {unknown > 0 ? <>{unknown} word{unknown > 1 ? 's are' : ' is'} not in the vocabulary and became <span className="font-mono">&lt;unk&gt;</span> — the model knows nothing about {unknown > 1 ? 'them' : 'it'}.</> : 'Every word is in the vocabulary.'}
        {t.truncated && ' The input was longer than the context window, so only the last tokens are used.'}
      </P>
      <table className="text-[12.5px] tnum border-collapse">
        <thead><tr style={{ color: 'var(--text-faint)' }}><th className="text-left pr-4 font-medium">position</th><th className="text-left pr-4 font-medium">token</th><th className="text-right font-medium">id</th></tr></thead>
        <tbody>
          {t.tokens.map((x, i) => (
            <tr key={i}>
              <td className="pr-4 font-mono" style={{ color: 'var(--text-muted)' }}>{i}</td>
              <td className="pr-4 font-mono" style={{ color: x.unknown ? 'var(--text-warn)' : 'var(--text-primary)' }}>{x.text}{x.unknown ? ' → <unk>' : ''}</td>
              <td className="text-right font-mono">{x.id}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {lab && <Eq>ids ∈ ℕ^{t.tokens.length} · context window {t.tokens.length}{t.truncated ? ' (truncated)' : ''}</Eq>}
    </div>
  );
}

function EmbeddingsStage({ t, lab, d }: { t: TransformerTrace; lab: boolean; d: number }) {
  const names = t.tokens.map((x) => x.text);
  return (
    <div className="space-y-3">
      <P>
        Each token id selects a learned vector (its <b>token embedding</b>); a second learned vector encodes its <b>position</b>. Their sum is the
        starting point of the <b>residual stream</b> that every layer reads from and writes to.
      </P>
      {lab && <Eq>x₀ = E[id] + P[pos]   E: V×{d}, P: T×{d} → x₀: {t.tokens.length}×{d}</Eq>}
      <div className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))' }}>
        <div className="space-y-1"><Title hint={`${d} dims`}>Token embedding</Title><TokenMatrix m={t.tok_emb} tokens={names} label="Token embeddings" /></div>
        <div className="space-y-1"><Title>Position embedding</Title><TokenMatrix m={t.pos_emb} tokens={names} label="Position embeddings" /></div>
        <div className="space-y-1"><Title>Sum = residual stream x₀</Title><TokenMatrix m={t.embed} tokens={names} label="Embedding sum" /></div>
      </div>
    </div>
  );
}

function AttentionStage({ t, lab }: { t: TransformerTrace; lab: boolean }) {
  const { layer, head, query, key, setHead, selectCell } = useTransformerLab();
  const L = t.layers[layer];
  const h = L.heads[head];
  const names = t.tokens.map((x) => x.text);
  const qi = query ?? names.length - 1;
  const row = h.attn[qi];
  const ranked = row.map((a, j) => ({ j, a })).filter((x) => x.j <= qi).sort((a, b) => b.a - a.a);
  const kj = key !== null && key <= qi ? key : ranked[0].j;
  const dot = h.q[qi].reduce((s, v, i) => s + v * h.k[kj][i], 0);
  const exps = row.slice(0, qi + 1).map((_, j) => Math.exp(h.scores[qi][j] - Math.max(...h.scores[qi].slice(0, qi + 1))));
  const z = exps.reduce((a, b) => a + b, 0);
  const dh = h.q[0].length;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 flex-wrap">
        {t.layers.map((l) => l.heads.map((hh) => (
          <button key={`${l.layer}-${hh.head}`} type="button" aria-pressed={layer === l.layer && head === hh.head}
            onClick={() => setHead(l.layer, hh.head)} className="choice !px-3"
            title={hh.ablated ? 'This head is ablated (output zeroed)' : undefined}>
            Layer {l.layer} · Head {hh.head}{hh.ablated ? ' (off)' : ''}
          </button>
        )))}
      </div>
      <P>
        Each position builds a <b>query</b> (what it looks for) and every earlier position offers a <b>key</b> (what it contains) and a <b>value</b>
        (what it passes on). The score q·k, scaled by 1/√{dh}, becomes a weight after a softmax over the positions it is allowed to see — the
        <b> causal mask</b> hides the future. Rows are the token that is reading; columns are the tokens it reads from.
        {h.ablated && <b style={{ color: 'var(--text-warn)' }}> This head is ablated: its weights are still computed, but its output is zeroed.</b>}
      </P>
      <div className="flex gap-5 flex-wrap items-start">
        <div className="overflow-x-auto" data-testid="attention-matrix">
          <AttentionMatrix h={h} tokens={names} query={qi} keyIdx={kj} onPick={(q, k) => selectCell(q, k)} />
        </div>
        <div className="flex-1 min-w-[260px] space-y-2">
          <Title hint="click a row label or a cell">“{names[qi]}” (position {qi}) attends to</Title>
          <div className="space-y-1">
            {ranked.slice(0, 8).map(({ j, a }) => (
              <button key={j} type="button" onClick={() => selectCell(qi, j)}
                className="w-full grid items-center gap-2 text-[12.5px] px-1 py-0.5 rounded hover:bg-[var(--bg-hover)]"
                style={{ gridTemplateColumns: '90px 1fr 52px', background: j === kj ? 'var(--bg-active)' : undefined }}>
                <span className="font-mono truncate text-left">{names[j]} <span style={{ color: 'var(--text-faint)' }}>@{j}</span></span>
                <span className="h-2.5 rounded-sm" style={{ background: 'var(--border)' }}><span className="block h-full rounded-sm" style={{ width: `${a * 100}%`, background: 'var(--accent)' }} /></span>
                <span className="font-mono tnum text-right">{a.toFixed(3)}</span>
              </button>
            ))}
          </div>
          <div className="flex gap-1 flex-wrap" aria-label="Choose the reading position">
            {names.map((n, i) => (
              <button key={i} type="button" className="choice !py-0.5 !px-1.5 font-mono" aria-pressed={i === qi} onClick={() => selectCell(i, null)}>{n}</button>
            ))}
          </div>
        </div>
      </div>

      <div className="rounded-md border p-3 space-y-2" style={{ borderColor: 'var(--border)' }}>
        <Title hint={`layer ${layer}, head ${head}`}>{names[qi]} → {names[kj]}</Title>
        <Eq>
          score = (q·k) × 1/√{dh} = {fmt(dot, 4)} × {fmt(t.scale, 4)} = <b>{fmt(h.scores[qi][kj], 4)}</b>   →   softmax: exp({fmt(h.scores[qi][kj] - Math.max(...h.scores[qi].slice(0, qi + 1)), 3)}) / {fmt(z, 4)} = <b>{fmt(h.attn[qi][kj], 4)}</b>
        </Eq>
        {lab && (
          <div className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
            <div className="space-y-1"><Title hint={`[${names.length}×${dh}]`}>Q</Title><TokenMatrix m={h.q} tokens={names} label="Queries" /></div>
            <div className="space-y-1"><Title hint={`[${names.length}×${dh}]`}>K</Title><TokenMatrix m={h.k} tokens={names} label="Keys" /></div>
            <div className="space-y-1"><Title hint={`[${names.length}×${dh}]`}>V</Title><TokenMatrix m={h.v} tokens={names} label="Values" /></div>
            <div className="space-y-1"><Title hint="Σ_j A[i,j]·V[j]">Head output A·V</Title><TokenMatrix m={h.out} tokens={names} label="Head output" /></div>
          </div>
        )}
        {lab && <Eq>Attention(Q,K,V) = softmax(QKᵀ/√{dh} + M)·V,  M[i,j] = −∞ for j &gt; i  ·  heads concatenated → W_O → added to the residual stream</Eq>}
      </div>
    </div>
  );
}

function BlockStage({ t, lab }: { t: TransformerTrace; lab: boolean }) {
  const names = t.tokens.map((x) => x.text);
  const keys = ['resid_pre', 'attn_out', 'mlp_out', 'resid_post'] as const;
  const labels: Record<(typeof keys)[number], string> = { resid_pre: 'stream in', attn_out: '+ attention', mlp_out: '+ MLP', resid_post: 'stream out' };
  return (
    <div className="space-y-3">
      <P>
        Each block reads the residual stream through a <b>LayerNorm</b>, adds the attention output, then reads again (second LayerNorm) and adds the
        output of a small <b>MLP</b> (32 → 128 → 32, GELU). The stream is a running sum: nothing is overwritten. The bars show the real size (L2 norm)
        of each piece at every position.
      </P>
      {lab && <Eq>x ← x + W_O·Attn(LN₁(x));   x ← x + W₂·GELU(W₁·LN₂(x) + b₁) + b₂;   logits = LN_f(x)·W_U</Eq>}
      {t.layers.map((L) => {
        const m = Math.max(...keys.flatMap((k) => L.norms[k]));
        return (
          <div key={L.layer} className="rounded-md border p-3" style={{ borderColor: 'var(--border)' }}>
            <Title hint="L2 norm per position">Block {L.layer}</Title>
            <div className="overflow-x-auto mt-2">
              <table className="text-[12px] tnum border-collapse w-full">
                <thead><tr style={{ color: 'var(--text-faint)' }}><th className="text-left font-medium pr-2">token</th>{keys.map((k) => <th key={k} className="text-left font-medium px-2">{labels[k]}</th>)}<th className="text-right font-medium pl-2">MLP units &gt; 0</th></tr></thead>
                <tbody>
                  {names.map((n, i) => (
                    <tr key={i}>
                      <td className="font-mono pr-2" style={{ color: 'var(--text-muted)' }}>{n}</td>
                      {keys.map((k) => (
                        <td key={k} className="px-2 py-[2px]">
                          <div className="flex items-center gap-1.5">
                            <span className="h-2 rounded-sm inline-block" style={{ width: `${(L.norms[k][i] / m) * 70}px`, background: k === 'attn_out' ? 'var(--tm-a)' : k === 'mlp_out' ? 'var(--tm-b)' : 'var(--text-faint)' }} />
                            <span className="font-mono text-[11px]">{L.norms[k][i].toFixed(2)}</span>
                          </div>
                        </td>
                      ))}
                      <td className="text-right font-mono pl-2">{pct(L.mlp_active_fraction[i], 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function NextStage({ t, lab, nHeads, nLayers }: { t: TransformerTrace; lab: boolean; nHeads: number; nLayers: number }) {
  const { ablate, toggleAblate, clearAblate, loading } = useTransformerLab();
  const names = t.tokens.map((x) => x.text);
  const top = t.next_token[0];
  const before = t.next_token_baseline?.[0];
  return (
    <div className="space-y-3">
      <P>
        The final residual vector at the last position (“{names[names.length - 1]}”) is normalised and multiplied by the unembedding matrix: one
        <b> logit</b> per vocabulary word. Softmax turns them into the next-token distribution. The model’s top guess is <b className="font-mono">{top.token}</b> at {pct(top.prob)}.
        {before && <> Without the ablated heads it was <b className="font-mono">{before.token}</b> at {pct(before.prob)}.</>}
      </P>
      <div className="grid gap-5" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
        <div className="space-y-2">
          <Title hint="top 10 · logit on the right">Next-token probabilities{ablate.length ? ' (with ablation)' : ''}</Title>
          <Bars items={t.next_token} compare={t.next_token_baseline} label="Next-token probabilities" />
          {t.next_token_baseline && <p className="text-[11.5px] m-0" style={{ color: 'var(--text-faint)' }}>Orange tick = probability before the ablation.</p>}
        </div>
        <div className="space-y-2">
          <Title hint="real what-if: zero a head's output">Ablate attention heads</Title>
          <div className="grid gap-1.5" style={{ gridTemplateColumns: `repeat(${nHeads}, minmax(0, 1fr))` }}>
            {Array.from({ length: nLayers }, (_, l) => Array.from({ length: nHeads }, (__, h) => {
              const on = ablate.some(([a, b]) => a === l && b === h);
              return (
                <button key={`${l}-${h}`} type="button" aria-pressed={on} disabled={loading} onClick={() => void toggleAblate(l, h)}
                  className="choice !py-2" style={on ? { borderColor: 'var(--warn)', color: 'var(--text-warn)', background: 'color-mix(in srgb, var(--warn) 10%, transparent)' } : undefined}>
                  L{l} · H{h} {on ? 'off' : 'on'}
                </button>
              );
            }))}
          </div>
          {ablate.length > 0 && <button type="button" className="btn-secondary !py-1 !text-xs" onClick={() => void clearAblate()}><RotateCcw size={12} />Restore all heads</button>}
          <Title hint="argmax at each position">What it predicts after each token</Title>
          <div className="flex flex-wrap gap-1.5">
            {names.map((n, i) => (
              <span key={i} className="text-[12px] font-mono px-1.5 py-0.5 rounded border" style={{ borderColor: 'var(--border)' }}>
                {n} <span style={{ color: 'var(--text-faint)' }}>→</span> <b>{t.per_position_top[i].token}</b> <span className="tnum" style={{ color: 'var(--text-faint)' }}>{pct(t.per_position_top[i].prob, 0)}</span>
              </span>
            ))}
          </div>
        </div>
      </div>
      {lab && <Eq>p(next) = softmax(LN_f(x_last)·W_U + b_U)   ·   |V| = {t.logits_last.length}</Eq>}
    </div>
  );
}

export function TransformerLab() {
  const lab = useForgeStore((s) => s.mode === 'lab');
  const { info, infoLoading, text, trace, loading, error, stage, loadInfo, setText, run, setStage } = useTransformerLab();
  useEffect(() => { void loadInfo(); }, [loadInfo]);
  useEffect(() => { if (info && !trace && !loading) void run(); }, [info, trace, loading, run]);

  return (
    <div className="h-full flex flex-col min-h-0">
      <div className="flex items-center gap-3 flex-wrap px-3 py-2 border-b flex-shrink-0" style={{ borderColor: 'var(--border)', background: 'var(--bg-sidebar)' }}>
        <span className="badge-green">REAL · TINY MODEL</span>
        {info ? (
          <span className="text-[12px] tnum" style={{ color: 'var(--text-muted)' }}>
            {info.n_layers} layers × {info.n_heads} heads · d = {info.d_model} · {info.param_count.toLocaleString()} parameters · vocabulary {info.vocab.length} words ·
            trained here in {info.train_seconds.toFixed(1)} s ({info.train_steps} steps, loss {info.initial_loss.toFixed(2)} → {info.final_loss.toFixed(2)})
          </span>
        ) : <span className="text-[12px]" style={{ color: 'var(--text-muted)' }}>{infoLoading ? 'Training the lab Transformer on this machine (once per backend start)…' : ''}</span>}
        <span className="text-[11.5px] ml-auto" style={{ color: 'var(--text-faint)' }}>Not GPT, ChatGPT or Claude: a transparent toy model on template sentences.</span>
      </div>

      <form className="flex items-center gap-2 flex-wrap px-3 py-2.5 border-b flex-shrink-0" style={{ borderColor: 'var(--border)' }}
        onSubmit={(e) => { e.preventDefault(); void run(); }}>
        <label className="sr-only" htmlFor="tl-input">Input text</label>
        <input id="tl-input" className="input-base !w-auto flex-1 min-w-[260px] font-mono" value={text} maxLength={300}
          onChange={(e) => setText(e.target.value)} placeholder="the cat sat on the" />
        <button type="submit" className="btn-primary" disabled={!info || loading}>{loading ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}Run</button>
        {info?.examples.map((ex) => (
          <button key={ex} type="button" className="choice !px-2 font-mono" onClick={() => void run(ex)} disabled={loading}>{ex}</button>
        ))}
      </form>

      {trace && (
        <div className="flex items-center gap-1 px-3 py-2 border-b flex-wrap flex-shrink-0" style={{ borderColor: 'var(--border)' }} aria-label="Tokens" data-testid="token-strip">
          {trace.tokens.map((x, i) => (
            <span key={i} className="text-[12.5px] font-mono px-1.5 py-0.5 rounded border"
              style={{ borderColor: x.unknown ? 'var(--warn)' : 'var(--border-soft)', color: x.unknown ? 'var(--text-warn)' : 'var(--text-primary)', background: 'var(--bg-card)' }}
              title={`position ${i} · id ${x.id}${x.unknown ? ' · not in the vocabulary (<unk>)' : ''}`}>
              {x.text}
            </span>
          ))}
          <span className="text-[12px] ml-1" style={{ color: 'var(--text-faint)' }}>→ <b className="font-mono" style={{ color: 'var(--text-primary)' }}>{trace.next_token[0].token}</b> {pct(trace.next_token[0].prob)}</span>
        </div>
      )}

      <nav className="flex items-center gap-1 px-3 py-2 border-b flex-shrink-0" style={{ borderColor: 'var(--border)' }} role="tablist" aria-label="Transformer stages">
        {STAGES.map((s, i) => (
          <span key={s.id} className="flex items-center gap-1">
            {i > 0 && <span aria-hidden="true" style={{ color: 'var(--text-faint)' }}>›</span>}
            <button type="button" role="tab" aria-selected={stage === s.id} onClick={() => setStage(s.id)}
              className="px-2.5 py-1 rounded-md text-[12.5px] border"
              style={stage === s.id ? { background: 'var(--accent)', color: '#fff', borderColor: 'var(--accent)' } : { borderColor: 'var(--border)', color: 'var(--text-muted)' }}>
              {s.label}
            </button>
          </span>
        ))}
      </nav>

      <div className="flex-1 min-h-0 overflow-y-auto p-4" style={{ opacity: loading ? 0.6 : 1, transition: 'opacity 0.15s' }}>
        {error && <p className="flex items-center gap-1.5 text-[13px] mb-3" role="alert" style={{ color: 'var(--text-neg)' }}><AlertCircle size={14} />{error}</p>}
        {!trace && !error && (
          <div className="h-full flex flex-col items-center justify-center gap-2 text-center" style={{ color: 'var(--text-muted)' }}>
            <Brain size={22} style={{ color: 'var(--accent)' }} />
            <p className="text-sm">{infoLoading ? 'Training a 2-layer Transformer locally — a few seconds, only once.' : 'Loading…'}</p>
            <Loader2 size={16} className="animate-spin" />
          </div>
        )}
        {trace && info && (
          <>
            {stage === 'tokens' && <TokensStage t={trace} lab={lab} />}
            {stage === 'embeddings' && <EmbeddingsStage t={trace} lab={lab} d={info.d_model} />}
            {stage === 'attention' && <AttentionStage t={trace} lab={lab} />}
            {stage === 'block' && <BlockStage t={trace} lab={lab} />}
            {stage === 'next' && <NextStage t={trace} lab={lab} nHeads={info.n_heads} nLayers={info.n_layers} />}
          </>
        )}
      </div>
    </div>
  );
}
