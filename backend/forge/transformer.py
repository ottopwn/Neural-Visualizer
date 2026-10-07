"""Transformer Lab: a tiny, real, fully inspectable decoder-only Transformer.

What this is
------------
A 2-layer, 2-head causal Transformer (d_model = 32, learned positions,
pre-LayerNorm, GELU MLP) trained *locally* on a small synthetic corpus of
simple English sentences generated from templates.  Training runs once, on
the CPU, the first time the lab is used (a few seconds), with a fixed seed.

What this is not
----------------
It is not GPT, ChatGPT, Claude or any production language model, and it does
not approximate their internals.  Its only purpose is transparency: every
tensor of its forward pass -- embeddings, Q, K, V, scaled scores, causal
mask, softmax attention, head outputs, residual stream, LayerNorm, MLP,
logits -- is returned exactly as computed.

Like ``mlp.py`` the model is functional (explicit parameter dict), so the
same forward pass serves training, tracing and head-ablation "what-if"
experiments without hooks or mutation.
"""

from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

# ── corpus & tokenizer ───────────────────────────────────────────────────────

SUBJECTS = ["the cat", "the dog", "a bird", "the girl", "the boy", "my friend", "the teacher", "a fox"]
VERBS = ["sat on", "ran to", "looked at", "jumped over", "walked to", "slept near"]
OBJECTS = ["the mat", "the park", "the tree", "the river", "the house", "the box", "the hill"]
ENDINGS = ["", " today", " again", " at night"]

SPECIAL = ["<pad>", "<unk>", "<bos>"]
PAD, UNK, BOS = 0, 1, 2


def build_corpus(seed: int = 0, n: int = 1200) -> List[str]:
    """Deterministic template sentences, e.g. 'the cat sat on the mat today .'"""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)}{rng.choice(ENDINGS)} ."
        if rng.random() < 0.35:  # a second clause referring back to the subject
            s += " then it " + rng.choice(VERBS) + " " + rng.choice(OBJECTS) + " ."
        out.append(s)
    return out


class Vocab:
    def __init__(self, corpus: Sequence[str]) -> None:
        words = sorted({w for line in corpus for w in line.split()})
        self.itos: List[str] = SPECIAL + words
        self.stoi: Dict[str, int] = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> Tuple[List[int], List[str], List[bool]]:
        """Lower-cases, splits on whitespace and before '.', maps unknown words to <unk>."""
        words = text.lower().replace(".", " . ").split()
        ids = [self.stoi.get(w, UNK) for w in words]
        return ids, words, [i == UNK for i in ids]


# ── model ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TConfig:
    vocab_size: int
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 2
    d_mlp: int = 128
    max_len: int = 24

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


Params = Dict[str, torch.Tensor]


def init_params(cfg: TConfig, seed: int = 0) -> Params:
    g = torch.Generator().manual_seed(seed)
    d, V = cfg.d_model, cfg.vocab_size

    def lin(i: int, o: int) -> torch.Tensor:
        return torch.randn(i, o, generator=g) / math.sqrt(i)

    p: Params = {
        "tok_emb": torch.randn(V, d, generator=g) * 0.5,
        "pos_emb": torch.randn(cfg.max_len, d, generator=g) * 0.2,
        "lnf_g": torch.ones(d), "lnf_b": torch.zeros(d),
        "W_U": lin(d, V), "b_U": torch.zeros(V),
    }
    for layer in range(cfg.n_layers):
        pre = f"l{layer}."
        p.update({
            pre + "ln1_g": torch.ones(d), pre + "ln1_b": torch.zeros(d),
            pre + "W_Q": lin(d, d), pre + "W_K": lin(d, d), pre + "W_V": lin(d, d),
            pre + "W_O": lin(d, d), pre + "b_O": torch.zeros(d),
            pre + "ln2_g": torch.ones(d), pre + "ln2_b": torch.zeros(d),
            pre + "W_1": lin(d, cfg.d_mlp), pre + "b_1": torch.zeros(cfg.d_mlp),
            pre + "W_2": lin(cfg.d_mlp, d), pre + "b_2": torch.zeros(d),
        })
    return p


def layer_norm(x: torch.Tensor, g: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), g, b, eps)


def forward(cfg: TConfig, p: Params, ids: torch.Tensor, ablate: Sequence[Tuple[int, int]] = (),
            record: Optional[dict] = None) -> torch.Tensor:
    """ids: [B, n] -> logits [B, n, V].  ``ablate`` zeroes (layer, head) outputs.

    When ``record`` is a dict, every intermediate tensor is stored in it.
    """
    B, n = ids.shape
    h, dh = cfg.n_heads, cfg.d_head
    tok = p["tok_emb"][ids]
    pos = p["pos_emb"][:n].unsqueeze(0).expand(B, n, -1)
    x = tok + pos
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)  # True = future (masked)
    if record is not None:
        record.update(tok=tok, pos=pos, embed=x, mask=mask, layers=[])
    for layer in range(cfg.n_layers):
        pre = f"l{layer}."
        x_ln = layer_norm(x, p[pre + "ln1_g"], p[pre + "ln1_b"])
        q = (x_ln @ p[pre + "W_Q"]).view(B, n, h, dh).transpose(1, 2)  # [B, h, n, dh]
        k = (x_ln @ p[pre + "W_K"]).view(B, n, h, dh).transpose(1, 2)
        v = (x_ln @ p[pre + "W_V"]).view(B, n, h, dh).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(dh)  # [B, h, n, n]
        masked = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(masked, dim=-1)
        z = attn @ v  # [B, h, n, dh]
        head_mask = torch.ones(h)
        for lyr, hd in ablate:
            if lyr == layer:
                head_mask[hd] = 0.0
        z = z * head_mask.view(1, h, 1, 1)
        concat = z.transpose(1, 2).reshape(B, n, h * dh)
        attn_out = concat @ p[pre + "W_O"] + p[pre + "b_O"]
        resid_mid = x + attn_out
        m_ln = layer_norm(resid_mid, p[pre + "ln2_g"], p[pre + "ln2_b"])
        hidden = F.gelu(m_ln @ p[pre + "W_1"] + p[pre + "b_1"])
        mlp_out = hidden @ p[pre + "W_2"] + p[pre + "b_2"]
        resid_post = resid_mid + mlp_out
        if record is not None:
            record["layers"].append(dict(
                resid_pre=x, ln1=x_ln, q=q, k=k, v=v, scores=scores, attn=attn, z=z, concat=concat,
                attn_out=attn_out, resid_mid=resid_mid, ln2=m_ln, hidden=hidden, mlp_out=mlp_out, resid_post=resid_post,
            ))
        x = resid_post
    final = layer_norm(x, p["lnf_g"], p["lnf_b"])
    logits = final @ p["W_U"] + p["b_U"]
    if record is not None:
        record.update(final=final, logits=logits)
    return logits


def param_count(p: Params) -> int:
    return sum(t.numel() for t in p.values())


# ── training (once, lazily) ──────────────────────────────────────────────────

@dataclass
class TrainedTransformer:
    cfg: TConfig
    params: Params
    vocab: Vocab
    corpus: List[str]
    initial_loss: float
    final_loss: float
    steps: int
    seconds: float
    seed: int


def _batches(vocab: Vocab, corpus: Sequence[str], cfg: TConfig, rng: random.Random, batch: int) -> torch.Tensor:
    stream = [BOS]
    for line in corpus:
        stream += [vocab.stoi[w] for w in line.split()]
    n = cfg.max_len + 1
    starts = [rng.randrange(0, len(stream) - n) for _ in range(batch)]
    return torch.tensor([stream[s:s + n] for s in starts])


def train_transformer(seed: int = 0, steps: int = 500, batch: int = 48, lr: float = 3e-3) -> TrainedTransformer:
    t0 = time.time()
    corpus = build_corpus(seed)
    vocab = Vocab(corpus)
    cfg = TConfig(vocab_size=len(vocab))
    params = init_params(cfg, seed)
    for t in params.values():
        t.requires_grad_(True)
    opt = torch.optim.Adam(params.values(), lr=lr)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    first = last = 0.0
    for step in range(steps):
        xy = _batches(vocab, corpus, cfg, rng, batch)
        logits = forward(cfg, params, xy[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), xy[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss.detach())
        last = float(loss.detach())
    frozen = {k: v.detach().clone() for k, v in params.items()}
    return TrainedTransformer(cfg, frozen, vocab, corpus, first, last, steps, time.time() - t0, seed)


_model: Optional[TrainedTransformer] = None
_lock = threading.Lock()


def get_model() -> TrainedTransformer:
    global _model
    with _lock:
        if _model is None:
            _model = train_transformer()
        return _model


# ── wire contract ────────────────────────────────────────────────────────────

class TransformerInfo(BaseModel):
    description: str
    d_model: int
    n_heads: int
    n_layers: int
    d_head: int
    d_mlp: int
    max_len: int
    vocab: List[str]
    param_count: int
    corpus_size: int
    corpus_examples: List[str]
    initial_loss: float
    final_loss: float
    train_steps: int
    train_seconds: float
    seed: int
    examples: List[str]


class TraceRequest(BaseModel):
    text: str = Field(..., max_length=400)
    ablate_heads: List[Tuple[int, int]] = Field(default_factory=list, description="(layer, head) pairs whose output is zeroed")
    top_k: int = 10


class TokenOut(BaseModel):
    id: int
    text: str
    unknown: bool


class HeadTrace(BaseModel):
    head: int
    q: List[List[float]]  # [n, d_head]
    k: List[List[float]]
    v: List[List[float]]
    scores: List[List[float]]  # QK^T / sqrt(d_head), before the mask
    attn: List[List[float]]  # softmax over keys (rows = queries / destinations)
    out: List[List[float]]  # attn @ V, [n, d_head]
    ablated: bool


class LayerTrace(BaseModel):
    layer: int
    resid_pre: List[List[float]]
    ln1: List[List[float]]
    heads: List[HeadTrace]
    attn_out: List[List[float]]
    resid_mid: List[List[float]]
    ln2: List[List[float]]
    mlp_hidden_norm: List[float]
    mlp_active_fraction: List[float]
    mlp_out: List[List[float]]
    resid_post: List[List[float]]
    norms: Dict[str, List[float]]  # per position L2 norms of the residual pieces


class Candidate(BaseModel):
    token: str
    prob: float
    logit: float


class TransformerTrace(BaseModel):
    tokens: List[TokenOut]
    truncated: bool
    scale: float  # 1/sqrt(d_head)
    tok_emb: List[List[float]]
    pos_emb: List[List[float]]
    embed: List[List[float]]
    layers: List[LayerTrace]
    final_ln: List[List[float]]
    logits_last: List[float]
    next_token: List[Candidate]  # top-k for the last position
    next_token_baseline: Optional[List[Candidate]] = None  # without head ablation, when ablation is active
    per_position_top: List[Candidate]  # argmax prediction at every position
    ablate_heads: List[Tuple[int, int]]


def _m(t: torch.Tensor) -> List[List[float]]:
    return [[float(v) for v in row] for row in t.detach().tolist()]


def _top(logits: torch.Tensor, vocab: Vocab, k: int) -> List[Candidate]:
    probs = torch.softmax(logits, -1)
    vals, idx = probs.topk(min(k, len(vocab)))
    return [Candidate(token=vocab.itos[i], prob=float(p), logit=float(logits[i])) for p, i in zip(vals.tolist(), idx.tolist())]


def trace(model: TrainedTransformer, req: TraceRequest) -> TransformerTrace:
    cfg, vocab = model.cfg, model.vocab
    ids, words, unknown = model.vocab.encode(req.text)
    if not ids:
        raise ValueError("enter at least one word")
    for lyr, hd in req.ablate_heads:
        if not (0 <= lyr < cfg.n_layers and 0 <= hd < cfg.n_heads):
            raise ValueError(f"no head ({lyr}, {hd})")
    ids = [BOS] + ids
    words = ["<bos>"] + words
    unknown = [False] + unknown
    truncated = len(ids) > cfg.max_len
    if truncated:
        ids, words, unknown = ids[-cfg.max_len:], words[-cfg.max_len:], unknown[-cfg.max_len:]
    x = torch.tensor([ids])
    rec: dict = {}
    with torch.no_grad():
        logits = forward(cfg, model.params, x, req.ablate_heads, rec)
        baseline = forward(cfg, model.params, x)[0, -1] if req.ablate_heads else None
    layers = []
    for li, L in enumerate(rec["layers"]):
        heads = [HeadTrace(
            head=h, q=_m(L["q"][0, h]), k=_m(L["k"][0, h]), v=_m(L["v"][0, h]),
            scores=_m(L["scores"][0, h]), attn=_m(L["attn"][0, h]), out=_m(L["z"][0, h]),
            ablated=(li, h) in {tuple(a) for a in req.ablate_heads},
        ) for h in range(cfg.n_heads)]
        hidden = L["hidden"][0]
        layers.append(LayerTrace(
            layer=li, resid_pre=_m(L["resid_pre"][0]), ln1=_m(L["ln1"][0]), heads=heads,
            attn_out=_m(L["attn_out"][0]), resid_mid=_m(L["resid_mid"][0]), ln2=_m(L["ln2"][0]),
            mlp_hidden_norm=[float(v) for v in hidden.norm(dim=-1)],
            mlp_active_fraction=[float(v) for v in (hidden > 0).float().mean(-1)],
            mlp_out=_m(L["mlp_out"][0]), resid_post=_m(L["resid_post"][0]),
            norms={k: [float(v) for v in L[k][0].norm(dim=-1)] for k in ("resid_pre", "attn_out", "mlp_out", "resid_post")},
        ))
    per_pos = [_top(logits[0, i], vocab, 1)[0] for i in range(len(ids))]
    return TransformerTrace(
        tokens=[TokenOut(id=i, text=w, unknown=u) for i, w, u in zip(ids, words, unknown)],
        truncated=truncated, scale=1 / math.sqrt(cfg.d_head),
        tok_emb=_m(rec["tok"][0]), pos_emb=_m(rec["pos"][0]), embed=_m(rec["embed"][0]),
        layers=layers, final_ln=_m(rec["final"][0]), logits_last=[float(v) for v in logits[0, -1]],
        next_token=_top(logits[0, -1], vocab, req.top_k),
        next_token_baseline=_top(baseline, vocab, req.top_k) if baseline is not None else None,
        per_position_top=per_pos, ablate_heads=list(req.ablate_heads),
    )


def info(model: TrainedTransformer) -> TransformerInfo:
    cfg = model.cfg
    return TransformerInfo(
        description=("A tiny decoder-only Transformer trained locally on template sentences. "
                     "It is not GPT, ChatGPT, Claude or any production model."),
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers, d_head=cfg.d_head, d_mlp=cfg.d_mlp,
        max_len=cfg.max_len, vocab=model.vocab.itos, param_count=param_count(model.params),
        corpus_size=len(model.corpus), corpus_examples=model.corpus[:6],
        initial_loss=model.initial_loss, final_loss=model.final_loss,
        train_steps=model.steps, train_seconds=model.seconds, seed=model.seed,
        examples=["the cat sat on", "the dog ran to the", "a bird jumped over the tree . then it", "my friend looked at the"],
    )
