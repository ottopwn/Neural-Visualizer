"""The Transformer Lab must report exactly what its forward pass computes."""
import math

import pytest
import torch
import torch.nn.functional as F

from forge import transformer as T


@pytest.fixture(scope="module")
def model():
    return T.train_transformer(steps=80)  # short run: enough to test the mechanics


def test_training_reduces_loss_and_is_deterministic(model):
    assert model.final_loss < model.initial_loss
    again = T.train_transformer(steps=80)
    assert again.final_loss == pytest.approx(model.final_loss, abs=1e-5)
    # Same seed, same model (up to float rounding of multi-threaded CPU kernels).
    assert all(torch.allclose(model.params[k], again.params[k], atol=1e-5) for k in model.params)


def test_tokenizer_marks_unknown_words(model):
    ids, words, unknown = model.vocab.encode("The cat zebra sat.")
    assert words == ["the", "cat", "zebra", "sat", "."]
    assert unknown == [False, False, True, False, False]
    assert ids[2] == T.UNK


def independent_forward(cfg, p, ids, ablate=()):
    """Plain re-implementation with torch.nn.functional primitives only."""
    n = len(ids)
    x = p["tok_emb"][ids] + p["pos_emb"][:n]
    out = {"layers": []}
    for layer in range(cfg.n_layers):
        pre = f"l{layer}."
        h = F.layer_norm(x, (cfg.d_model,), p[pre + "ln1_g"], p[pre + "ln1_b"])
        heads = []
        for hd in range(cfg.n_heads):
            sl = slice(hd * cfg.d_head, (hd + 1) * cfg.d_head)
            q, k, v = h @ p[pre + "W_Q"][:, sl], h @ p[pre + "W_K"][:, sl], h @ p[pre + "W_V"][:, sl]
            s = (q @ k.T) / math.sqrt(cfg.d_head)
            a = torch.softmax(s + torch.triu(torch.full((n, n), float("-inf")), 1), -1)
            z = a @ v * (0.0 if (layer, hd) in ablate else 1.0)
            heads.append((q, k, v, s, a, z))
        x = x + torch.cat([hz[5] for hz in heads], -1) @ p[pre + "W_O"] + p[pre + "b_O"]
        m = F.layer_norm(x, (cfg.d_model,), p[pre + "ln2_g"], p[pre + "ln2_b"])
        x = x + F.gelu(m @ p[pre + "W_1"] + p[pre + "b_1"]) @ p[pre + "W_2"] + p[pre + "b_2"]
        out["layers"].append(heads)
    logits = F.layer_norm(x, (cfg.d_model,), p["lnf_g"], p["lnf_b"]) @ p["W_U"] + p["b_U"]
    out["logits"] = logits
    return out


def test_trace_matches_independent_forward(model):
    tr = T.trace(model, T.TraceRequest(text="the cat sat on the"))
    ids = [t.id for t in tr.tokens]
    assert tr.tokens[0].text == "<bos>" and [t.text for t in tr.tokens[1:]] == ["the", "cat", "sat", "on", "the"]
    ref = independent_forward(model.cfg, model.params, torch.tensor(ids))
    n = len(ids)
    for li, L in enumerate(tr.layers):
        for hd, H in enumerate(L.heads):
            q, k, v, s, a, z = ref["layers"][li][hd]
            assert torch.allclose(torch.tensor(H.q), q, atol=1e-5)
            assert torch.allclose(torch.tensor(H.scores), s, atol=1e-5)
            assert torch.allclose(torch.tensor(H.attn), a, atol=1e-6)
            assert torch.allclose(torch.tensor(H.out), z, atol=1e-5)
            A = torch.tensor(H.attn)
            assert torch.allclose(A.sum(-1), torch.ones(n), atol=1e-6)  # softmax rows
            assert torch.all(A[torch.triu(torch.ones(n, n, dtype=torch.bool), 1)] == 0)  # causal mask
            # scores really are Q K^T * scale
            assert torch.allclose(torch.tensor(H.q) @ torch.tensor(H.k).T * tr.scale, torch.tensor(H.scores), atol=1e-5)
    assert torch.allclose(torch.tensor(tr.logits_last), ref["logits"][-1], atol=1e-4)
    probs = torch.softmax(ref["logits"][-1], -1)
    assert tr.next_token[0].prob == pytest.approx(float(probs.max()), abs=1e-6)
    assert sum(c.prob for c in tr.next_token) <= 1 + 1e-6


def test_residual_stream_identities(model):
    tr = T.trace(model, T.TraceRequest(text="a fox walked to the park ."))
    emb = torch.tensor(tr.tok_emb) + torch.tensor(tr.pos_emb)
    assert torch.allclose(emb, torch.tensor(tr.embed), atol=1e-6)
    prev = torch.tensor(tr.embed)
    for L in tr.layers:
        assert torch.allclose(torch.tensor(L.resid_pre), prev, atol=1e-6)
        assert torch.allclose(torch.tensor(L.resid_mid), prev + torch.tensor(L.attn_out), atol=1e-5)
        assert torch.allclose(torch.tensor(L.resid_post), torch.tensor(L.resid_mid) + torch.tensor(L.mlp_out), atol=1e-5)
        prev = torch.tensor(L.resid_post)


def test_head_ablation_is_a_real_what_if(model):
    text = "the girl ran to the house . then it"
    base = T.trace(model, T.TraceRequest(text=text))
    abl = T.trace(model, T.TraceRequest(text=text, ablate_heads=[(0, 1)]))
    assert abl.layers[0].heads[1].ablated and all(v == 0 for row in abl.layers[0].heads[1].out for v in row)
    assert abl.next_token_baseline is not None
    assert [c.token for c in abl.next_token_baseline] == [c.token for c in base.next_token]
    ref = independent_forward(model.cfg, model.params, torch.tensor([t.id for t in abl.tokens]), ablate={(0, 1)})
    assert torch.allclose(torch.tensor(abl.logits_last), ref["logits"][-1], atol=1e-4)
    with pytest.raises(ValueError):
        T.trace(model, T.TraceRequest(text=text, ablate_heads=[(5, 0)]))
    with pytest.raises(ValueError):
        T.trace(model, T.TraceRequest(text="   "))


def test_long_input_is_truncated_to_the_context(model):
    tr = T.trace(model, T.TraceRequest(text=" ".join(["the cat sat on the mat ."] * 6)))
    assert tr.truncated and len(tr.tokens) == model.cfg.max_len
