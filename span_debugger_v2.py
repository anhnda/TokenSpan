"""
span_debugger.py — logit-lens CLI tool.

Flow
----
1. Read a sentence (CLI arg, --stdin, or interactive).
2. Greedy-decode an answer from an Instruct model.
3. Concat [prompt + answer], ONE forward pass with output_hidden_states.
4. For each answer token y_t, project the hidden state at its prediction
   slot through the model's final-norm + lm_head (the "logit lens") at
   several uniformly-sampled layers, and report p(y_t) at each.

Output columns
--------------
    idx | token | p(last) | p@layer | p@layer | ...   (final -> lower)

- col 1: token index in the answer
- col 2: the token (decoded)
- col 3: p of the LAST (final) layer  == the model's real probability
- col 4..: p of the SAME token read off lower layers, high -> low

Layers sampled uniformly in linear space across [0 .. n_layers], count set
by --logit-len (default 5). The final layer is always col 3.

The point: a token whose probability is already high several layers early is
"decided" deep in the stack; a run of such tokens is a committed span.
Tokens that only resolve in the top layers light up late.
"""

import argparse
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


# ---------------------------------------------------------------------------
# Model / prompt helpers
# ---------------------------------------------------------------------------

def get_stop_token_ids(tokenizer):
    stop = set()
    if tokenizer.eos_token_id is not None:
        stop.add(tokenizer.eos_token_id)
    for s in ("<|eot_id|>", "<|end_of_text|>"):
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop.add(tid)
    return stop


def build_prompt_ids(tokenizer, sentence):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": sentence}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    return ids["input_ids"][0].to(DEVICE)


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt_ids, stop_ids, max_new_tokens):
    cur = prompt_ids.clone()
    out = []
    for _ in range(max_new_tokens):
        logits = model(cur.unsqueeze(0)).logits[0, -1]
        nxt = int(logits.argmax().item())
        if nxt in stop_ids:
            break
        out.append(nxt)
        cur = torch.cat([cur, torch.tensor([nxt], device=DEVICE)])
    return torch.tensor(out, dtype=torch.long, device=DEVICE)


def get_final_norm_and_head(model):
    """
    Locate the final RMSNorm/LayerNorm and the lm_head so the logit lens is
    applied consistently with how the model produces its real logits.
    Works for Llama/Mistral/Gemma-style `model.model.norm` + `model.lm_head`.
    """
    inner = getattr(model, "model", model)
    final_norm = getattr(inner, "norm", None)
    if final_norm is None:
        final_norm = getattr(inner, "final_layernorm", None)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        lm_head = getattr(model, "embed_out", None)
    if final_norm is None or lm_head is None:
        raise RuntimeError("Could not locate final norm / lm_head for logit lens.")
    return final_norm, lm_head


def pick_layers(n_hidden_states, k, only_last=False):
    """
    hidden_states has length n_layers+1 (index 0 = embeddings, last = final).

    only_last=False (default): sample k indices uniformly in linear space
        across [0, n_layers], always including the final layer.
    only_last=True: take the top k layers counting back from the final one
        (n_layers, n_layers-1, ...), since the relevant signal usually lives
        in the last layers.

    Returns indices sorted HIGH -> LOW so the final layer prints first.
    """
    last = n_hidden_states - 1
    if k <= 1:
        return [last]
    if only_last:
        lo = max(0, last - (k - 1))
        return list(range(last, lo - 1, -1))   # last, last-1, ...
    raw = torch.linspace(0, last, steps=k).round().long().tolist()
    uniq = sorted(set(raw))
    if last not in uniq:
        uniq.append(last)
        uniq = sorted(set(uniq))
    return sorted(uniq, reverse=True)   # high layer first


@torch.no_grad()
def logit_lens_probs(model, prompt_ids, answer_ids, layer_indices,
                     final_norm, lm_head):
    """
    One forward pass with hidden states. For each answer token y_t and each
    selected layer L: p_L(y_t) = softmax(lm_head(norm(h_L[slot])))[y_t].

    Returns:
        probs:    Tensor [T, k]        p of the TARGET token per layer
        top_p:    Tensor [T, k]        p of the argmax token per layer
        top_id:   LongTensor [T, k]    id of the argmax token per layer
    (cols ordered as layer_indices)
    """
    full = torch.cat([prompt_ids, answer_ids]).unsqueeze(0)
    out = model(full, output_hidden_states=True)
    hs = out.hidden_states                       # tuple, len = n_layers+1
    L = prompt_ids.numel()
    T = answer_ids.numel()
    pos = torch.arange(L - 1, L - 1 + T, device=DEVICE)   # prediction slots

    p_cols, top_p_cols, top_id_cols = [], [], []
    for li in layer_indices:
        h = hs[li][0, pos]                       # [T, d]
        logits = lm_head(final_norm(h))          # [T, V]
        lp = F.log_softmax(logits, dim=-1)
        probs_all = lp.exp()
        p = probs_all.gather(1, answer_ids.unsqueeze(1)).squeeze(1)  # [T]
        top_p, top_id = probs_all.max(dim=-1)                        # [T], [T]
        p_cols.append(p)
        top_p_cols.append(top_p)
        top_id_cols.append(top_id)
    return (torch.stack(p_cols, dim=1),
            torch.stack(top_p_cols, dim=1),
            torch.stack(top_id_cols, dim=1))


def gradxembed_sensitivity(model, prompt_ids, answer_ids, layer_indices,
                           final_norm, lm_head, use_logit=False):
    """
    grad x embedding sensitivity of each input-token hidden state to the
    output. One forward pass WITH grad and output_hidden_states.

    Scalar S = sum over answer tokens y_t of:
        use_logit=False -> p(y_t)            (sum of output probabilities)
        use_logit=True  -> logit(y_t)        (sum of output logits; no softmax,
                                              avoids saturation)
    where p/logit at slot t are read off the FINAL layer via the logit lens
    (same head/norm as the model's real output).

    For each selected layer L we then form  (dS/dh_L) . h_L  summed over the
    hidden dim, giving one sensitivity scalar per sequence position.

    Returns:
        sens:  Tensor [N, k]   grad.embedding per position (rows) per layer
                               (cols ordered as layer_indices), N = L+T
    """
    full = torch.cat([prompt_ids, answer_ids]).unsqueeze(0)
    out = model(full, output_hidden_states=True)
    hs = out.hidden_states                       # tuple, len = n_layers+1
    L = prompt_ids.numel()
    T = answer_ids.numel()
    pos = torch.arange(L - 1, L - 1 + T, device=DEVICE)   # prediction slots

    # scalar from the FINAL layer at the prediction slots
    h_final = hs[-1][0, pos]                      # [T, d]
    logits = lm_head(final_norm(h_final))         # [T, V]
    tgt = logits.gather(1, answer_ids.unsqueeze(1)).squeeze(1)   # [T]
    if use_logit:
        scalar = tgt.sum()
    else:
        scalar = F.softmax(logits, dim=-1).gather(
            1, answer_ids.unsqueeze(1)).squeeze(1).sum()

    sel = [hs[li] for li in layer_indices]        # each [1, N, d], requires grad
    grads = torch.autograd.grad(scalar, sel, retain_graph=False)
    cols = []
    for g, h in zip(grads, sel):
        cols.append((g[0] * h[0]).sum(dim=-1))    # [N]  grad . embedding
    return torch.stack(cols, dim=1)               # [N, k]


# ---------------------------------------------------------------------------
# Integrated Gradients from a hidden layer to the TRUE output logit.
#
# Unlike the logit lens, this never re-applies lm_head to an intermediate
# layer. It replaces h_L by a path  alpha * h_L  (alpha: 0 -> 1) and forwards
# through the REAL downstream decoder layers + final norm + head, integrating
# the gradient of the target logit w.r.t. h_L along the path. By the IG
# completeness axiom the per-position attributions sum to
#     logit_target(h_L) - logit_target(baseline=0)
# which is reported as a residual check (no proxy, an actual guarantee).
# ---------------------------------------------------------------------------

def _get_decoder_layers(model):
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError("Could not locate decoder layers for IG.")
    return inner, layers


def _run_upper_stack(model, inner, layers, h_start, start_layer,
                     position_ids, causal_mask, position_embeddings):
    """
    Forward hidden state h_start (output of decoder block index start_layer-1,
    i.e. hidden_states[start_layer]) through decoder layers start_layer..end.

    hidden_states index convention: hs[k] is the input to decoder layer k
    (hs[0] = embeddings, hs[n_layers] = final pre-norm output). So to continue
    from hs[L] we run decoder layers L, L+1, ..., n_layers-1.
    """
    h = h_start
    for k in range(start_layer, len(layers)):
        layer = layers[k]
        kwargs = {}
        # newer HF: position_embeddings precomputed (cos,sin); older: not.
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings
        try:
            out = layer(
                h,
                attention_mask=causal_mask,
                position_ids=position_ids,
                **kwargs,
            )
        except TypeError:
            out = layer(h, attention_mask=causal_mask, position_ids=position_ids)
        h = out[0] if isinstance(out, tuple) else out
    return h


def _baseline_hidden_states(model, tokenizer, prompt_ids, answer_ids,
                            kind, full_hs):
    """
    Build per-layer baseline hidden states aligned to the full [prompt+answer]
    sequence, for IG. Returns a tuple like hidden_states (len n_layers+1),
    each [1, N, d], OR a string error message if the baseline is unavailable.

    kind:
        zero   -> all zeros (handled by caller; returns None here)
        mean   -> per-layer mean over positions, broadcast to all positions
        corrupt-> hidden states of a content-shuffled prompt (causal-tracing
                  style contrast); answer region kept as-is in token space
        pad    -> hidden states of an all-pad-token sequence (same length)
        mask   -> hidden states of an all-mask-token sequence (if tokenizer
                  exposes a mask token; else error string)
    """
    if kind == "zero":
        return None
    if kind == "mean":
        # per-layer centroid over positions, broadcast back
        return tuple(h.mean(dim=1, keepdim=True).expand_as(h)
                     for h in full_hs)

    full = torch.cat([prompt_ids, answer_ids]).unsqueeze(0)
    N = full.shape[1]

    if kind == "corrupt":
        # shuffle non-special prompt tokens; keep specials and answer in place
        specials = set(tokenizer.all_special_ids or [])
        ids = full.clone()
        L = prompt_ids.numel()
        movable = [i for i in range(L)
                   if int(ids[0, i].item()) not in specials]
        if len(movable) > 1:
            perm = movable[:]
            g = torch.Generator(device="cpu").manual_seed(0)
            idx = torch.randperm(len(perm), generator=g).tolist()
            shuffled = [perm[i] for i in idx]
            vals = ids[0, perm].clone()
            ids[0, shuffled] = vals
        base_ids = ids

    elif kind == "pad":
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            return "pad baseline unavailable: tokenizer has no pad/eos token"
        base_ids = torch.full_like(full, int(pad_id))

    elif kind == "mask":
        mask_id = getattr(tokenizer, "mask_token_id", None)
        if mask_id is None:
            return ("mask baseline unavailable: tokenizer has no mask token "
                    "(decoder-only models like Llama/Mistral lack one)")
        base_ids = torch.full_like(full, int(mask_id))

    else:
        return f"unknown baseline kind: {kind}"

    with torch.no_grad():
        out = model(base_ids, output_hidden_states=True)
    return out.hidden_states


def ig_from_layer(model, tokenizer, prompt_ids, answer_ids, layer_indices,
                  final_norm, lm_head, steps=16, baseline_kind="zero"):
    """
    IG of the TRUE target logit w.r.t. each selected layer's hidden state.

    baseline_kind: zero | mean | corrupt | pad | mask  (see _baseline_hidden_states)

    Returns:
        attr:      Tensor [N, k]   IG attribution per position per layer
        residual:  Tensor [k]      completeness residual per layer:
                                    sum_pos attr - (logit(h_L) - logit(baseline)),
                                    should be ~0 (Riemann error only).
        note:      str | None      message if a requested baseline was
                                    unavailable and zero was used instead.
    """
    inner, layers = _get_decoder_layers(model)
    full = torch.cat([prompt_ids, answer_ids]).unsqueeze(0)
    L = prompt_ids.numel()
    T = answer_ids.numel()
    pos = torch.arange(L - 1, L - 1 + T, device=DEVICE)
    seqlen = full.shape[1]

    # Capture the model's own positional / mask machinery via one clean pass.
    with torch.no_grad():
        base = model(full, output_hidden_states=True)
    hs = base.hidden_states

    # Build baseline hidden states (per layer) once.
    note = None
    base_hs = _baseline_hidden_states(
        model, tokenizer, prompt_ids, answer_ids, baseline_kind, hs)
    if isinstance(base_hs, str):
        note = base_hs + "  -> falling back to zero baseline"
        base_hs = None  # zero

    position_ids = torch.arange(seqlen, device=DEVICE).unsqueeze(0)
    # rotary embeddings, if the model precomputes them (newer HF)
    position_embeddings = None
    rotary = getattr(inner, "rotary_emb", None)
    if rotary is not None:
        try:
            emb0 = inner.embed_tokens(full)
            position_embeddings = rotary(emb0, position_ids)
        except Exception:
            position_embeddings = None
    # causal mask: rely on layer's internal handling by passing None when the
    # model builds it itself; otherwise an additive triangular mask.
    causal_mask = None

    attr_cols, resid = [], []
    for li in layer_indices:
        h_L = hs[li].detach()                     # [1, N, d]  fixed endpoint
        if base_hs is None:
            baseline = torch.zeros_like(h_L)
        else:
            baseline = base_hs[li].detach()
        delta = h_L - baseline

        grad_acc = torch.zeros_like(h_L)
        # endpoint logits for the completeness check
        def target_logit(h_in):
            h_top = _run_upper_stack(
                model, inner, layers, h_in, li,
                position_ids, causal_mask, position_embeddings)
            lg = lm_head(final_norm(h_top[0, pos]))            # [T, V]
            return lg.gather(1, answer_ids.unsqueeze(1)).squeeze(1).sum()

        for s in range(steps):
            alpha = (s + 0.5) / steps             # midpoint rule
            h_in = (baseline + alpha * delta).clone().requires_grad_(True)
            S = target_logit(h_in)
            g, = torch.autograd.grad(S, h_in)
            grad_acc += g
        avg_grad = grad_acc / steps
        ig = (avg_grad * delta)[0]                # [N, d]
        attr = ig.sum(dim=-1)                     # [N]

        with torch.no_grad():
            f_hi = target_logit(h_L)
            f_lo = target_logit(baseline)
        residual = attr.sum() - (f_hi - f_lo)
        attr_cols.append(attr)
        resid.append(residual)

    return torch.stack(attr_cols, dim=1), torch.stack(resid), note


def stability_panel(model, prompt_ids, answer_ids, layer_indices,
                    final_norm, lm_head, target_idx=0):
    """
    Proxy-free stability / responsiveness panel. No lm_head on inner layers.

    For ONE target answer token y_{target_idx}, read at its prediction slot,
    with scalar = its TRUE logit (single logit, not a sum):

      drift[i, L]   = 1 - cos(h_L[i], h_final[i])
                      intrinsic representational convergence of position i:
                      small  -> representation already settled by layer L
                      large  -> still being rewritten in upper layers

      relsens[i, L] = ||d logit_target / d h_L[i]|| * ||h_L[i]|| / |logit_target|
                      dimensionless output responsiveness:
                      fractional change in the target logit per fractional
                      change in this token's layer-L representation.
                      small  -> output insensitive here (stable / irrelevant)
                      large  -> output still swings with this token (unstable /
                                load-bearing)

    Returns:
        drift:   Tensor [N, k]
        relsens: Tensor [N, k]
        tgt_tok: int   the answer token id that was attributed
    """
    full = torch.cat([prompt_ids, answer_ids]).unsqueeze(0)
    L = prompt_ids.numel()
    T = answer_ids.numel()
    target_idx = max(0, min(target_idx, T - 1))
    slot = L - 1 + target_idx                      # prediction slot of target
    tgt_id = int(answer_ids[target_idx].item())

    out = model(full, output_hidden_states=True)
    hs = out.hidden_states                         # len n_layers+1, each [1,N,d]
    h_final = hs[-1][0]                            # [N, d]

    # scalar = the single TRUE target logit
    logit_vec = lm_head(final_norm(hs[-1][0, slot:slot + 1]))[0]   # [V]
    target_logit = logit_vec[tgt_id]
    denom = target_logit.detach().abs().clamp_min(1e-6)

    sel = [hs[li] for li in layer_indices]
    grads = torch.autograd.grad(target_logit, sel, retain_graph=False)

    drift_cols, rel_cols = [], []
    cos = torch.nn.functional.cosine_similarity
    for g, h, li in zip(grads, sel, layer_indices):
        hL = h[0]                                  # [N, d]
        # drift: 1 - cos(h_L, h_final), per position
        d = 1.0 - cos(hL, h_final, dim=-1)         # [N]
        # relsens: ||g|| * ||h|| / |logit|, per position
        gnorm = g[0].norm(dim=-1)                  # [N]
        hnorm = hL.norm(dim=-1)                     # [N]
        rs = gnorm * hnorm / denom                  # [N]
        drift_cols.append(d.detach())
        rel_cols.append(rs.detach())
    return (torch.stack(drift_cols, dim=1),
            torch.stack(rel_cols, dim=1),
            tgt_id)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def decode_tok(tokenizer, tid):
    return tokenizer.decode([tid])


def report(tokenizer, prompt_ids, answer_ids, probs, top_p, top_id,
           sens, layer_indices, n_layers, use_logit):
    """
    One table per answer token, with prob (logit lens) and grad.embedding
    sensitivity side by side for each selected layer.

    For answer token i: prob is read at its prediction slot; the grad.embedding
    sensitivity is taken at sequence position L+i (the answer token's own slot).
    """
    target_ids = answer_ids.tolist()
    toks = [decode_tok(tokenizer, t) for t in target_ids]
    P = probs.tolist()
    TP = top_p.tolist()
    TID = top_id.tolist()
    L = prompt_ids.numel()
    S = sens.tolist()                            # [N, k], N = L+T

    # interleaved column labels: p(L..) then g.e@L.. for each layer
    p_w, g_w = 30, 13
    hdr = f"{'idx':>3}  {'token':<16}"
    for j, li in enumerate(layer_indices):
        plab = f"p(L{li})" if j == 0 else f"p@L{li}"
        hdr += f"{plab:>{p_w}}{('g.e@L'+str(li)):>{g_w}}"
    print()
    print(hdr)
    print("-" * len(hdr))
    for i, tok in enumerate(toks):
        row = f"{i:>3}  {repr(tok):<16}"
        for j in range(len(layer_indices)):
            cell = f"{P[i][j]:.3f}"
            if TID[i][j] != target_ids[i]:
                win_tok = decode_tok(tokenizer, TID[i][j])
                cell += f" [{TP[i][j]:.3f}:{win_tok!r}]"
            row += f"{cell:>{p_w}}"
            row += f"{S[L + i][j]:>{g_w}.4f}"     # sensitivity at answer slot L+i
        print(row)
    print()
    scalar_kind = "sum logit(target)" if use_logit else "sum p(target)"
    print(f"Layers sampled (of {n_layers}): "
          + ", ".join(f"L{li}" for li in layer_indices)
          + f"   [hidden_states idx; 0=embeddings, {n_layers}=final]")
    print("p(...) = p(target) via logit lens; if layer top-1 != target, "
          "[p:'tok'] of argmax shown.")
    print(f"g.e@L  = grad x embedding sensitivity at the answer slot "
          f"[scalar = {scalar_kind}].")
    print(f"Answer: {tokenizer.decode(answer_ids)!r}")
    print()


def report_sensitivity(tokenizer, prompt_ids, answer_ids, sens,
                       layer_indices, n_layers, use_logit):
    """
    Print grad.embedding per sequence position (input tokens + answer tokens).
    """
    seq_ids = torch.cat([prompt_ids, answer_ids]).tolist()
    L = prompt_ids.numel()
    S = sens.tolist()

    layer_labels = [f"g.e@L{li}" for li in layer_indices]
    col_w = 14
    hdr = f"{'idx':>3}  {'pos':<5}{'token':<16}" + "".join(
        f"{lab:>{col_w}}" for lab in layer_labels)
    scalar_kind = "sum logit(target)" if use_logit else "sum p(target)"
    print(f"grad x embedding sensitivity  [scalar = {scalar_kind}]")
    print(hdr)
    print("-" * len(hdr))
    for i, tid in enumerate(seq_ids):
        tag = "ans" if i >= L else "in"
        tok = decode_tok(tokenizer, tid)
        row = f"{i:>3}  {tag:<5}{repr(tok):<16}"
        for j in range(len(layer_indices)):
            row += f"{S[i][j]:>{col_w}.4f}"
        print(row)
    print()


def report_ig(tokenizer, prompt_ids, answer_ids, attr, residual,
              layer_indices, n_layers, steps, baseline_kind="zero"):
    """
    IG-from-layer attribution per sequence position, with the completeness
    residual per layer printed in the footer.
    """
    seq_ids = torch.cat([prompt_ids, answer_ids]).tolist()
    L = prompt_ids.numel()
    A = attr.tolist()
    R = residual.tolist()

    layer_labels = [f"IG@L{li}" for li in layer_indices]
    col_w = 14
    hdr = f"{'idx':>3}  {'pos':<5}{'token':<16}" + "".join(
        f"{lab:>{col_w}}" for lab in layer_labels)
    print(f"Integrated Gradients to TRUE logit  "
          f"[steps={steps}, baseline={baseline_kind}, scalar=sum logit(target)]")
    print(hdr)
    print("-" * len(hdr))
    for i, tid in enumerate(seq_ids):
        tag = "ans" if i >= L else "in"
        tok = decode_tok(tokenizer, tid)
        row = f"{i:>3}  {tag:<5}{repr(tok):<16}"
        for j in range(len(layer_indices)):
            row += f"{A[i][j]:>{col_w}.4f}"
        print(row)
    print()
    print("completeness residual (should be ~0): "
          + ", ".join(f"L{li}={R[j]:+.4f}"
                      for j, li in enumerate(layer_indices)))
    print()


def report_stability(tokenizer, prompt_ids, answer_ids, drift, relsens,
                     layer_indices, n_layers, tgt_id, target_idx):
    """
    Proxy-free stability panel: drift (1-cos to final) and relsens
    (dimensionless output responsiveness) side by side per layer.
    """
    seq_ids = torch.cat([prompt_ids, answer_ids]).tolist()
    L = prompt_ids.numel()
    D = drift.tolist()
    RS = relsens.tolist()

    d_w, r_w = 11, 12
    hdr = f"{'idx':>3}  {'pos':<5}{'token':<16}"
    for li in layer_indices:
        hdr += f"{('drift@L'+str(li)):>{d_w}}{('rsens@L'+str(li)):>{r_w}}"
    tgt_tok = decode_tok(tokenizer, tgt_id)
    print(f"stability panel  [target = answer tok #{target_idx} "
          f"{tgt_tok!r}; drift=1-cos(h_L,h_final); "
          f"rsens=||g||*||h||/|logit|, proxy-free]")
    print(hdr)
    print("-" * len(hdr))
    for i, tid in enumerate(seq_ids):
        tag = "ans" if i >= L else "in"
        tok = decode_tok(tokenizer, tid)
        row = f"{i:>3}  {tag:<5}{repr(tok):<16}"
        for j in range(len(layer_indices)):
            row += f"{D[i][j]:>{d_w}.4f}{RS[i][j]:>{r_w}.4f}"
        print(row)
    print()
    print("read: low drift + low rsens = settled & irrelevant; "
          "low drift + high rsens = decided early but load-bearing; "
          "high drift + high rsens = still actively shaping the output.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_sentence(args):
    if args.sentence:
        return " ".join(args.sentence)
    if args.stdin:
        return sys.stdin.read().strip()
    try:
        return input("sentence> ").strip()
    except EOFError:
        return ""


def main():
    ap = argparse.ArgumentParser(description="Logit-lens per-token span debugger.")
    ap.add_argument("sentence", nargs="*", help="input sentence")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--logit-len", type=int, default=5,
                    help="number of layers to sample (uniform in linear space)")
    ap.add_argument("--only_last", action="store_true",
                    help="pick the top --logit-len layers from the last "
                         "backward (final, final-1, ...) instead of uniform")
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("--logit", action="store_true",
                    help="grad.embedding scalar uses sum of output LOGITS "
                         "instead of sum of probs (avoids softmax saturation)")
    ap.add_argument("--ig", action="store_true",
                    help="also compute Integrated Gradients from each selected "
                         "layer to the TRUE output logit (no borrowed-head "
                         "proxy; reports completeness residual)")
    ap.add_argument("--ig-steps", type=int, default=16,
                    help="number of IG integration steps (midpoint rule)")
    ap.add_argument("--ig-baseline", default="zero",
                    choices=["zero", "mean", "corrupt", "pad", "mask"],
                    help="IG baseline: zero | mean (per-layer centroid) | "
                         "corrupt (content-shuffled prompt) | pad | mask "
                         "(if tokenizer has a mask token)")
    ap.add_argument("--stability", action="store_true",
                    help="proxy-free stability panel: drift (1-cos to final) "
                         "and rsens (||g||*||h||/|logit|) per layer, single "
                         "target logit, no inner-layer head")
    ap.add_argument("--stab-target", type=int, default=0,
                    help="answer-token index to attribute for --stability "
                         "(default 0 = first answer token)")
    ap.add_argument("--loop", action="store_true",
                    help="keep prompting for sentences until empty/EOF")
    args = ap.parse_args()

    print(f"Loading {args.model} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32
    ).to(DEVICE).eval()
    stop_ids = get_stop_token_ids(tokenizer)
    final_norm, lm_head = get_final_norm_and_head(model)

    while True:
        sentence = get_sentence(args)
        if not sentence:
            break

        prompt_ids = build_prompt_ids(tokenizer, sentence)
        answer_ids = greedy_generate(
            model, tokenizer, prompt_ids, stop_ids, args.max_new_tokens)

        if answer_ids.numel() == 0:
            print("  (model produced no answer)\n")
            if not args.loop:
                break
            args.sentence = []
            continue

        full = torch.cat([prompt_ids, answer_ids]).unsqueeze(0)
        with torch.no_grad():
            n_hs = len(model(full, output_hidden_states=True).hidden_states)
        n_layers = n_hs - 1
        layer_indices = pick_layers(n_hs, args.logit_len, args.only_last)

        probs, top_p, top_id = logit_lens_probs(
            model, prompt_ids, answer_ids, layer_indices, final_norm, lm_head)
        sens = gradxembed_sensitivity(
            model, prompt_ids, answer_ids, layer_indices,
            final_norm, lm_head, use_logit=args.logit)
        report(tokenizer, prompt_ids, answer_ids, probs, top_p, top_id,
               sens, layer_indices, n_layers, args.logit)
        report_sensitivity(tokenizer, prompt_ids, answer_ids, sens,
                           layer_indices, n_layers, args.logit)

        if args.ig:
            attr, residual, ig_note = ig_from_layer(
                model, tokenizer, prompt_ids, answer_ids, layer_indices,
                final_norm, lm_head, steps=args.ig_steps,
                baseline_kind=args.ig_baseline)
            if ig_note:
                print(f"  [IG baseline note] {ig_note}\n")
            report_ig(tokenizer, prompt_ids, answer_ids, attr, residual,
                      layer_indices, n_layers, args.ig_steps, args.ig_baseline)

        if args.stability:
            drift, relsens, tgt_id = stability_panel(
                model, prompt_ids, answer_ids, layer_indices,
                final_norm, lm_head, target_idx=args.stab_target)
            report_stability(tokenizer, prompt_ids, answer_ids, drift, relsens,
                             layer_indices, n_layers, tgt_id, args.stab_target)

        if not args.loop:
            break
        args.sentence = []


if __name__ == "__main__":
    main()