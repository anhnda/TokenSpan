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

        if not args.loop:
            break
        args.sentence = []


if __name__ == "__main__":
    main()