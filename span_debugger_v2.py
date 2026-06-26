"""
span_debugger.py — logit-lens + representation-drift CLI tool.

Flow
----
1. Read a sentence (CLI arg, --stdin, or interactive).
2. Greedy-decode an answer from an Instruct model.
3. Concat [prompt + answer], ONE forward pass with output_hidden_states.
4. For each answer token y_t, at its prediction slot, report two things per
   selected layer L, side by side:

     p@L     logit lens: softmax(lm_head(norm(h_L[slot])))[y_t]
             == "how much does the borrowed final head already read y_t off
             layer L". A proxy: lm_head is trained for the FINAL layer only.

     drift@L 1 - cos(h_L[slot], h_final[slot])
             == intrinsic representational convergence, head-free. Small means
             the hidden state is already close to its final value (settled);
             large means it is still being rewritten in the upper layers.

Reading them together
---------------------
- high p@L + low drift@L : decided early AND representation settled.
- high p@L + high drift@L: the lens already reads y_t, but the representation
  is still moving a lot -> the lens converges in its low-rank readout subspace
  while the full residual stream keeps changing (lens = shadow of a still-
  moving trajectory).
- low p@L + high drift@L : still actively being computed.

Layers are the top --logit-len layers (with --only_last) or sampled uniformly
in linear space across [0..n_layers]; the final layer always prints first.
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
    only_last=True: take the top k layers counting back from the final one.

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


# ---------------------------------------------------------------------------
# Lens + drift (one forward pass, no grad)
# ---------------------------------------------------------------------------

@torch.no_grad()
def lens_and_drift(model, prompt_ids, answer_ids, layer_indices,
                   final_norm, lm_head):
    """
    One forward pass. For each answer token y_t (read at its prediction slot)
    and each selected layer L:

        p[t, L]      logit-lens prob of y_t
        top_p[t, L]  prob of the layer's argmax token
        top_id[t, L] id of the layer's argmax token
        drift[t, L]  1 - cos(h_L[slot_t], h_final[slot_t])

    Returns probs, top_p, top_id, drift  (each [T, k], cols = layer_indices).
    """
    full = torch.cat([prompt_ids, answer_ids]).unsqueeze(0)
    out = model(full, output_hidden_states=True)
    hs = out.hidden_states                       # tuple, len = n_layers+1
    L = prompt_ids.numel()
    T = answer_ids.numel()
    pos = torch.arange(L - 1, L - 1 + T, device=DEVICE)   # prediction slots

    h_final = hs[-1][0, pos]                      # [T, d]
    cos = F.cosine_similarity

    p_cols, top_p_cols, top_id_cols, drift_cols = [], [], [], []
    for li in layer_indices:
        h = hs[li][0, pos]                       # [T, d]
        logits = lm_head(final_norm(h))          # [T, V]
        probs_all = F.softmax(logits, dim=-1)
        p = probs_all.gather(1, answer_ids.unsqueeze(1)).squeeze(1)   # [T]
        top_p, top_id = probs_all.max(dim=-1)                         # [T]
        drift = 1.0 - cos(h, h_final, dim=-1)                         # [T]
        p_cols.append(p)
        top_p_cols.append(top_p)
        top_id_cols.append(top_id)
        drift_cols.append(drift)
    return (torch.stack(p_cols, dim=1),
            torch.stack(top_p_cols, dim=1),
            torch.stack(top_id_cols, dim=1),
            torch.stack(drift_cols, dim=1))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def decode_tok(tokenizer, tid):
    return tokenizer.decode([tid])


def report(tokenizer, answer_ids, probs, top_p, top_id, drift,
           layer_indices, n_layers):
    """
    One row per answer token, with p (logit lens) and drift side by side for
    each selected layer.
    """
    target_ids = answer_ids.tolist()
    toks = [decode_tok(tokenizer, t) for t in target_ids]
    P = probs.tolist()
    TP = top_p.tolist()
    TID = top_id.tolist()
    D = drift.tolist()

    p_w, d_w = 26, 11
    hdr = f"{'idx':>3}  {'token':<16}"
    for j, li in enumerate(layer_indices):
        plab = f"p(L{li})" if j == 0 else f"p@L{li}"
        hdr += f"{plab:>{p_w}}{('drift@L'+str(li)):>{d_w}}"
    print()
    print(hdr)
    print("-" * len(hdr))
    for i, tok in enumerate(toks):
        row = f"{i:>3}  {repr(tok):<16}"
        for j in range(len(layer_indices)):
            cell = f"{P[i][j]:.3f}"
            # If the layer's argmax is NOT the target token, append the winner.
            if TID[i][j] != target_ids[i]:
                win_tok = decode_tok(tokenizer, TID[i][j])
                cell += f" [{TP[i][j]:.3f}:{win_tok!r}]"
            row += f"{cell:>{p_w}}{D[i][j]:>{d_w}.4f}"
        print(row)
    print()
    print(f"Layers sampled (of {n_layers}): "
          + ", ".join(f"L{li}" for li in layer_indices)
          + f"   [hidden_states idx; 0=embeddings, {n_layers}=final]")
    print("p(...)  = logit-lens prob of target (borrowed final head; proxy "
          "for inner layers). If layer top-1 != target, [p:'tok'] of the "
          "layer's argmax is shown.")
    print("drift@L = 1 - cos(h_L, h_final) at the token's slot (head-free). "
          "small = settled; large = still being rewritten upstream.")
    print(f"Answer: {tokenizer.decode(answer_ids)!r}")
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
    ap = argparse.ArgumentParser(
        description="Logit-lens + representation-drift span debugger.")
    ap.add_argument("sentence", nargs="*", help="input sentence")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--logit-len", type=int, default=5,
                    help="number of layers to sample (uniform in linear space)")
    ap.add_argument("--only_last", action="store_true",
                    help="pick the top --logit-len layers from the last "
                         "backward (final, final-1, ...) instead of uniform")
    ap.add_argument("--stdin", action="store_true")
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

        probs, top_p, top_id, drift = lens_and_drift(
            model, prompt_ids, answer_ids, layer_indices, final_norm, lm_head)
        report(tokenizer, answer_ids, probs, top_p, top_id, drift,
               layer_indices, n_layers)

        if not args.loop:
            break
        args.sentence = []


if __name__ == "__main__":
    main()