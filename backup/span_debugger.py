"""
span_debugger.py — minimal CLI tool.

Flow
----
1. Read a sentence (CLI arg, --stdin, or interactive prompt).
2. Greedy-decode an answer from an Instruct model.
3. Concat [prompt + answer], run ONE forward pass over the whole thing.
4. For each answer token y_t, print log p(y_t | prompt, y_<t) recovered from
   that single refit pass — alongside p, surprisal (bits), and per-token
   entropy of the predictive distribution.

The point is to *see the spans*: contiguous answer tokens that the model
emits as one committed chunk show up as a run of near-1.0 probabilities
(low surprisal, low entropy) bracketed by a high-surprisal "decision" token
at the span's start. A simple span segmenter marks those boundaries.
"""

import argparse
import math
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

LN2 = math.log(2.0)
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


@torch.no_grad()
def refit_one_pass(model, prompt_ids, answer_ids):
    """
    Single forward pass over [prompt ++ answer].
    Returns, for each answer position t:
        lp[t]   = log p(y_t | prompt, y_<t)
        ent[t]  = entropy (bits) of the predictive distribution at t
    """
    full = torch.cat([prompt_ids, answer_ids]).unsqueeze(0)
    logits = model(full).logits[0]                       # [L, V]
    L = prompt_ids.numel()
    pos = torch.arange(L - 1, L - 1 + answer_ids.numel(), device=DEVICE)
    pred = logits[pos]                                   # [T, V]
    log_probs = F.log_softmax(pred, dim=-1)
    lp = log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(1)
    p = log_probs.exp()
    ent = -(p * log_probs).sum(dim=-1) / LN2             # bits
    return lp, ent


# ---------------------------------------------------------------------------
# Span segmentation
# ---------------------------------------------------------------------------

def segment_spans(surprisal_bits, boundary_bits):
    """
    Greedy: a new span starts whenever a token's surprisal exceeds
    `boundary_bits`. Everything after it until the next high-surprisal token
    belongs to the same committed chunk. Returns a list of (start, end_exclusive).
    """
    spans, start = [], 0
    n = len(surprisal_bits)
    for t in range(1, n):
        if surprisal_bits[t] >= boundary_bits:
            spans.append((start, t))
            start = t
    spans.append((start, n))
    return spans


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def decode_tok(tokenizer, tid):
    return tokenizer.decode([tid])


def bar(p, width=20):
    filled = int(round(p * width))
    return "█" * filled + "·" * (width - filled)


def report(tokenizer, answer_ids, lp, ent, boundary_bits):
    surprisal = (-lp / LN2).tolist()           # bits
    probs = lp.exp().tolist()
    ent_l = ent.tolist()
    toks = [decode_tok(tokenizer, t) for t in answer_ids.tolist()]
    spans = segment_spans(surprisal, boundary_bits)

    # map token index -> span id
    span_of = {}
    for sid, (a, b) in enumerate(spans):
        for i in range(a, b):
            span_of[i] = sid

    print()
    hdr = (f"{'idx':>3} {'sp':>3}  {'token':<16} {'p':>7} "
           f"{'surp(b)':>8} {'ent(b)':>7}  prob")
    print(hdr)
    print("-" * (len(hdr) + 14))
    prev_sid = None
    for i, tok in enumerate(toks):
        sid = span_of[i]
        sep = "  " if sid == prev_sid else "──"   # visual span break
        prev_sid = sid
        print(f"{i:>3} {sid:>3}{sep}{repr(tok):<16} {probs[i]:>7.3f} "
              f"{surprisal[i]:>8.3f} {ent_l[i]:>7.3f}  {bar(probs[i])}")

    print()
    print(f"Boundary threshold: {boundary_bits:.2f} bits surprisal "
          f"(a token at/above this opens a new span).")
    print(f"{len(spans)} span(s) detected:")
    for sid, (a, b) in enumerate(spans):
        chunk = "".join(toks[a:b])
        print(f"  span {sid}: toks[{a}:{b}] {chunk!r}")
    print()
    total_bits = sum(surprisal)
    print(f"Answer: {tokenizer.decode(answer_ids)!r}")
    print(f"Total answer surprisal: {total_bits:.2f} bits "
          f"({len(toks)} tokens, {total_bits/max(len(toks),1):.2f} bits/tok)")
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
    ap = argparse.ArgumentParser(description="Per-token span debugger.")
    ap.add_argument("sentence", nargs="*", help="input sentence")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--boundary-bits", type=float, default=2.0,
                    help="surprisal threshold (bits) that opens a new span")
    ap.add_argument("--stdin", action="store_true",
                    help="read the sentence from stdin")
    ap.add_argument("--loop", action="store_true",
                    help="keep prompting for sentences until empty/EOF")
    args = ap.parse_args()

    print(f"Loading {args.model} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32
    ).to(DEVICE).eval()
    stop_ids = get_stop_token_ids(tokenizer)

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
            args.sentence = []          # force re-prompt
            continue

        # Single refit forward pass over prompt ++ answer.
        lp, ent = refit_one_pass(model, prompt_ids, answer_ids)
        report(tokenizer, answer_ids, lp, ent, args.boundary_bits)

        if not args.loop:
            break
        args.sentence = []              # force re-prompt next iteration


if __name__ == "__main__":
    main()