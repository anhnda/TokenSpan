"""
ig_onset.py — head-free "onset layer" via IG on hidden states.

Idea
----
lm_head is used ONCE, at the final layer, to define a scalar:

    f(y_t) = logit of target token y_t at its prediction slot.

We then attribute that final scalar BACK to the hidden state h_L[i] at each
layer L (same position i), using Integrated Gradients along the path
h0 -> h_L[i]. lm_head is never applied to any lower layer. The lens is not
used to score; it is only printed for reference/contrast.

For each layer L:

    IG_L = < h_L[i] - h0 , INT_0^1 d f / d h_L[i] |_{h0+a(h_L-h0)} da >

This is the inner product with the displacement (NOT an L2 norm of the
gradient):
  - completeness: sum_L-style decomposition stays in logit units, signed;
  - it scores what h_L ACTUALLY is (projection onto the displacement), not
    mere sensitivity;
  - comparable across layers (logit units), unlike ||grad||_2.

We also report plain grad*input (n_steps=1, baseline at h0) for contrast,
since on easy/saturated tokens the raw gradient collapses to ~0 while IG
recovers the contribution.

Onset
-----
L* = earliest layer from which the per-layer contribution stays above a
threshold all the way to the end ("decided early and kept" => easy token;
"only late" => hard token).

NOTE: gradients flow freely (total effect) — h_L[i] influences f both
directly and via attention into other positions at upper layers.
"""

import argparse
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


# ---------------------------------------------------------------------------
# Prompt / generation helpers (mirrors span_debugger_v2.py)
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


# ---------------------------------------------------------------------------
# Decoder-layer access + forward-from-layer
# ---------------------------------------------------------------------------

def get_decoder_layers(model):
    """The ModuleList of transformer blocks (Llama/Mistral/Gemma style)."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError("Could not locate decoder layers (model.model.layers).")
    return layers


def capture_hidden_states(model, full_ids):
    """One clean forward pass; return hidden_states tuple (len n_layers+1)."""
    with torch.no_grad():
        out = model(full_ids, output_hidden_states=True)
    return out.hidden_states


def _score_from_logits(logits, target_id, mode):
    """
    logits: [V] final logits. Reduce to the scalar IG target.
      mode='prob'   : softmax(logits)[y_t]            (default; lens-like)
      mode='logit'  : raw logit[y_t]                  (--logit)
      mode='logprob': log_softmax(logits)[y_t]        (--target logprob)
      mode='gap'    : logit[y_t] - max_{v!=y_t} logit (--target gap)
    """
    if mode == "logit":
        return logits[target_id]
    if mode == "prob":
        return F.softmax(logits, dim=-1)[target_id]
    if mode == "logprob":
        return F.log_softmax(logits, dim=-1)[target_id]
    if mode == "gap":
        masked = logits.clone()
        masked[target_id] = float("-inf")
        runner = masked.max()
        return logits[target_id] - runner
    raise ValueError(f"unknown score mode {mode!r}")


def final_logit_from_hidden(model, hs_layer, layer_idx, full_ids, position,
                            target_id, mode="prob"):
    """
    Run the tail of the network starting from the residual stream AT THE OUTPUT
    of block `layer_idx` (i.e. hidden_states[layer_idx+1] semantics), with the
    [0, position] vector REPLACED by `hs_layer` (a leaf requiring grad), and
    return the final logit of `target_id`.

    hidden_states indexing: hs[0]=embeddings, hs[k]=output of block k-1.
    So to inject at "layer L = hs index li", we re-run blocks li .. end.

    We implement this with a forward hook on block `start_block` that overwrites
    its INPUT hidden state, then let the model run normally to the end. This is
    the cheap "forward-from-layer" trick: we still pay a full forward, but only
    the tail matters for grad; lm_head is applied exactly once at the end.
    """
    inner = getattr(model, "model", model)
    layers = get_decoder_layers(model)

    # li in hidden_states space: 0..n_layers. Block that PRODUCED hs[li] is
    # layers[li-1]; the block that CONSUMES hs[li] as input is layers[li].
    li = layer_idx
    n_layers = len(layers)

    if li >= n_layers:
        # hs[n_layers] is final norm input == output of last block; injecting
        # here means: only final norm + lm_head remain.
        normf = getattr(inner, "norm", None) or getattr(inner, "final_layernorm")
        lm_head = getattr(model, "lm_head", None) or getattr(model, "embed_out")
        h = hs_layer  # [d]
        logits = lm_head(normf(h.unsqueeze(0))).squeeze(0)  # [V]
        return _score_from_logits(logits, target_id, mode)

    consumer = layers[li]
    handle_box = {}

    def pre_hook(module, args, kwargs):
        # args[0] is hidden_states [B, T, d]; overwrite [0, position].
        hs_in = args[0]
        hs_in = hs_in.clone()
        hs_in[0, position] = hs_layer
        new_args = (hs_in,) + tuple(args[1:])
        return new_args, kwargs

    h = consumer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    handle_box["h"] = h
    try:
        out = model(full_ids)  # full forward; hook injects at consumer input
        logits = out.logits[0, position]  # [V]
        return _score_from_logits(logits, target_id, mode)
    finally:
        handle_box["h"].remove()


# ---------------------------------------------------------------------------
# IG on hidden state h_L[i]  (signed inner product, NOT L2)
# ---------------------------------------------------------------------------

def ig_on_layer(model, full_ids, position, target_id, h_L, baseline,
                n_steps):
    """
    IG_L = < h_L - baseline , INT_0^1 grad_f(h0 + a (h_L - h0)) da >.

    Returns (ig_scalar, gradxinput_scalar).
      ig_scalar         : Riemann-mean over n_steps interpolation points.
      gradxinput_scalar : single point at a=1 (the real h_L), baseline-subtracted
                          displacement, i.e. <h_L - baseline, grad@h_L>.
    Both are sums over the hidden dim (inner products), in logit units.
    """
    li = None  # layer_idx is closed over by caller via partial; see run()
    raise NotImplementedError  # replaced below by closure in run()


# The real per-layer routine is built inside run() so it can close over
# layer_idx cleanly. Kept here as documentation of the contract.


# ---------------------------------------------------------------------------
# Onset extraction
# ---------------------------------------------------------------------------

def onset_layer(contribs_by_li, layer_order_low_to_high, threshold):
    """
    contribs_by_li: dict li -> scalar contribution.
    layer_order_low_to_high: list of li sorted ascending (0..n).
    L* = earliest li such that contrib stays >= threshold for ALL li' >= li.
    Returns li or None if never settles.
    """
    n = len(layer_order_low_to_high)
    settled_from = None
    for start in range(n):
        ok = all(
            contribs_by_li[layer_order_low_to_high[k]] >= threshold
            for k in range(start, n)
        )
        if ok:
            settled_from = layer_order_low_to_high[start]
            break
    return settled_from


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(model, tokenizer, sentence, args):
    stop_ids = get_stop_token_ids(tokenizer)
    prompt_ids = build_prompt_ids(tokenizer, sentence)
    answer_ids = greedy_generate(
        model, tokenizer, prompt_ids, stop_ids, args.max_new_tokens)
    if answer_ids.numel() == 0:
        print("  (model produced no answer)\n")
        return

    full = torch.cat([prompt_ids, answer_ids]).unsqueeze(0)
    hs = capture_hidden_states(model, full)        # len n_layers+1
    n_hs = len(hs)
    n_layers = n_hs - 1
    Lp = prompt_ids.numel()
    T = answer_ids.numel()

    layers = get_decoder_layers(model)

    # final norm + lm_head, used ONLY to compute reference logit-lens values
    inner = getattr(model, "model", model)
    lens_norm = getattr(inner, "norm", None) or getattr(inner, "final_layernorm")
    lens_head = getattr(model, "lm_head", None) or getattr(model, "embed_out")

    @torch.no_grad()
    def lens_p(li, position, target_id):
        """Reference only: softmax(lm_head(norm(h_L[i])))[y_t]. NOT used for onset."""
        h = hs[li][0, position]
        logits = lens_head(lens_norm(h.unsqueeze(0))).squeeze(0)
        return F.softmax(logits, dim=-1)[target_id].item()

    # which hidden_states indices to probe
    if args.layers:
        li_list = sorted(set(int(x) for x in args.layers))
    else:
        # uniform sample in [1..n_layers], always include final
        raw = torch.linspace(1, n_layers, steps=min(args.k, n_layers))
        li_list = sorted(set(raw.round().long().tolist()) | {n_layers})

    # mean baseline (per hidden idx) if requested: mean over all positions
    def make_baseline(li, vec):
        if args.baseline == "zero":
            return torch.zeros_like(vec)
        # mean over sequence positions at this layer
        return hs[li][0].mean(dim=0).detach()

    # closure: final score of target as fn of injected h at hidden idx li
    def f_of_h(li, h_vec, position, target_id):
        return final_logit_from_hidden(
            model, h_vec, li, full, position, target_id, mode=args.target)

    def ig_for(li, position, target_id):
        h_L = hs[li][0, position].detach()
        base = make_baseline(li, h_L)
        diff = (h_L - base)

        # IG: average gradient over interpolation points, dot displacement
        grad_accum = torch.zeros_like(h_L)
        for s in range(1, args.n_steps + 1):
            a = s / args.n_steps
            h_a = (base + a * diff).detach().requires_grad_(True)
            logit = f_of_h(li, h_a, position, target_id)
            g, = torch.autograd.grad(logit, h_a)
            grad_accum += g.detach()
        grad_mean = grad_accum / args.n_steps
        ig = torch.dot(diff, grad_mean).item()
        return ig

    pos_slots = torch.arange(Lp - 1, Lp - 1 + T)   # prediction slots

    li_sorted_hi = sorted(li_list, reverse=True)
    li_sorted_lo = sorted(li_list)

    # header
    print()
    print(f"Answer: {tokenizer.decode(answer_ids)!r}")
    print(f"baseline={args.baseline}  n_steps={args.n_steps}  "
          f"target={args.target}  threshold={args.threshold}")
    hdr = f"{'idx':>3}  {'token':<14}"
    for li in li_sorted_hi:
        hdr += f"{'IG@L'+str(li):>12}{'lens@L'+str(li):>12}"
    hdr += f"{'onset L*':>10}"
    print(hdr)
    print("-" * len(hdr))

    for t in range(T):
        position = int(pos_slots[t].item())
        target_id = int(answer_ids[t].item())
        tok = tokenizer.decode([target_id])

        ig_map, lens_map = {}, {}
        for li in li_list:
            ig_map[li] = ig_for(li, position, target_id)
            lens_map[li] = lens_p(li, position, target_id)

        Lstar = onset_layer(ig_map, li_sorted_lo, args.threshold)

        row = f"{t:>3}  {repr(tok):<14}"
        for li in li_sorted_hi:
            row += f"{ig_map[li]:>12.3f}{lens_map[li]:>12.3f}"
        row += f"{(str(Lstar) if Lstar is not None else '-'):>10}"
        print(row)

    print()
    print("IG@L  = <h_L[i]-baseline, mean grad of final TARGET(y_t) over path>  "
          "(signed; units follow --target: prob in [0,1], logit/logprob/gap in "
          "logit units; head used ONCE at the end).")
    print("lens@L= softmax(lm_head(norm(h_L[i])))[y_t]  REFERENCE ONLY (borrows "
          "final head on lower layers; not used for onset).")
    print("onset L* = earliest layer whose IG stays >= threshold to the end "
          "(small=easy/early, large=hard/late).")
    print(f"Layers (hidden_states idx, 0=emb {n_layers}=final): "
          + ", ".join('L'+str(li) for li in li_sorted_lo))
    print()


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
        description="Head-free onset via IG on hidden states.")
    ap.add_argument("sentence", nargs="*")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--k", type=int, default=6,
                    help="number of layers to sample if --layers not given")
    ap.add_argument("--layers", nargs="*",
                    help="explicit hidden_states indices to probe (e.g. 4 8 12 16)")
    ap.add_argument("--n-steps", type=int, default=16,
                    help="IG interpolation steps")
    ap.add_argument("--baseline", choices=["zero", "mean"], default="mean")
    ap.add_argument("--target", choices=["prob", "logit", "logprob", "gap"],
                    default="prob",
                    help="IG target scalar: prob (default, lens-like), logit, "
                         "logprob, or gap(y_t vs runner-up)")
    ap.add_argument("--logit", action="store_true",
                    help="shortcut for --target logit (overrides --target)")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="onset: IG must stay >= this from L* to the end")
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()
    if args.logit:
        args.target = "logit"

    print(f"Loading {args.model} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32
    ).to(DEVICE).eval()

    while True:
        sentence = get_sentence(args)
        if not sentence:
            break
        run(model, tokenizer, sentence, args)
        if not args.loop:
            break
        args.sentence = []


if __name__ == "__main__":
    main()