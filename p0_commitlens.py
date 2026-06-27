"""
p0_commitlens.py — CommitLens P0 smoke test trên ĐÚNG MỘT cặp.

Đo 3 thứ FALSIFIABLE (không cần ground truth về onset):
  S_l : patch clean state @q (sau layer l) vào run counterfactual -> margin recover? (sufficiency)
  N_l : patch counterfactual state @q vào run clean -> margin sụp? (necessity)
  placebo_l : patch state @q từ prompt VÔ CAN -> margin PHẢI ~ phẳng (off-manifold control)
  + readout onset (logit lens favor y+) để tính G = l*(commit) - l*(readout)

Margin: s(x) = z_{y+} - z_{y-}  (logit margin, KHÔNG softmax).
Patch tại vị trí dự đoán q = token cuối, thay hidden sau block l, các block trên chạy tiếp.

Chạy:
  python p0_commitlens.py \
    --x-plus "The capital of France is" \
    --x-minus "The capital of Germany is" \
    --y-plus " Paris" --y-minus " Berlin" \
    --x-placebo "The weather today is rather" \
    --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def single_token_id(tok, s):
    ids = tok(s, add_special_tokens=False)["input_ids"]
    assert len(ids) == 1, f"{s!r} -> {len(ids)} tokens (cần 1). Sửa answer."
    return ids[0]


def get_layers(model):
    # llama: model.model.layers ; gpt2: transformer.h
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "transformer", inner), "h")
    return layers


def encode(tok, text):
    return tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEVICE)


# ---------------------------------------------------------------------------
# capture hidden states after each decoder block at position q (full forward)
# ---------------------------------------------------------------------------
@torch.no_grad()
def capture_states(model, layers, ids, q):
    """Return list states[l] = hidden AFTER block l, at position q. l=0..L-1."""
    store = {}
    handles = []
    for l, layer in enumerate(layers):
        def mk(l):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                store[l] = h[0, q].detach().clone()
            return hook
        handles.append(layer.register_forward_hook(mk(l)))
    try:
        model(ids)
    finally:
        for h in handles:
            h.remove()
    return [store[l] for l in range(len(layers))]


# ---------------------------------------------------------------------------
# run forward on `ids` but at block `l_patch`, overwrite position q with `vec`
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_with_patch(model, layers, ids, q, l_patch, vec):
    handle = None
    def hook(mod, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            h[0, q] = vec
            return (h,) + out[1:]
        else:
            out[0, q] = vec
            return out
    handle = layers[l_patch].register_forward_hook(hook)
    try:
        logits = model(ids).logits[0, -1]
    finally:
        handle.remove()
    return logits


@torch.no_grad()
def margin(model, ids, yp, ym):
    logits = model(ids).logits[0, -1]
    return (logits[yp] - logits[ym]).item()


@torch.no_grad()
def margin_patched(model, layers, ids, q, l, vec, yp, ym):
    logits = forward_with_patch(model, layers, ids, q, l, vec)
    return (logits[yp] - logits[ym]).item()


# ---------------------------------------------------------------------------
# logit-lens readout onset: earliest layer where lens favors y+ over y-
# ---------------------------------------------------------------------------
@torch.no_grad()
def lens_margins(model, layers, ids, q, yp, ym):
    inner = getattr(model, "model", model)
    normf = getattr(inner, "norm", None) or getattr(inner, "final_layernorm")
    head = getattr(model, "lm_head", None) or getattr(model, "embed_out")
    states = capture_states(model, layers, ids, q)
    out = []
    for h in states:
        z = head(normf(h.unsqueeze(0))).squeeze(0)
        out.append((z[yp] - z[ym]).item())
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-plus", required=True)
    ap.add_argument("--x-minus", required=True)
    ap.add_argument("--y-plus", required=True)
    ap.add_argument("--y-minus", required=True)
    ap.add_argument("--x-placebo", default=None,
                    help="prompt vô can cùng độ dài cho off-manifold control")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--plot", default="p0_curves.png")
    args = ap.parse_args()

    print(f"Loading {args.model} on {DEVICE} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32).to(DEVICE).eval()
    layers = get_layers(model)
    Ln = len(layers)

    yp = single_token_id(tok, args.y_plus)
    ym = single_token_id(tok, args.y_minus)

    ids_p = encode(tok, args.x_plus)
    ids_m = encode(tok, args.x_minus)
    assert ids_p.shape[1] == ids_m.shape[1], \
        f"prefix lệch: {ids_p.shape[1]} vs {ids_m.shape[1]} -> q misaligned. Sửa cặp."
    q = ids_p.shape[1] - 1

    s_plus = margin(model, ids_p, yp, ym)
    s_minus = margin(model, ids_m, yp, ym)
    ds = s_plus - s_minus
    print(f"\ns+ = {s_plus:+.3f}   s- = {s_minus:+.3f}   Δs = {ds:+.3f}")
    if ds <= 0:
        print("Δs <= 0: cặp không hợp lệ (model không tách contrast). Dừng.")
        return

    # capture clean / counterfactual states at q
    H_plus = capture_states(model, layers, ids_p, q)
    H_minus = capture_states(model, layers, ids_m, q)

    # placebo states (unrelated prompt, must match length)
    H_plac = None
    if args.x_placebo:
        ids_plac = encode(tok, args.x_placebo)
        if ids_plac.shape[1] == ids_p.shape[1]:
            H_plac = capture_states(model, layers, ids_plac, q)
        else:
            print(f"[warn] placebo length {ids_plac.shape[1]} != {ids_p.shape[1]}, "
                  f"bỏ placebo. Cho 1 prompt vô can cùng độ dài để bật control.")

    # per-layer S, N, placebo
    S, N, P = [], [], []
    for l in range(Ln):
        # sufficiency: clean state @q vào run counterfactual
        sl = margin_patched(model, layers, ids_m, q, l, H_plus[l], yp, ym)
        S.append((sl - s_minus) / ds)
        # necessity: counterfactual state @q vào run clean
        nl = margin_patched(model, layers, ids_p, q, l, H_minus[l], yp, ym)
        N.append((s_plus - nl) / ds)
        # placebo: unrelated state @q vào run counterfactual (kỳ vọng ~0)
        if H_plac is not None:
            pl = margin_patched(model, layers, ids_m, q, l, H_plac[l], yp, ym)
            P.append((pl - s_minus) / ds)

    C = [min(s, n) for s, n in zip(S, N)]

    # persistent commitment onset: earliest l where min_{r>=l} C_r >= tau
    def persistent_onset(C, tau):
        for l in range(Ln):
            if min(C[l:]) >= tau:
                return l
        return None
    l_commit = persistent_onset(C, args.tau)

    # readout onset: earliest layer where lens margin > 0
    lens = lens_margins(model, layers, ids_p, q, yp, ym)
    l_read = next((l for l in range(Ln) if lens[l] > 0), None)

    G = (l_commit - l_read) if (l_commit is not None and l_read is not None) else None

    # ---- report ----
    print(f"\n{'l':>3} {'S':>8} {'N':>8} {'C=min':>8} "
          + (f"{'placebo':>8} " if P else "") + f"{'lensΔ':>8}")
    print("-" * (3 + 8*4 + (9 if P else 0)))
    for l in range(Ln):
        row = f"{l:>3} {S[l]:>8.3f} {N[l]:>8.3f} {C[l]:>8.3f} "
        if P:
            row += f"{P[l]:>8.3f} "
        row += f"{lens[l]:>8.3f}"
        print(row)

    print(f"\nreadout onset l*(lens>0)      = {l_read}")
    print(f"commit onset  l*(persist C>={args.tau}) = {l_commit}")
    print(f"G = l_commit - l_read         = {G}")

    # ---- verdict ----
    print("\n--- P0 VERDICT ---")
    print(f"(1) S leo tới ~1?         max S = {max(S):.2f}  "
          f"-> {'OK' if max(S) > 0.8 else 'YẾU: can thiệp không đủ tác dụng'}")
    if P:
        print(f"(2) placebo phẳng?        max|placebo| = {max(abs(p) for p in P):.2f}  "
              f"-> {'OK' if max(abs(p) for p in P) < 0.3 else 'CONFOUND: off-manifold ăn, S vô giá trị'}")
    else:
        print("(2) placebo: CHƯA CHẠY (cho --x-placebo cùng độ dài để bật)")
    if G is not None:
        print(f"(3) readout sớm hơn commit? G = {G}  "
              f"-> {'OK: có khoảng cách, framing sống' if G > 0 else 'G<=0: readability=commitment, framing YẾU'}")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = list(range(Ln))
        plt.figure(figsize=(8, 5))
        plt.axhline(args.tau, ls=":", c="gray", lw=1, label=f"τ={args.tau}")
        plt.axhline(0, c="k", lw=0.5)
        plt.plot(xs, S, "-o", ms=3, label="S (sufficiency)")
        plt.plot(xs, N, "-o", ms=3, label="N (necessity)")
        plt.plot(xs, C, "-", lw=2, label="C=min(S,N)")
        if P:
            plt.plot(xs, P, "--", c="red", label="placebo")
        if l_read is not None:
            plt.axvline(l_read, c="green", ls="--", lw=1, label=f"readout l*={l_read}")
        if l_commit is not None:
            plt.axvline(l_commit, c="purple", ls="--", lw=1, label=f"commit l*={l_commit}")
        plt.xlabel("layer l"); plt.ylabel("normalized effect")
        plt.title(f"CommitLens P0: {args.x_plus!r} -> {args.y_plus!r}")
        plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(args.plot, dpi=130)
        print(f"\nsaved plot -> {args.plot}")
    except Exception as e:
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()