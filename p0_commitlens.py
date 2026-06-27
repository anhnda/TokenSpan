"""
p0_commitlens.py — CommitLens P0 smoke test (verify PHÉP ĐO, chưa kết luận gì).

Mục tiêu phiên bản này: làm SẠCH phép đo trước khi tin bất kỳ con số nào.
Thêm so với bản trước:
  - S_adj = (S - placebo)/(1 - placebo): trừ nền off-manifold (1a)
  - --mode denoise: nền=clean, làm hỏng q bằng counterfactual (thường on-manifold hơn) (1b)
  - KL toàn vocab song song với margin: robust hơn margin 2-token (2a)
  - readout onset = earliest lens>0 VÀ GIỮ tới cuối (bỏ nhiễu early-layer)
  - placebo KL: nếu patch vô can mà KL nhỏ -> on-manifold; KL lớn -> off-manifold

KHÔNG in "VERDICT framing sống/chết" nữa — bản này chỉ verify đo. Kết luận để sau.

Margin: s(x) = z_{y+} - z_{y-} (logit). KL: KL(patched_probs || clean_probs) toàn vocab.

Chạy (noising, mặc định):
  python p0_commitlens.py --x-plus "The capital of France is" \
    --x-minus "The capital of Germany is" --y-plus " Paris" --y-minus " Berlin" \
    --x-placebo "A wooden table in the" --model meta-llama/Llama-3.2-1B-Instruct

Chạy denoising:  thêm  --mode denoise
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def single_token_id(tok, s):
    ids = tok(s, add_special_tokens=False)["input_ids"]
    assert len(ids) == 1, f"{s!r} -> {len(ids)} tokens (cần 1). Sửa answer."
    return ids[0]


def get_layers(model):
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "transformer", inner), "h")
    return layers


def encode(tok, text):
    return tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEVICE)


@torch.no_grad()
def capture_states(model, layers, ids, q):
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


@torch.no_grad()
def logits_with_patch(model, layers, ids, q, l_patch, vec):
    def hook(mod, inp, out):
        if isinstance(out, tuple):
            h = out[0]; h[0, q] = vec; return (h,) + out[1:]
        out[0, q] = vec; return out
    handle = layers[l_patch].register_forward_hook(hook)
    try:
        logits = model(ids).logits[0, -1]
    finally:
        handle.remove()
    return logits


@torch.no_grad()
def clean_logits(model, ids):
    return model(ids).logits[0, -1]


def margin_of(logits, yp, ym):
    return (logits[yp] - logits[ym]).item()


def kl_to(logits, ref_logp):
    # KL(p_patched || p_ref) ở vị trí cuối
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    return torch.sum(p * (logp - ref_logp)).item()


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


def readout_onset_hold(lens, Ln):
    """earliest l where lens>0 AND stays >0 to the end (bỏ nhiễu early-layer)."""
    for l in range(Ln):
        if all(lens[r] > 0 for r in range(l, Ln)):
            return l
    return None


def persistent_onset(C, Ln, tau):
    for l in range(Ln):
        if min(C[l:]) >= tau:
            return l
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-plus", required=True)
    ap.add_argument("--x-minus", required=True)
    ap.add_argument("--y-plus", required=True)
    ap.add_argument("--y-minus", required=True)
    ap.add_argument("--x-placebo", default=None)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--mode", choices=["noise", "denoise"], default="noise",
                    help="noise: nền counterfactual, patch clean @q (sufficiency cổ điển). "
                         "denoise: nền clean, patch counterfactual @q (thường on-manifold hơn).")
    ap.add_argument("--plot", default="p0_curves.png")
    args = ap.parse_args()

    print(f"Loading {args.model} on {DEVICE} ... (mode={args.mode})")
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
        f"prefix lệch {ids_p.shape[1]} vs {ids_m.shape[1]} -> q misaligned."
    q = ids_p.shape[1] - 1

    Lp = clean_logits(model, ids_p)
    Lm = clean_logits(model, ids_m)
    s_plus, s_minus = margin_of(Lp, yp, ym), margin_of(Lm, yp, ym)
    ds = s_plus - s_minus
    logp_clean = F.log_softmax(Lp, dim=-1)   # ref cho KL = phân phối clean (đích mong muốn)
    print(f"\ns+ = {s_plus:+.3f}  s- = {s_minus:+.3f}  Δs = {ds:+.3f}")
    if ds <= 0:
        print("Δs <= 0: cặp không hợp lệ. Dừng."); return

    H_plus = capture_states(model, layers, ids_p, q)
    H_minus = capture_states(model, layers, ids_m, q)

    H_plac = None
    if args.x_placebo:
        ids_plac = encode(tok, args.x_placebo)
        if ids_plac.shape[1] == ids_p.shape[1]:
            H_plac = capture_states(model, layers, ids_plac, q)
        else:
            print(f"[warn] placebo len {ids_plac.shape[1]} != {ids_p.shape[1]}; bỏ placebo.")

    # base run phụ thuộc mode
    if args.mode == "noise":
        base_ids, donor, base_margin = ids_m, H_plus, s_minus  # patch clean vào counterfactual
    else:
        base_ids, donor, base_margin = ids_p, H_minus, s_plus  # patch counterfactual vào clean

    S, Sadj, N, P, KL, KLp = [], [], [], [], [], []
    for l in range(Ln):
        # sufficiency-style: donor[l] @q vào base run
        lg = logits_with_patch(model, layers, base_ids, q, l, donor[l])
        if args.mode == "noise":
            S.append((margin_of(lg, yp, ym) - s_minus) / ds)
        else:
            # denoise: đo mức margin BỊ HỎNG đi từ clean -> dùng (s+ - patched)/ds ~ "necessity"
            S.append((s_plus - margin_of(lg, yp, ym)) / ds)
        KL.append(kl_to(lg, logp_clean))

        # necessity (chiều ngược, luôn tính cho đối chiếu)
        lg_n = logits_with_patch(model, layers, ids_p, q, l, H_minus[l])
        N.append((s_plus - margin_of(lg_n, yp, ym)) / ds)

        if H_plac is not None:
            lg_p = logits_with_patch(model, layers, base_ids, q, l, H_plac[l])
            if args.mode == "noise":
                P.append((margin_of(lg_p, yp, ym) - s_minus) / ds)
            else:
                P.append((s_plus - margin_of(lg_p, yp, ym)) / ds)
            KLp.append(kl_to(lg_p, logp_clean))

    # S điều chỉnh theo placebo: phần vượt nền artifact
    if P:
        Sadj = [(S[l] - P[l]) / (1 - P[l] + 1e-6) for l in range(Ln)]

    C = [min(s, n) for s, n in zip(S, N)]
    Cadj = [min(sa, n) for sa, n in zip(Sadj, N)] if P else None

    lens = lens_margins(model, layers, ids_p, q, yp, ym)
    l_read_naive = next((l for l in range(Ln) if lens[l] > 0), None)
    l_read_hold = readout_onset_hold(lens, Ln)
    l_commit_raw = persistent_onset(C, Ln, args.tau)
    l_commit_adj = persistent_onset(Cadj, Ln, args.tau) if P else None

    # ---- bảng ----
    hdr = f"{'l':>3} {'S':>7} {'Sadj':>7} {'N':>7} {'C':>7} {'Cadj':>7} "
    hdr += f"{'plac':>7} {'KL':>8} {'KLplac':>8} {'lensΔ':>8}"
    print("\n" + hdr); print("-" * len(hdr))
    for l in range(Ln):
        print(f"{l:>3} {S[l]:>7.3f} "
              f"{(Sadj[l] if P else float('nan')):>7.3f} "
              f"{N[l]:>7.3f} {C[l]:>7.3f} "
              f"{(Cadj[l] if P else float('nan')):>7.3f} "
              f"{(P[l] if P else float('nan')):>7.3f} "
              f"{KL[l]:>8.3f} "
              f"{(KLp[l] if P else float('nan')):>8.3f} "
              f"{lens[l]:>8.3f}")

    print(f"\nreadout onset (naive lens>0)      = {l_read_naive}")
    print(f"readout onset (hold-to-end)       = {l_read_hold}")
    print(f"commit onset  (raw C>={args.tau})        = {l_commit_raw}")
    print(f"commit onset  (placebo-adj C>={args.tau}) = {l_commit_adj}")

    # ---- chỉ báo chẩn đoán phép đo, KHÔNG phán framing ----
    print("\n--- DIAGNOSTIC phép đo (chưa kết luận framing) ---")
    if P:
        print(f"placebo margin nền: trung bình {sum(P)/len(P):+.2f}, "
              f"max|.| {max(abs(p) for p in P):.2f}")
        print(f"placebo KL nền:     trung bình {sum(KLp)/len(KLp):.3f}, "
              f"max {max(KLp):.3f}   (nhỏ = on-manifold, lớn = off)")
        print(f"-> nếu placebo margin cao nhưng Sadj vẫn nhấc rõ ở layer cao, "
              f"signal thật nằm ở Sadj/Cadj, không phải S thô.")
    else:
        print("placebo CHƯA chạy (cho --x-placebo cùng độ dài).")
    if l_read_hold is not None and l_commit_adj is not None:
        print(f"G(hold, adj) = commit_adj - read_hold = {l_commit_adj - l_read_hold}  "
              f"(diễn giải sau khi có >=3 cặp)")

    # ---- plot ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = list(range(Ln))
        plt.figure(figsize=(9, 5))
        plt.axhline(args.tau, ls=":", c="gray", lw=1)
        plt.axhline(0, c="k", lw=0.5)
        plt.plot(xs, S, "-o", ms=3, alpha=.4, label="S raw")
        if P:
            plt.plot(xs, Sadj, "-o", ms=3, label="S adj (−placebo)")
            plt.plot(xs, P, "--", c="red", alpha=.6, label="placebo")
        plt.plot(xs, N, "-s", ms=3, alpha=.6, label="N")
        if l_read_hold is not None:
            plt.axvline(l_read_hold, c="green", ls="--", lw=1, label=f"read(hold)={l_read_hold}")
        if l_commit_adj is not None:
            plt.axvline(l_commit_adj, c="purple", ls="--", lw=1, label=f"commit(adj)={l_commit_adj}")
        plt.xlabel("layer"); plt.ylabel("normalized effect")
        plt.title(f"P0 [{args.mode}]: {args.x_plus!r} -> {args.y_plus!r}")
        plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(args.plot, dpi=130)
        print(f"\nsaved plot -> {args.plot}")
    except Exception as e:
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()