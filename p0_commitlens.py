"""
p0_commitlens.py — CommitLens P0 (verify phép đo + tách off-manifold).

Bản này thêm patch ON-MANIFOLD để cô lập "commit thật" khỏi "off-manifold onset".

Hai cơ chế patch (--patch):
  replace : ghi đè TOÀN BỘ h_q bằng donor state (cũ; off-manifold ở layer cao)
  delta   : giữ h_q hiện tại, CỘNG alpha*(donor - base_state_at_q) — đẩy theo hướng
            hiệu, giữ phần lớn vector nguyên -> on-manifold hơn. alpha=--alpha.

Hai chiều (--mode): noise (nền counterfactual) | denoise (nền clean).

Câu hỏi cô lập: nếu ở patch=delta, KLplac về NHỎ tại layer cao MÀ commit onset VẪN ~L9
  -> L9 là commit thật. Nếu onset DỊCH khi KL sạch -> L9 cũ là artifact off-manifold.

Margin s=z_y+ - z_y-. KL toàn vocab so với clean. Chỉ verify, KHÔNG kết luận framing.
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def single_token_id(tok, s):
    ids = tok(s, add_special_tokens=False)["input_ids"]
    assert len(ids) == 1, f"{s!r} -> {len(ids)} tokens (cần 1)."
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
    store, handles = {}, []
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
def logits_with_patch(model, layers, ids, q, l_patch, donor, base_at_q=None,
                      patch="replace", alpha=1.0):
    """
    patch=replace: h_q <- donor
    patch=delta  : h_q <- h_q + alpha*(donor - base_at_q)   (giữ phần lớn h_q)
    base_at_q: state của donor-source ở base run tại (l_patch, q); cần cho delta.
    """
    def hook(mod, inp, out):
        tup = isinstance(out, tuple)
        h = out[0] if tup else out
        if patch == "replace":
            h[0, q] = donor
        else:  # delta
            h[0, q] = h[0, q] + alpha * (donor - base_at_q)
        return (h,) + out[1:] if tup else h
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
    logp = F.log_softmax(logits, dim=-1)
    return torch.sum(logp.exp() * (logp - ref_logp)).item()


@torch.no_grad()
def lens_margins(model, layers, ids, q, yp, ym):
    inner = getattr(model, "model", model)
    normf = getattr(inner, "norm", None) or getattr(inner, "final_layernorm")
    head = getattr(model, "lm_head", None) or getattr(model, "embed_out")
    out = []
    for h in capture_states(model, layers, ids, q):
        z = head(normf(h.unsqueeze(0))).squeeze(0)
        out.append((z[yp] - z[ym]).item())
    return out


def readout_onset_hold(lens, Ln):
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
    ap.add_argument("--mode", choices=["noise", "denoise"], default="noise")
    ap.add_argument("--patch", choices=["replace", "delta"], default="replace",
                    help="replace: ghi đè toàn vector. delta: cộng hiệu (on-manifold hơn).")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="hệ số cho patch=delta (1.0 = full hiệu).")
    ap.add_argument("--plot", default="p0_curves.png")
    args = ap.parse_args()

    print(f"Loading {args.model} on {DEVICE} ... (mode={args.mode}, patch={args.patch}, alpha={args.alpha})")
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
        f"prefix lệch {ids_p.shape[1]} vs {ids_m.shape[1]}."
    q = ids_p.shape[1] - 1

    Lp, Lm = clean_logits(model, ids_p), clean_logits(model, ids_m)
    s_plus, s_minus = margin_of(Lp, yp, ym), margin_of(Lm, yp, ym)
    ds = s_plus - s_minus
    logp_clean = F.log_softmax(Lp, dim=-1)
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

    # base run + donor + base_state (state của base run tại q, để tính hiệu cho delta)
    if args.mode == "noise":
        base_ids, base_H, donor_H, base_margin = ids_m, H_minus, H_plus, s_minus
    else:
        base_ids, base_H, donor_H, base_margin = ids_p, H_plus, H_minus, s_plus

    def patched_logits(ids, l, donor, base_at_q):
        return logits_with_patch(model, layers, ids, q, l, donor,
                                 base_at_q=base_at_q, patch=args.patch, alpha=args.alpha)

    S, Sadj, N, P, KL, KLp = [], [], [], [], [], []
    for l in range(Ln):
        lg = patched_logits(base_ids, l, donor_H[l], base_H[l])
        if args.mode == "noise":
            S.append((margin_of(lg, yp, ym) - s_minus) / ds)
        else:
            S.append((s_plus - margin_of(lg, yp, ym)) / ds)
        KL.append(kl_to(lg, logp_clean))

        # necessity luôn replace-style để giữ ý nghĩa "thay clean bằng counterfactual"
        lg_n = logits_with_patch(model, layers, ids_p, q, l, H_minus[l],
                                 base_at_q=H_plus[l], patch=args.patch, alpha=args.alpha)
        N.append((s_plus - margin_of(lg_n, yp, ym)) / ds)

        if H_plac is not None:
            lg_p = patched_logits(base_ids, l, H_plac[l], base_H[l])
            if args.mode == "noise":
                P.append((margin_of(lg_p, yp, ym) - s_minus) / ds)
            else:
                P.append((s_plus - margin_of(lg_p, yp, ym)) / ds)
            KLp.append(kl_to(lg_p, logp_clean))

    if P:
        Sadj = [(S[l] - P[l]) / (1 - P[l] + 1e-6) for l in range(Ln)]
    C = [min(s, n) for s, n in zip(S, N)]
    Cadj = [min(sa, n) for sa, n in zip(Sadj, N)] if P else None

    lens = lens_margins(model, layers, ids_p, q, yp, ym)
    l_read_hold = readout_onset_hold(lens, Ln)
    l_commit_raw = persistent_onset(C, Ln, args.tau)
    l_commit_adj = persistent_onset(Cadj, Ln, args.tau) if P else None

    hdr = f"{'l':>3} {'S':>7} {'Sadj':>7} {'N':>7} {'C':>7} {'Cadj':>7} {'plac':>7} {'KL':>8} {'KLplac':>8} {'lensΔ':>8}"
    print("\n" + hdr); print("-" * len(hdr))
    for l in range(Ln):
        print(f"{l:>3} {S[l]:>7.3f} {(Sadj[l] if P else float('nan')):>7.3f} "
              f"{N[l]:>7.3f} {C[l]:>7.3f} {(Cadj[l] if P else float('nan')):>7.3f} "
              f"{(P[l] if P else float('nan')):>7.3f} {KL[l]:>8.3f} "
              f"{(KLp[l] if P else float('nan')):>8.3f} {lens[l]:>8.3f}")

    print(f"\nreadout onset (hold-to-end)       = {l_read_hold}")
    print(f"commit onset  (raw C>={args.tau})        = {l_commit_raw}")
    print(f"commit onset  (placebo-adj C>={args.tau}) = {l_commit_adj}")

    print("\n--- DIAGNOSTIC (chưa kết luận framing) ---")
    if P:
        print(f"placebo KL nền: tb {sum(KLp)/len(KLp):.2f}, max {max(KLp):.2f}  "
              f"(nhỏ=on-manifold). patch={args.patch}")
        # ngưỡng KL: layer đầu tiên KLplac vượt 2x mức layer-0
        base_kl = KLp[0] if KLp[0] > 1e-6 else 0.05
        thr = next((l for l in range(Ln) if KLp[l] > max(2.0, 3*base_kl)), None)
        print(f"KLplac vượt ngưỡng (off-manifold onset) tại layer = {thr}")
        print(f"-> SO SÁNH: commit onset (adj) = {l_commit_adj} vs off-manifold onset = {thr}")
        print(f"   nếu patch=delta làm KLplac NHỎ mà commit vẫn ~cũ -> commit THẬT.")
        print(f"   nếu commit dịch theo off-manifold onset -> commit là ARTIFACT.")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = list(range(Ln))
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
        a1.axhline(args.tau, ls=":", c="gray"); a1.axhline(0, c="k", lw=.5)
        a1.plot(xs, S, "-o", ms=3, alpha=.4, label="S raw")
        if P:
            a1.plot(xs, Sadj, "-o", ms=3, label="S adj")
            a1.plot(xs, P, "--", c="red", alpha=.6, label="placebo")
        a1.plot(xs, N, "-s", ms=3, alpha=.6, label="N")
        if l_commit_adj is not None:
            a1.axvline(l_commit_adj, c="purple", ls="--", label=f"commit={l_commit_adj}")
        a1.set_xlabel("layer"); a1.set_ylabel("norm effect"); a1.legend(fontsize=8); a1.grid(alpha=.3)
        a1.set_title(f"effect [{args.mode}/{args.patch}]")
        a2.plot(xs, KL, "-o", ms=3, label="KL (true patch)")
        if P:
            a2.plot(xs, KLp, "--", c="red", label="KL placebo")
        a2.set_xlabel("layer"); a2.set_ylabel("KL to clean"); a2.legend(fontsize=8); a2.grid(alpha=.3)
        a2.set_title("off-manifold (KL)")
        plt.tight_layout(); plt.savefig(args.plot, dpi=130)
        print(f"\nsaved plot -> {args.plot}")
    except Exception as e:
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()