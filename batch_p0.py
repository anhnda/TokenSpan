"""
batch_p0.py — chạy P0 cho NHIỀU cặp một lượt, in bảng tóm tắt.

Mục tiêu: thấy ngay binding (kỳ vọng G>0) vs factual (kỳ vọng G~0) có tách không,
với CÙNG setup sạch: denoise + delta + placebo cùng khung.

In mỗi cặp: Δs, readout onset, commit onset, G, max KLplac (kiểm on-manifold).
Cảnh báo cặp Δs<=0 (bỏ) hoặc placebo lệch độ dài (bỏ placebo).

Tái dùng logic từ p0_commitlens.py (import trực tiếp).

Chạy:
  python batch_p0.py commitbench_pairs.jsonl \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --mode denoise --patch delta --alpha 0.3 --tau 0.15

LƯU Ý tau: với delta alpha<1, hiệu ứng nhỏ hơn replace, nên hạ tau cho hợp
(alpha=0.3 -> S đỉnh ~0.3, dùng tau ~0.15). Onset = chỗ Sadj vượt tau VÀ giữ.
So sánh G GIỮA các cặp quan trọng hơn giá trị onset tuyệt đối.
"""

import argparse, json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# import các hàm đã viết, tránh lặp code
import importlib.util
spec = importlib.util.spec_from_file_location("p0", "p0_commitlens.py")
p0 = importlib.util.module_from_spec(spec)
# p0_commitlens.py có main() gọi argparse; chỉ load định nghĩa, không chạy main
import sys
_src = open("p0_commitlens.py").read().split("\ndef main()")[0]
exec(compile(_src, "p0_commitlens.py", "exec"), p0.__dict__)

DEVICE = p0.DEVICE


def measure_pair(model, layers, tok, ex, args):
    """Trả về dict kết quả cho một cặp, hoặc None nếu cặp không hợp lệ."""
    Ln = len(layers)
    try:
        yp = p0.single_token_id(tok, ex["y_plus"])
        ym = p0.single_token_id(tok, ex["y_minus"])
    except AssertionError as e:
        return {"id": ex["id"], "skip": f"answer multi-token: {e}"}

    ids_p = p0.encode(tok, ex["x_plus"])
    ids_m = p0.encode(tok, ex["x_minus"])
    if ids_p.shape[1] != ids_m.shape[1]:
        return {"id": ex["id"], "skip": f"prefix lệch {ids_p.shape[1]}!={ids_m.shape[1]}"}
    q = ids_p.shape[1] - 1

    Lp, Lm = p0.clean_logits(model, ids_p), p0.clean_logits(model, ids_m)
    s_plus = p0.margin_of(Lp, yp, ym)
    s_minus = p0.margin_of(Lm, yp, ym)
    ds = s_plus - s_minus
    if ds <= 0:
        return {"id": ex["id"], "skip": f"Δs={ds:.2f}<=0 (model không tách contrast)"}
    logp_clean = F.log_softmax(Lp, dim=-1)

    H_plus = p0.capture_states(model, layers, ids_p, q)
    H_minus = p0.capture_states(model, layers, ids_m, q)

    H_plac = None
    plac_warn = ""
    if ex.get("x_placebo"):
        ids_plac = p0.encode(tok, ex["x_placebo"])
        if ids_plac.shape[1] == ids_p.shape[1]:
            H_plac = p0.capture_states(model, layers, ids_plac, q)
        else:
            plac_warn = f"placebo len {ids_plac.shape[1]}!={ids_p.shape[1]} (bỏ)"

    if args.mode == "noise":
        base_ids, base_H, donor_H = ids_m, H_minus, H_plus
    else:
        base_ids, base_H, donor_H = ids_p, H_plus, H_minus

    def patched(ids, l, donor, base_at_q):
        return p0.logits_with_patch(model, layers, ids, q, l, donor,
                                    base_at_q=base_at_q, patch=args.patch, alpha=args.alpha)

    S, N, P, KL, KLp = [], [], [], [], []
    for l in range(Ln):
        lg = patched(base_ids, l, donor_H[l], base_H[l])
        S.append((p0.margin_of(lg, yp, ym) - s_minus)/ds if args.mode=="noise"
                 else (s_plus - p0.margin_of(lg, yp, ym))/ds)
        KL.append(p0.kl_to(lg, logp_clean))
        lg_n = p0.logits_with_patch(model, layers, ids_p, q, l, H_minus[l],
                                    base_at_q=H_plus[l], patch=args.patch, alpha=args.alpha)
        N.append((s_plus - p0.margin_of(lg_n, yp, ym))/ds)
        if H_plac is not None:
            lg_p = patched(base_ids, l, H_plac[l], base_H[l])
            P.append((p0.margin_of(lg_p, yp, ym) - s_minus)/ds if args.mode=="noise"
                     else (s_plus - p0.margin_of(lg_p, yp, ym))/ds)
            KLp.append(p0.kl_to(lg_p, logp_clean))

    Sadj = [(S[l]-P[l])/(1-P[l]+1e-6) for l in range(Ln)] if P else S
    Cadj = [min(sa, n) for sa, n in zip(Sadj, N)]

    lens = p0.lens_margins(model, layers, ids_p, q, yp, ym)
    l_read = p0.readout_onset_hold(lens, Ln)
    l_commit = p0.persistent_onset(Cadj, Ln, args.tau)
    G = (l_commit - l_read) if (l_commit is not None and l_read is not None) else None

    return {
        "id": ex["id"], "suite": ex.get("suite", "?"),
        "ds": ds, "read": l_read, "commit": l_commit, "G": G,
        "maxKLp": max(KLp) if KLp else None,
        "maxSadj": max(Sadj),
        "plac_warn": plac_warn, "skip": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--mode", choices=["noise","denoise"], default="denoise")
    ap.add_argument("--patch", choices=["replace","delta"], default="delta")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--tau", type=float, default=0.15)
    args = ap.parse_args()

    print(f"Loading {args.model} ... (mode={args.mode} patch={args.patch} "
          f"alpha={args.alpha} tau={args.tau})")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32).to(DEVICE).eval()
    layers = p0.get_layers(model)

    rows = []
    for line in open(args.pairs):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ex = json.loads(line)
        r = measure_pair(model, layers, tok, ex, args)
        rows.append(r)
        if r.get("skip"):
            print(f"  [skip] {r['id']:<14} {r['skip']}")

    print(f"\n{'id':<14}{'suite':<10}{'Δs':>8}{'read':>6}{'commit':>8}{'G':>5}"
          f"{'maxKLp':>9}{'maxSadj':>9}")
    print("-"*72)
    for r in rows:
        if r.get("skip"):
            continue
        g = "None" if r["G"] is None else str(r["G"])
        klp = "-" if r["maxKLp"] is None else f"{r['maxKLp']:.2f}"
        print(f"{r['id']:<14}{r['suite']:<10}{r['ds']:>8.2f}"
              f"{str(r['read']):>6}{str(r['commit']):>8}{g:>5}"
              f"{klp:>9}{r['maxSadj']:>9.2f}"
              + (f"  ⚠{r['plac_warn']}" if r['plac_warn'] else ""))

    # tóm tắt theo suite
    print()
    by = {}
    for r in rows:
        if r.get("skip") or r["G"] is None:
            continue
        by.setdefault(r["suite"], []).append(r["G"])
    for suite, gs in by.items():
        print(f"  {suite:<10} G: {gs}  (mean {sum(gs)/len(gs):+.1f})")
    print("\nĐọc: binding G có > factual G một cách hệ thống không?")
    print("Nếu maxKLp nhỏ (<1) -> on-manifold, G đáng tin. Nếu lớn -> tăng/giảm alpha.")
    print("Nếu commit=None -> Sadj không vượt tau; hạ tau hoặc tăng alpha.")


if __name__ == "__main__":
    main()