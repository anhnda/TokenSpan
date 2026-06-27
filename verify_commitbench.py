"""
verify_commitbench.py — lọc seed CommitBench bằng tokenizer THẬT.

Kiểm 2 ràng buộc cứng mà mắt thường không chắc được:
  (1) y_plus, y_minus mỗi cái == ĐÚNG 1 token  -> margin z_y+ - z_y- mới định nghĩa được
  (2) prefix tokenization của x_plus và x_minus có CÙNG độ dài tới q
      -> "patch tại q" mới so cùng vị trí

In ra PASS/FAIL từng cặp + lý do. Không sửa data, chỉ báo cáo.
Chạy:  python verify_commitbench.py commitbench_seed.jsonl --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse, json, sys
from transformers import AutoTokenizer


def is_single_token(tok, s):
    # answer đứng sau khoảng trắng trong câu -> giữ nguyên leading space của s
    ids = tok(s, add_special_tokens=False)["input_ids"]
    return len(ids) == 1, ids


def prefix_len(tok, s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    n_pass = n_fail = 0

    for line in open(args.path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ex = json.loads(line)
        reasons = []

        ok_p, ids_p = is_single_token(tok, ex["y_plus"])
        ok_m, ids_m = is_single_token(tok, ex["y_minus"])
        if not ok_p:
            reasons.append(f"y_plus {ex['y_plus']!r} -> {len(ids_p)} tokens")
        if not ok_m:
            reasons.append(f"y_minus {ex['y_minus']!r} -> {len(ids_m)} tokens")

        lp = prefix_len(tok, ex["x_plus"])
        lm = prefix_len(tok, ex["x_minus"])
        if lp != lm:
            reasons.append(f"prefix len mismatch: x_plus={lp} vs x_minus={lm} (q misaligned)")

        status = "PASS" if not reasons else "FAIL"
        n_pass += status == "PASS"
        n_fail += status == "FAIL"
        print(f"[{status}] {ex['id']:<10} {ex['suite']:<11} "
              + ("" if not reasons else "| " + "; ".join(reasons)))

    print(f"\n{n_pass} pass, {n_fail} fail. "
          f"Chỉ dùng PASS cho P0/benchmark. FAIL: sửa answer hoặc bỏ cặp.")


if __name__ == "__main__":
    main()
