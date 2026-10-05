"""Read-only: norms and bit-identity of every expert across the task checkpoints of a DMPEL run.

Usage: python scripts/expert_norms.py <run_seed_dir> [--out analysis/exp4/expert_norms_fixed.txt] [--ref <other_run_seed_dir>]

For each checkpoint k and expert e <= k: norm_A, norm_B, norm_bias (L2 over all A-type / B-type / bias pool tensors of that expert),
and whether the expert's weights are bit-identical to its creation-time weights (the row in task{e}_model.pth).
--ref prints the largest relative change of frozen experts between consecutive checkpoints for a second run (e.g. the decayed one).
"""
import argparse, glob, os, re
import torch

POOL = ("A_pool", "A_q_pool", "A_v_pool", "B_pool", "B_q_pool", "B_v_pool", "bias_pool")
KINDS = {"A": POOL[:3], "B": POOL[3:6], "bias": POOL[6:]}


def load(d):
    ks = sorted(int(re.search(r"task(\d+)_model", f).group(1)) for f in glob.glob(f"{d}/task*_model.pth"))
    return {k: torch.load(f"{d}/task{k}_model.pth", map_location="cpu", weights_only=False)["state_dict"] for k in ks}


def pools(sd):
    return [n for n in sd if n.split(".")[-1] in POOL]


def norm(sd, e, kind):
    return float(sum((v[e].float() ** 2).sum() for n, v in sd.items() if n.split(".")[-1] in KINDS[kind]) ** 0.5)


def max_rel_change(sds, k):
    """largest relative change of any frozen expert (0..k-1) between checkpoint k-1 and k."""
    worst = 0.0
    for e in range(k):
        num = den = 0.0
        for n in pools(sds[k - 1]):
            a, b = sds[k - 1][n][e].float(), sds[k][n][e].float()
            num += float(((b - a) ** 2).sum()); den += float((a ** 2).sum())
        worst = max(worst, (num / max(den, 1e-30)) ** 0.5)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    sds = load(a.run)
    ks = sorted(sds)
    lines = [f"run: {a.run}", "norm_A / norm_B / norm_bias of each expert after each task; 'same' = bit-identical to its creation-time weights", "",
             f"{'ckpt':>4} {'expert':>6} {'norm_A':>8} {'norm_B':>8} {'norm_bias':>9} {'same as creation':>17}"]
    all_same = True
    for k in ks:
        for e in range(k + 1):
            same = all(torch.equal(sds[k][n][e], sds[e][n][e]) for n in pools(sds[k]))
            all_same &= same
            lines.append(f"{k:>4} {e:>6} {norm(sds[k], e, 'A'):>8.3f} {norm(sds[k], e, 'B'):>8.3f} {norm(sds[k], e, 'bias'):>9.4f} {str(same):>17}")
    lines += ["", "largest relative change of a frozen expert between consecutive checkpoints:"]
    for k in ks[1:]:
        lines.append(f"  task {k - 1} -> {k}: {max_rel_change(sds, k):.6f}")
    lines.append(f"all frozen experts bit-identical to creation-time weights in every checkpoint: {all_same}")
    if a.ref:
        ref = load(a.ref)
        lines += ["", f"reference run: {a.ref}"] + [f"  task {k - 1} -> {k}: {max_rel_change(ref, k):.6f}" for k in sorted(ref)[1:]]
    text = "\n".join(lines)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
