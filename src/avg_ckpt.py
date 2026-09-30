"""Average the EMA weights of several checkpoints from ONE run (single model, rule-compliant).

Late checkpoints of one run sit in the same loss basin, so averaging them cuts the seed/step noise
that has been drowning our experiments (two identical runs scored 0.65797 and 0.64523). Costs no
training time: the checkpoints already exist.

  python src/avg_ckpt.py out.pth runs/<run>/checkpoint0031.pth runs/<run>/checkpoint0035.pth runs/<run>/last.pth
"""
import sys
import torch


def main(out, paths):
    acc, n, ref = None, 0, None
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["ema"]["module"] if "ema" in ck else ck["model"]
        if acc is None:
            acc = {k: v.clone().float() for k, v in sd.items()}
            ref = ck
        else:
            for k in acc:
                acc[k] += sd[k].float()
        n += 1
        print(f"  + {p.split('/')[-1]} (epoch {ck.get('last_epoch')})")
    for k in acc:
        acc[k] /= n
    src = ref["ema"]["module"] if "ema" in ref else ref["model"]
    for k, v in src.items():          # keep original dtypes (ints/buffers untouched by averaging)
        acc[k] = acc[k].to(v.dtype)
    if "ema" in ref:
        ref["ema"]["module"] = acc
    else:
        ref["model"] = acc
    torch.save(ref, out)
    print(f"averaged {n} checkpoints -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
