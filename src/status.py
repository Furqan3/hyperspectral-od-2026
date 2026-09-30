"""Training dashboard: one line per run + GPU + live progress.

  python src/status.py          # snapshot
  watch -n 30 python src/status.py
"""
import glob
import os
import re
import subprocess

import pandas as pd
import yaml

from common import PROJECT, train_procs

RUNS = os.path.join(PROJECT, "runs")


def main():
    running = "".join(train_procs())
    rows = []
    for f in sorted(glob.glob(f"{RUNS}/*/results.csv"), key=os.path.getmtime):
        name = os.path.basename(os.path.dirname(f))
        if name == "smoke":
            continue
        d = pd.read_csv(f)
        d.columns = d.columns.str.strip()
        args = yaml.safe_load(open(os.path.join(os.path.dirname(f), "args.yaml")))
        total, done = args["epochs"], int(d.epoch.iloc[-1])
        best = d["metrics/mAP50-95(B)"].idxmax()
        sec = d.time.diff().iloc[-5:].mean() if len(d) > 1 else d.time.iloc[-1]
        live = f"--name {name} " in running
        rows.append({
            "run": name, "epoch": f"{done}/{total}",
            "mAP50-95": round(d["metrics/mAP50-95(B)"].iloc[-1], 4), "mAP50": round(d["metrics/mAP50(B)"].iloc[-1], 4),
            "best50-95 (ep)": f"{d['metrics/mAP50-95(B)'][best]:.4f} ({int(d.epoch[best])})",
            "s/epoch": round(sec), "ETA": f"{(total - done) * sec / 3600:.1f} h" if live else ("done" if done == total else "stopped"),
            "state": "RUNNING" if live else "-",
        })
    print(pd.DataFrame(rows).to_string(index=False) if rows else "no runs yet")

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"\nGPU: {gpu}")
    for log in sorted(glob.glob(f"{RUNS}/logs/*.log"), key=os.path.getmtime)[-1:]:
        tail = open(log, "rb").read()[-2000:].decode(errors="ignore").replace("\r", "\n")
        last = [re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", l).strip() for l in tail.splitlines() if l.strip()]
        print(f"{os.path.basename(log)}: {last[-1][:150] if last else ''}")
    st = os.path.join(RUNS, "logs", "baselines.status")
    if os.path.exists(st):
        print("finished:", open(st).read().replace("\n", "  "))


if __name__ == "__main__":
    main()
