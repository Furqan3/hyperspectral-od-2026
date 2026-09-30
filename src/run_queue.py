"""Sequential experiment queue for the single GPU.

Re-reads configs/queue.yaml before every job, so experiments can be appended while it runs.
Each job: train (src/train.py) -> OOF inference on the fold's val split scored with the
competition metric -> runs/<name>/oof_metrics.json + one row in runs/experiments.csv.
Jobs with oof_metrics.json (done) or FAILED are skipped. `eval_only: true` scores an existing run.
A training job that runs out of GPU memory is retried once with half the batch.

  setsid nohup python src/run_queue.py > runs/logs/queue.log 2>&1 < /dev/null &
"""
import csv
import json
import os
import subprocess
import sys
import time

import yaml

from common import PREP, PROJECT, train_procs

RUNS = os.path.join(PROJECT, "runs")
QUEUE = os.path.join(PROJECT, "configs", "queue.yaml")
PY = sys.executable


def training_procs():
    return "".join(train_procs())


def run(cmd, log):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, cwd=PROJECT, stdout=f, stderr=subprocess.STDOUT).returncode


def arg(args, key, default):
    return args[args.index(key) + 1] if key in args else default


def evaluate_run(name, data, imgsz, log):
    w = f"{RUNS}/{name}/weights/best.pt"
    out = f"{RUNS}/{name}/oof_metrics.json"
    if not data.startswith("fold"):  # final model on all data: nothing held out
        json.dump({}, open(out, "w"))
        return {}
    fold = data.split("_")[0]  # fold0_street -> fold0: variants share the fold's val split
    rc = run([PY, "src/predict.py", w, "--split", "train", "--ids", f"{PREP}/{fold}_val.txt",
              "--imgsz", str(imgsz), "--eval", "--metrics-out", out], log)
    return json.load(open(out)) if rc == 0 and os.path.exists(out) else None


def record(name, data, args, m, hours):
    path = f"{RUNS}/experiments.csv"
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["name", "data", "args", "mAP50_95", "mAP50", "mAP75", "mAP_small", "mAP_medium", "hours", "finished"])
        w.writerow([name, data, " ".join(args)] + [round(m.get(k, float("nan")), 4) for k in
                   ("mAP50_95", "mAP50", "mAP75", "mAP_small", "mAP_medium")] + [round(hours, 2), time.strftime("%F %T")])


def main():
    import fcntl
    lock = open(f"{RUNS}/logs/run_queue.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # one runner only: two would train the same job twice
    except BlockingIOError:
        sys.exit("another run_queue.py is already running")
    jobs, bad_queue, idle = [], False, False
    while True:
        try:  # a typo while editing queue.yaml must not kill an overnight queue: keep the last good version
            jobs = yaml.safe_load(open(QUEUE)) or []
            bad_queue = False
        except (yaml.YAMLError, OSError) as e:
            if not bad_queue:
                print(time.strftime("%F %T"), "queue.yaml unreadable, keeping previous queue:", str(e).splitlines()[0], flush=True)
            bad_queue = True
            if not jobs:
                time.sleep(60)
                continue
        todo = [j for j in jobs if not os.path.exists(f"{RUNS}/{j['name']}/oof_metrics.json")
                and not os.path.exists(f"{RUNS}/{j['name']}/FAILED")]
        if not todo:
            # Keep waiting instead of exiting: jobs appended to queue.yaml later would otherwise sit
            # unstarted until someone relaunches the runner (that stranded f0_rtdetr_l_photo once).
            if not idle:
                print(time.strftime("%F %T"), "queue empty, waiting for new jobs", flush=True)
                idle = True
            time.sleep(60)
            continue
        idle = False
        procs = training_procs()
        # eval-only jobs may run beside a training as long as their own run has finished
        ready = [j for j in todo if j.get("eval_only") and os.path.exists(f"{RUNS}/{j['name']}/weights/best.pt")
                 and f"--name {j['name']} " not in procs]
        if not ready and procs.strip():
            time.sleep(60)
            continue
        j = ready[0] if ready else next((j for j in todo if not j.get("eval_only")), None)
        if j is None:
            time.sleep(60)
            continue

        name, data = j["name"], j.get("data", "fold0")
        args = [str(x) for x in j.get("args", [])]
        log = f"{RUNS}/logs/{name}.log"
        t0 = time.time()
        print(time.strftime("%F %T"), "start", name, "(eval only)" if j.get("eval_only") else args, flush=True)
        if not j.get("eval_only"):
            rc = run([PY, "src/train.py", "--data", data, "--name", name] + args, log)
            if rc != 0 and "out of memory" in open(log, errors="ignore").read()[-20000:].lower():
                b = int(arg(args, "--batch", 8))
                args = [a for i, a in enumerate(args) if a != "--batch" and (i == 0 or args[i - 1] != "--batch")]
                args += ["--batch", str(max(2, b // 2))]
                print(time.strftime("%F %T"), f"OOM -> retry {name} with batch {max(2, b // 2)}", flush=True)
                rc = run([PY, "src/train.py", "--data", data, "--name", name] + args, log)
            if rc != 0 or not os.path.exists(f"{RUNS}/{name}/weights/best.pt"):
                os.makedirs(f"{RUNS}/{name}", exist_ok=True)
                open(f"{RUNS}/{name}/FAILED", "w").write(f"exit {rc}\n")
                print(time.strftime("%F %T"), "FAILED", name, rc, flush=True)
                continue
        imgsz = arg(args, "--imgsz", yaml.safe_load(open(f"{RUNS}/{name}/args.yaml"))["imgsz"])
        m = evaluate_run(name, data, imgsz, log)
        if m is None:
            open(f"{RUNS}/{name}/FAILED", "w").write("evaluation failed\n")
            print(time.strftime("%F %T"), "EVAL FAILED", name, flush=True)
            continue
        record(name, data, args, m, (time.time() - t0) / 3600)
        print(time.strftime("%F %T"), "done", name, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}, flush=True)


if __name__ == "__main__":
    main()
