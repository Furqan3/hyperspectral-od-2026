"""One-look progress: what is training, how far, how fast, what finished, submission quota.

  python src/progress.py            # snapshot
  watch -n 60 python src/progress.py
"""
import glob, json, os, re, subprocess, sys, time
import pandas as pd

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# running config -> the comparable earlier run (same model, the recipe it must beat)
REF = {
    's_ms1536_all':      ('deim_f0_dfine_s_ms1536', 'S ms1536 fold0'),
    's_ms1536_rxmild':   ('deim_f0_dfine_s_ms1536', 'S ms1536 plain'),
    's_ms1536_rxstrong': ('deim_f0_dfine_s_ms1536', 'S ms1536 plain'),
    'l_ms1536':          ('deim_f0_dfine_l',        'L @1024'),
    'l_ms1536_rxmild':   ('deim_f0_dfine_l_ms1536', 'L ms1536 plain'),
    'l_ms1536_rxstrong': ('deim_f0_dfine_l_ms1536', 'L ms1536 plain'),
    'deim_all_ms1536_e40':   ('deim_all_ms1536', 'all-data 72ep'),
    'deim_all_ms1536_rxaug': ('deim_all_ms1536_e40', 'all-data 40ep'),
}


def curve(log):
    if not log or not os.path.exists(log):
        return {}
    t = open(log, errors='ignore').read()
    return {int(e): float(a) for e, a in re.findall(r"'epoch': (\d+), 'coco_eval_bbox': ([0-9.]+)", t)}


def procs():
    """running trainings: (config name, pid, elapsed)"""
    out = []
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            argv = [a for a in open(f'/proc/{pid}/cmdline', 'rb').read().decode(errors='ignore').split('\0') if a]
        except OSError:
            continue
        try:  # skip dataloader workers: they share the parent's cmdline
            ppid = open(f'/proc/{pid}/stat').read().split(') ')[1].split()[1]
            pargv = [a for a in open(f'/proc/{ppid}/cmdline', 'rb').read().decode(errors='ignore').split('\0') if a]
        except OSError:
            pargv = []
        if pargv[:3] == argv[:3]:
            continue
        if len(argv) > 2 and argv[1].endswith('train.py') and 'configs/hsi' in ' '.join(argv):
            cfg = next((os.path.basename(a) for a in argv if a.endswith('.yml')), '?')
            el = subprocess.run(['ps', '-o', 'etime=', '-p', pid], capture_output=True, text=True).stdout.strip()
            out.append((cfg, pid, el))
        elif len(argv) > 2 and 'tools/train.py' in ' '.join(argv):
            out.append((os.path.basename(argv[2]), pid, ''))
    return sorted(set(out))


def log_state(log):
    t = open(log, errors='ignore').read() if os.path.exists(log) else ''
    ep = re.findall(r'Epoch: \[(\d+)\]', t)
    tot = re.findall(r'Epoch: \[\d+\] Total time: (\d+):(\d+):(\d+)', t)
    ap = re.findall(r"'epoch': (\d+), 'coco_eval_bbox': ([0-9.]+)", t)
    secs = [int(h)*3600+int(m)*60+int(s) for h, m, s in tot[-5:]]
    return (int(ep[-1]) if ep else None,
            sum(secs)/len(secs) if secs else None,
            (int(ap[-1][0]), float(ap[-1][1])) if ap else None,
            max((float(a) for _, a in ap), default=None))


def main():
    print(f"=== {time.strftime('%F %H:%M')} ===\nRUNNING")
    live = procs()
    if not live:
        print("  (nothing training)")
    for cfg, pid, el in live:
        name = cfg.replace('hsi_deim_dfine_', '').replace('.yml', '')
        cfg_path = f'{P}/third_party/DEIM/configs/hsi/{cfg}'
        run = None
        if os.path.exists(cfg_path):
            m = re.search(r'^output_dir:\s*(\S+)', open(cfg_path).read(), re.M)
            run = os.path.basename(m.group(1)) if m else None
        log = f'{P}/runs/logs/{run}.log' if run and os.path.exists(f'{P}/runs/logs/{run}.log') else None
        if log is None:
            log = next((l for l in glob.glob(f'{P}/runs/logs/*.log')
                        if re.search(re.escape(cfg), open(l, errors='ignore').read()[:8000])), None)
        name = run or name
        ep, sec, last, best = log_state(log) if log else (None, None, None, None)
        total = 40 if 'e40' in cfg or 'rxaug' in cfg else (58 if '_l' in cfg and 'ms1536' in cfg else 72)
        eta = f"{(total-ep)*sec/3600:4.1f} h" if ep and sec else "  ? "
        print(f"  {name:28s} epoch {str(ep or '?'):>3}/{total}  {sec/60 if sec else 0:4.1f} min/ep  ETA {eta}"
              f"  last AP {last[1] if last else float('nan'):.4f}  (up {el})")
        ref_run, ref_label = REF.get(name, (None, None))
        if ref_run and last:
            rc = curve(f'{P}/runs/logs/{ref_run}.log')
            if rc:
                near = min(rc, key=lambda e: abs(e - last[0]))
                d = last[1] - rc[near]
                print(f"  {'':28s} vs {ref_label:16s} @ep{near:<3d} {rc[near]:.4f}  ->  {d:+.4f}   "
                      f"(its final best {max(rc.values()):.4f})")
    try:
        g = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                            '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
        print(f"  GPU: {g}")
    except Exception:
        pass

    print("\nFINISHED (fold-0 held-out score, best first)")
    rows = []
    for f in glob.glob(f'{P}/runs/*/oof_metrics.json'):
        m = json.load(open(f))
        if m.get('mAP50_95'):
            rows.append((os.path.basename(os.path.dirname(f)), m['mAP50_95'], m['mAP50']))
    for n, a, b in sorted(rows, key=lambda r: -r[1])[:8]:
        print(f"  {n:30s} {a:.4f}   mAP50 {b:.4f}")

    q = f'{P}/runs/logs/rx_queue.log'
    print("\nQUEUE")
    pend = [l.split('"')[1].split()[0] for l in open(f'{P}/runs/logs/rx_queue.sh') if l.strip().startswith('"deim')]
    done = {os.path.basename(os.path.dirname(f)) for f in glob.glob(f'{P}/runs/*/oof_metrics.json')}
    for n in pend:
        print(f"  {'done' if n in done else 'waiting'}: {n}")
    if os.path.exists(q):
        print("  last:", open(q).read().strip().splitlines()[-1] if open(q).read().strip() else "(idle)")

    try:
        tok = open(os.path.expanduser('~/.kaggle/api_token')).read().strip()
        import urllib.request, datetime
        url = 'https://www.kaggle.com/api/v1/competitions/submissions/list/hyperspectral-object-detection-challenge-2026'
        s = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={'Authorization': f'Bearer {tok}'}), timeout=30))
        today = datetime.datetime.utcnow().strftime('%Y-%m-%d')
        scored = [(x['date'][:16], float(x['publicScore']), x['fileName']) for x in s if x.get('publicScore')]
        print(f"\nKAGGLE  used {len([x for x in s if x['date'].startswith(today)])}/3 today (resets 05:00 local)")
        for d, sc, f in sorted(scored, key=lambda r: -r[1])[:3]:
            print(f"  best: {sc:.5f}  {f}  ({d})")
    except Exception as e:
        print("\nKAGGLE  (offline:", str(e)[:40] + ")")


if __name__ == '__main__':
    main()
