#!/usr/bin/env python3
"""
Evaluate a multi-node DDP run from SLURM logs produced by launch_tiny.sh.

Parses:
  - Rendezvous banner
  - [RANK r] WORLD_SIZE=w
  - Sanity banners with versions
  - Failure signatures
  - Scientific signals:
      * step_ms / samples_per_sec / tokens_per_sec
      * TRAIN_RUNTIME_SEC
      * EVAL accuracy
      * INFER global_* line (from infer_ddp.py)

Usage:
  python labs/tiny/eval_logs.py --job 123456
"""

import argparse, glob, os, re, sys
from collections import Counter

RANK_RE   = re.compile(r"\[RANK\s+(\d+)\]\s+WORLD_SIZE=(\d+)")
RDZV_RE   = re.compile(r"torchrun:\s+nnodes=(\d+)\s+nproc_per_node=(\d+)\s+node_rank=(\d+)\s+rdzv=([^\s]+)")
SANITY_RE = re.compile(r"NODE\s+(\S+)\s+OK\s+->\s+PY\s+(\S+)\s+torch\s+(\S+)\s+tfm\s+(\S+)\s+numpy\s+(\S+)\s+datasets\s+(\S+)\s+root\s+(\S+)")
FAIL_PAT  = re.compile(r"(Traceback|ERROR|RuntimeError|OSError|Address already in use|Timed out)", re.IGNORECASE)

STEP_RE   = re.compile(r"\[rank\s+(\d+)\s+\|\s+step\s+(\d+)\]\s+.*?step_ms=([\d.]+)\s+samples_per_sec=([\d.]+)\s+tokens_per_sec=([\d.]+)")
RT_RE     = re.compile(r"TRAIN_RUNTIME_SEC=([\d.]+)")
EVAL_RE   = re.compile(r"(?:eval_accuracy|EVAL accuracy)=\s*([0-9]*\.?[0-9]+)")
INFER_RE  = re.compile(r"\[RANK 0\]\s+INFER.*global_accuracy=\s*([0-9]*\.?[0-9]+).*?global_samples_per_sec=([\d.]+).*?global_tokens_per_sec=([\d.]+)")

def read_text(p):
    try:
        with open(p, "r", errors="replace") as f: return f.read()
    except Exception as e:
        return f"[[could not read {p}: {e}]]"

def parse_preflight(path):
    if not os.path.exists(path): return {}
    status = {}
    for line in read_text(path).splitlines():
        parts = line.strip().split()
        if not parts: continue
        host = parts[0]; proj="missing"; pkgs="missing"
        for p in parts[1:]:
            if p.startswith("proj="): proj = p.split("=",1)[1]
            if p.startswith("pkgs="): pkgs = p.split("=",1)[1]
        status[host]=(proj,pkgs)
    return status

def pct(vals, q):
    if not vals: return None
    s=sorted(float(x) for x in vals)
    k=(len(s)-1)*q
    f=int(k); c=min(f+1,len(s)-1)
    if f==c: return s[f]
    return s[f] + (s[c]-s[f])*(k-f)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--logs-dir", default=os.path.expanduser("~/slurm_logs"))
    args=ap.parse_args()

    preflight=os.path.join(args.logs_dir, f"preflight.{args.job}.txt")
    outs=sorted(glob.glob(os.path.join(args.logs_dir, f"*.{args.job}.*.out")))
    if not outs:
        print(f"[!] No log files found matching {args.logs_dir}/*.{args.job}.*.out", file=sys.stderr); sys.exit(2)

    ranks_seen=set(); world_sizes=Counter(); ws_candidates=set()
    rdzv_records=[]; sanity={}; failures=[]; file_fail_counts=Counter()
    step_ms=[]; sps=[]; tps=[]; runtimes=[]; eval_acc=[]; infer_global=[]

    for path in outs:
        txt=read_text(path)

        for m in RANK_RE.finditer(txt):
            r=int(m.group(1)); ws=int(m.group(2))
            ranks_seen.add(r); world_sizes[ws]+=1; ws_candidates.add(ws)

        for m in RDZV_RE.finditer(txt):
            rdzv_records.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)))

        for m in SANITY_RE.finditer(txt):
            host, py, tv, tfmv, npv, dsv, root=m.groups()
            sanity[host]=(py,tv,tfmv,npv,dsv,root)

        for m in STEP_RE.finditer(txt):
            step_ms.append(float(m.group(3)))
            sps.append(float(m.group(4)))
            tps.append(float(m.group(5)))

        runtimes += [float(x) for x in RT_RE.findall(txt)]
        eval_acc += [float(x) for x in EVAL_RE.findall(txt)]
        infer_global += [tuple(map(float,m.groups())) for m in INFER_RE.finditer(txt)]

        m_iter=list(FAIL_PAT.finditer(txt))
        if m_iter:
            file_fail_counts[os.path.basename(path)] = len(m_iter)
            failures.extend([f"{os.path.basename(path)}: {m.group(0)}" for m in m_iter])

    pf = parse_preflight(preflight)

    print("="*72); print(f" EVALUATION REPORT FOR JOB {args.job}"); print("="*72)

    # RDZV
    print("\n[RDZV] Rendezvous records (unique):")
    uniq = sorted(set((nn,npn,ep) for nn,npn,_nr,ep in rdzv_records))
    for nn,npn,ep in uniq: print(f"  - nnodes={nn}  nproc_per_node={npn}  endpoint={ep}")
    if not uniq: print("  (!) No rendezvous banner found.")
    elif len(uniq)>1: print("  (!) Inconsistent rendezvous configs detected.")

    # RANKS
    if ranks_seen:
        print("\n[RANKS] Rank coverage:")
        print(f"  - ranks seen: min={min(ranks_seen)}  max={max(ranks_seen)}  count={len(ranks_seen)}")
        print(f"  - WORLD_SIZE candidates (value -> observations): {sorted(world_sizes.items())}")
        if len(ws_candidates)==1:
            ws=next(iter(ws_candidates))
            miss=sorted(set(range(ws))-ranks_seen)
            if miss: print(f"  (!) Missing ranks: {miss[:15]}{' ...' if len(miss)>15 else ''}")
            elif max(ranks_seen)!=ws-1: print(f"  (!) Max rank {max(ranks_seen)} != WORLD_SIZE-1 ({ws-1})")
            else: print("  ✓ All ranks accounted for (max == WORLD_SIZE-1).")
        else:
            print("  (!) Multiple WORLD_SIZE values observed; check consistency.")
    else:
        print("\n[RANKS] (!) No '[RANK r] WORLD_SIZE=w' lines found.")

    # SANITY
    if sanity:
        print("\n[SANITY] Per-node versions:")
        hdr=f"{'HOST':<16} {'PY':<6} {'torch':<8} {'tfm':<8} {'numpy':<8} {'datasets':<10} root"
        print("  "+hdr); print("  "+"-"*len(hdr))
        for h in sorted(sanity):
            py,tv,tfmv,npv,dsv,root=sanity[h]
            print(f"  {h:<16} {py:<6} {tv:<8} {tfmv:<8} {npv:<8} {dsv:<10} {root}")
        for name,idx in [("PY",0),("torch",1),("tfm",2),("numpy",3),("datasets",4)]:
            vals=Counter(v[idx] for v in sanity.values())
            if len(vals)>1: print(f"  (!) Version drift in {name}: {dict(vals)}")
    else:
        print("\n[SANITY] (!) No sanity banners found.")

    # PREFLIGHT
    if pf:
        ok=sum(1 for v in pf.values() if v==("ok","ok"))
        print("\n[PREFLIGHT] Project/pkgs visibility by node:")
        for host in sorted(pf):
            proj,pkgs=pf[host]; tag="OK" if (proj,pkgs)==("ok","ok") else "STAGED/MISSING"
            print(f"  {host:<16} proj={proj:<7} pkgs={pkgs:<7} -> {tag}")
        print(f"  Summary: {ok}/{len(pf)} nodes had both proj & pkgs visible.")
    else:
        print("\n[PREFLIGHT] (!) No preflight file found.")

    # THROUGHPUT
    if step_ms:
        print("\n[THROUGHPUT] Step-time & throughput (all ranks, across steps):")
        p50_ms, p95_ms = pct(step_ms,0.50), pct(step_ms,0.95)
        p50_sps, p95_sps = pct(sps,0.50), pct(sps,0.95)
        p50_tps, p95_tps = pct(tps,0.50), pct(tps,0.95)
        print(f"  - step_ms:     P50={p50_ms:.2f}  P95={p95_ms:.2f}")
        print(f"  - samples/sec: P50={p50_sps:.1f} P95={p95_sps:.1f}")
        print(f"  - tokens/sec:  P50={p50_tps:.1f} P95={p95_tps:.1f}")
    if runtimes:
        print(f"  - TRAIN_RUNTIME_SEC (rank0): min={min(runtimes):.2f} max={max(runtimes):.2f}")

    # EVAL
    if eval_acc:
        print(f"\n[EVAL] eval_accuracy (epoch logs): best={max(eval_acc):.4f} last={eval_acc[-1]:.4f}")
    if infer_global:
        acc, sps_g, tps_g = infer_global[-1]
        print(f"[INFER] global_accuracy={acc:.4f} global_samples_per_sec={sps_g:.1f} global_tokens_per_sec={tps_g:.1f}")

    # FAILURES
    if failures:
        print("\n[FAILURES] Signatures found:")
        for f,c in file_fail_counts.most_common(5): print(f"  - {f}: {c} hits")
        for line in failures[:8]: print(f"    e.g., {line}")
        if len(failures)>8: print(f"    ... (+{len(failures)-8} more)")
    else:
        print("\n[FAILURES] ✓ No failure signatures detected.")

    # VERDICT
    print("\n[VERDICT]")
    verdict_ok = bool(uniq) and bool(ranks_seen) and len(ws_candidates)==1
    if verdict_ok:
        ws=next(iter(ws_candidates))
        verdict_ok = (set(range(ws))-ranks_seen)==set() and not failures
    print("  ✓ PASS" if verdict_ok else "  ✗ CHECK LOGS (see sections above)")

if __name__ == "__main__":
    main()
