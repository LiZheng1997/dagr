"""Pool lead_time_dsec.py outputs over several sequences and print Table-IV style numbers.

Usage: python3 lead_time_aggregate.py --labels <dir with **/<seq>/**/tracks.npy> <prefix> [<prefix> ...]

Per score threshold: pooled precision (detections and false positives summed over sequences),
per-class track counts after excluding tracks already labelled at the first evaluated pair of
their sequence, first-detection delay after the first label for each model, lead (frame minus
event) on tracks found by both, and recall of each model by labelled box height (full-res px).
"""
import argparse
import glob
import json
import os

import numpy as np


def q(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def track_heights(labels_root):
    """(seq, track_id) -> max labelled box height in full-resolution pixels."""
    H = {}
    for f in glob.glob(os.path.join(labels_root, "**", "tracks.npy"), recursive=True):
        seq = next(p for p in f.split(os.sep)[::-1] if p.startswith("zurich_") or p.startswith("interlaken_") or p.startswith("thun_"))
        t = np.load(f)
        for tid in np.unique(t["track_id"]):
            H[(seq, int(tid))] = float(t["h"][t["track_id"] == tid].max())
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--buckets", default="20,40", help="height bucket edges, full-res px")
    a = ap.parse_args()
    edges = [float(x) for x in a.buckets.split(",")]
    H = track_heights(a.labels)

    runs = []
    for p in a.prefixes:
        S = json.load(open(p + "_summary.json"))
        evals = [json.loads(l) for l in open(p + "_evals.jsonl")]
        first_t0 = {}
        for e in evals:
            first_t0[e["seq"]] = min(first_t0.get(e["seq"], e["t0"]), e["t0"])
        runs.append((p, S, first_t0))
    seqs = sorted({s for _, S, _ in runs for s in S["sequences"]})
    pairs = sum(S["pairs"] for _, S, _ in runs)
    print(f"sequences {seqs}\npairs {pairs}  steps_us {runs[0][1]['steps_us']}  iou {runs[0][1]['iou']}")
    ms_f = np.mean([S["ms_mean"]["frame"] for _, S, _ in runs]); ms_e = np.mean([S["ms_mean"]["event"] for _, S, _ in runs])
    print(f"forward ms mean (unweighted over runs): frame-only {ms_f:.0f}  event+frame {ms_e:.0f}")

    scores = list(runs[0][1]["by_score"].keys())
    for thr in scores:
        dets = {"frame": 0, "event": 0}; fps = {"frame": 0, "event": 0}
        rows = []
        n_excl = 0
        for p, S, first_t0 in runs:
            R = S["by_score"][thr]
            for m in dets:
                dets[m] += R["detections"][m]; fps[m] += R["false_positives"][m]
            for r in R["tracks"]:
                if r["t_label"] > first_t0[r["seq"]]:
                    rows.append(r)
                else:
                    n_excl += 1
        prec = {m: (dets[m] - fps[m]) / dets[m] if dets[m] else float("nan") for m in dets}
        print(f"\n=== score >= {thr}   precision frame-only {prec['frame']:.3f}  event+frame {prec['event']:.3f}   "
              f"(dets {dets}, fp {fps})   excluded {n_excl} tracks visible at sequence start")
        for cls in ("pedestrian", "car"):
            rr = [r for r in rows if r["cls"] == cls]
            fd = np.array([r["frame_delay_ms"] for r in rr if r["frame_delay_ms"] is not None])
            ed = np.array([r["event_delay_ms"] for r in rr if r["event_delay_ms"] is not None])
            lead = np.array([r["lead_ms"] for r in rr if r["lead_ms"] is not None])
            only_e = sum(1 for r in rr if r["event_delay_ms"] is not None and r["frame_delay_ms"] is None)
            only_f = sum(1 for r in rr if r["frame_delay_ms"] is not None and r["event_delay_ms"] is None)
            neither = sum(1 for r in rr if r["frame_delay_ms"] is None and r["event_delay_ms"] is None)
            print(f"-- {cls}: tracks {len(rr)}  found: frame-only {len(fd)}  event+frame {len(ed)}  "
                  f"event-only {only_e}  frame-only {only_f}  neither {neither}")
            if len(fd):
                print(f"   first det. after first label, frame-only : median {np.median(fd):6.0f}  mean {fd.mean():6.0f}  p25 {q(fd,25):6.0f}  p75 {q(fd,75):6.0f}")
            if len(ed):
                print(f"   first det. after first label, event+frame: median {np.median(ed):6.0f}  mean {ed.mean():6.0f}  p25 {q(ed,25):6.0f}  p75 {q(ed,75):6.0f}")
            if len(lead):
                pos = int((lead > 0).sum()); zero = int((lead == 0).sum()); neg = int((lead < 0).sum())
                print(f"   lead on {len(lead)} tracks found by both: median {np.median(lead):5.0f}  mean {lead.mean():5.0f}  "
                      f"p25 {q(lead,25):5.0f}  p75 {q(lead,75):5.0f}  earlier/same/later {pos}/{zero}/{neg}  "
                      f">=40ms {int((lead>=40).sum())}  >=100ms {int((lead>=100).sum())}")
                hist = {}
                for v in lead:
                    hist[int(round(v / 10) * 10)] = hist.get(int(round(v / 10) * 10), 0) + 1
                print("   lead histogram (ms:n):", " ".join(f"{k}:{v}" for k, v in sorted(hist.items())))
            # recall by labelled height
            print("   recall by max box height (full-res px):")
            lo = 0.0
            for hi in edges + [1e9]:
                b = [r for r in rr if lo <= H.get((r["seq"], r["track"]), -1) < hi]
                if b:
                    rf = sum(1 for r in b if r["frame_delay_ms"] is not None) / len(b)
                    re = sum(1 for r in b if r["event_delay_ms"] is not None) / len(b)
                    lab = f"[{lo:.0f},{hi:.0f})" if hi < 1e9 else f">={lo:.0f}"
                    print(f"     {lab:>10}: n {len(b):3d}  frame-only {rf:.2f}  event+frame {re:.2f}")
                lo = hi
            miss = sum(1 for r in rr if (r["seq"], r["track"]) not in H)
            if miss:
                print(f"     ({miss} tracks without label height)")


if __name__ == "__main__":
    main()
