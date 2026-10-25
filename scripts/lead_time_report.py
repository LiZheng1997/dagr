"""Report first-detection timing from lead_time_dsec.py outputs.

Reads <prefix>_summary.json and <prefix>_evals.jsonl. Tracks already labelled at the first
evaluated pair cannot be compared (the frame-only model is evaluated at t0, the event model
from t0 + 10 ms on), so they are excluded; so are tracks neither model ever matched.

Usage: python3 lead_time_report.py <prefix> [--min-track-ms 200]
"""
import argparse
import json

import numpy as np


def q(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--min-track-ms", type=float, default=0.0, help="ignore tracks labelled for less than this")
    a = ap.parse_args()
    S = json.load(open(a.prefix + "_summary.json"))
    evals = [json.loads(l) for l in open(a.prefix + "_evals.jsonl")]
    first_t0 = min(e["t0"] for e in evals)
    # track duration from the evaluation log is not stored; approximate with label span if needed later
    print(f"sequences {S['sequences']} pairs {S['pairs']} steps_us {S['steps_us']} iou {S['iou']}")
    print(f"forward ms mean: frame-only {S['ms_mean']['frame']:.0f}  event+frame {S['ms_mean']['event']:.0f}")
    print(f"tracks total: {S['tracks_total']}")
    for thr, R in S["by_score"].items():
        print(f"\n=== score >= {thr}   precision frame-only {R['precision']['frame']:.3f}  event+frame {R['precision']['event']:.3f} "
              f"(dets {R['detections']}, fp {R['false_positives']})")
        rows = [r for r in R["tracks"] if r["t_label"] > first_t0]
        n_excl = len(R["tracks"]) - len(rows)
        print(f"excluded {n_excl} tracks visible at sequence start")
        for cls in ("pedestrian", "car"):
            rr = [r for r in rows if r["cls"] == cls]
            fd = np.array([r["frame_delay_ms"] for r in rr if r["frame_delay_ms"] is not None])
            ed = np.array([r["event_delay_ms"] for r in rr if r["event_delay_ms"] is not None])
            both = [r for r in rr if r["lead_ms"] is not None]
            lead = np.array([r["lead_ms"] for r in both])
            only_e = sum(1 for r in rr if r["event_delay_ms"] is not None and r["frame_delay_ms"] is None)
            only_f = sum(1 for r in rr if r["frame_delay_ms"] is not None and r["event_delay_ms"] is None)
            neither = sum(1 for r in rr if r["frame_delay_ms"] is None and r["event_delay_ms"] is None)
            print(f"-- {cls}: tracks {len(rr)}  found by frame-only {len(fd)}  by event+frame {len(ed)}  "
                  f"event-only {only_e}  frame-only {only_f}  neither {neither}")
            if len(fd):
                print(f"   delay after first label, frame-only : median {np.median(fd):6.0f} ms  p25 {q(fd,25):6.0f}  p75 {q(fd,75):6.0f}  mean {fd.mean():6.0f}")
            if len(ed):
                print(f"   delay after first label, event+frame: median {np.median(ed):6.0f} ms  p25 {q(ed,25):6.0f}  p75 {q(ed,75):6.0f}  mean {ed.mean():6.0f}")
            if len(lead):
                pos = int((lead > 0).sum()); zero = int((lead == 0).sum()); neg = int((lead < 0).sum())
                print(f"   lead (frame - event) on {len(lead)} tracks found by both: median {np.median(lead):5.0f} ms  mean {lead.mean():5.0f}  "
                      f"p25 {q(lead,25):5.0f}  p75 {q(lead,75):5.0f}   earlier/same/later: {pos}/{zero}/{neg}")
                hist = {}
                for v in lead:
                    hist[int(v)] = hist.get(int(v), 0) + 1
                print("   lead histogram (ms: n):", " ".join(f"{k}:{v}" for k, v in sorted(hist.items())))


if __name__ == "__main__":
    main()
