"""Between-image box freshness from lead_time_dsec.py outputs with box logging.

For each image pair and each step d after the first image, compare against the
interpolated label at t0+d: (a) the frame-only box found at t0 and held, and
(b) the event+frame box produced at t0+d. Only tracks matched by both models
are used, so both columns describe the same objects. Errors are reported in
full-resolution pixels (evaluation frame x scale).

Usage: python3 lead_time_freshness.py [--score 0.3] [--scale 2] <prefix>...
"""
import argparse, json
import numpy as np

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0.0

def centre(b): return ((b[0]+b[2])/2, (b[1]+b[3])/2)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("prefixes", nargs="+"); ap.add_argument("--score", default="0.3")
    ap.add_argument("--scale", type=float, default=2.0, help="evaluation px -> full-res px")
    a = ap.parse_args(); key = f"mb@{a.score}"
    rows = []  # (cls, step_ms, err_held, err_event, iou_held, iou_event, label_motion)
    for pre in a.prefixes:
        S = json.load(open(pre + "_summary.json"))
        cls_of = {}
        for r in S["by_score"][a.score]["tracks"]: cls_of[int(r["track"])] = r["cls"]
        frame_boxes = {}  # (pair) -> {tid: box}
        frame_gt = {}
        for line in open(pre + "_evals.jsonl"):
            r = json.loads(line)
            if r["model"] == "frame":
                frame_boxes[r["pair"]] = {int(b[0]): b[1:] for b in r.get(key, [])}
                frame_gt[r["pair"]] = {int(g[0]): g[1:] for g in r.get("gt", [])}
                continue
            fb = frame_boxes.get(r["pair"], {}); g0 = frame_gt.get(r["pair"], {})
            eb = {int(b[0]): b[1:] for b in r.get(key, [])}
            gt = {int(g[0]): g[1:] for g in r.get("gt", [])}
            step = (r["t"] - r["t0"]) / 1000.0
            for tid, lab in gt.items():
                if tid not in fb or tid not in eb: continue
                c = cls_of.get(tid, "?")
                eh = np.hypot(*(np.subtract(centre(fb[tid]), centre(lab)))) * a.scale
                ee = np.hypot(*(np.subtract(centre(eb[tid]), centre(lab)))) * a.scale
                mo = np.hypot(*(np.subtract(centre(g0[tid]), centre(lab)))) * a.scale if tid in g0 else np.nan
                rows.append((c, step, eh, ee, iou(fb[tid], lab), iou(eb[tid], lab), mo))
    print(f"instants with both models matched: {len(rows)}  (score >= {a.score}, px at full resolution)")
    for c in ("pedestrian", "car"):
        R = [x for x in rows if x[0] == c]
        print(f"\n-- {c}: {len(R)} track-instants")
        print("   step   n   label moved   held frame box: err med / p75  IoU med   event+frame box: err med / p75  IoU med   event closer")
        for st in sorted(set(x[1] for x in R)):
            Q = [x for x in R if x[1] == st]
            eh = np.array([x[2] for x in Q]); ee = np.array([x[3] for x in Q]); ih = np.array([x[4] for x in Q]); ie = np.array([x[5] for x in Q]); mo = np.array([x[6] for x in Q])
            print(f"   {st:4.0f} {len(Q):5d}   {np.nanmedian(mo):5.1f} px       {np.median(eh):5.1f} / {np.percentile(eh,75):5.1f}   {np.median(ih):.2f}       {np.median(ee):5.1f} / {np.percentile(ee,75):5.1f}   {np.median(ie):.2f}      {100*np.mean(ee<eh):4.0f}%")
        eh = np.array([x[2] for x in R]); ee = np.array([x[3] for x in R]); ih = np.array([x[4] for x in R]); ie = np.array([x[5] for x in R])
        print(f"   all  {len(R):5d}                     {np.median(eh):5.1f} / {np.percentile(eh,75):5.1f}   {np.median(ih):.2f}       {np.median(ee):5.1f} / {np.percentile(ee,75):5.1f}   {np.median(ie):.2f}      {100*np.mean(ee<eh):4.0f}%")
        # moving objects only: label moved >= 4 px between images
        M = [x for x in R if x[1] == 50 and not np.isnan(x[6]) and x[6] >= 4]
        if M:
            eh = np.array([x[2] for x in M]); ee = np.array([x[3] for x in M])
            print(f"   at +50 ms, objects whose label moved >= 4 px in the interval: n={len(M)}  held err med {np.median(eh):.1f}  event err med {np.median(ee):.1f}  event closer {100*np.mean(ee<eh):.0f}%")

if __name__ == "__main__":
    main()
