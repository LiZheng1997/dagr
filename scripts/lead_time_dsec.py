"""First-detection lead time of the event+frame model over its frame-only output on one DSEC sequence.

For every labelled image pair (t0, t1) of the sequence the frame-only output (image at t0, no
events) is evaluated against the labels at t0, and the event+frame output is evaluated at
t0 + d for d in LT_STEPS_MS (events up to t0 + d, labels interpolated to t0 + d). A detection
matches a ground-truth track when the class agrees and IoU >= LT_IOU. For each track the first
matched time of each model is recorded; lead = first_frame_only - first_event_frame.

Extra settings come from the environment because FLAGS() owns argv:
  LT_STEPS_MS   comma list, default "10,20,30,40,50"
  LT_SCORES     comma list of score thresholds, default "0.3,0.5"
  LT_IOU        default 0.5
  LT_OUT        output prefix (JSONL of every evaluation + summary JSON)
  LT_MAX_PAIRS  optional cap for smoke runs

Run (container):
  cd /opt/dagr && LT_OUT=/output/lt_09b python3 scripts/lead_time_dsec.py --config config/dagr-s-dsec.yaml \
      --use_image --img_net resnet50 --checkpoint data/dagr_s_50.pth --batch_size 1 --n_nodes 20000 \
      --dataset_directory /dsec_night --output_directory /output/lt_log
"""
import json
import os
import time

import numpy as np
import torch
from torch_geometric.data import Batch

from dagr.data.augment import Augmentations
from dagr.data.dsec_data import DSEC
from dagr.data.dsec_utils import crop_tracks, filter_small_bboxes, map_classes, rescale_tracks
from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA
from dagr.utils.args import FLAGS
from dagr.utils.buffers import format_data

MIN_DIAG, MIN_H = 15, 10


def iou_xyxy(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-6)


def gt_struct_to_xyxy(tr):
    return np.stack([tr["x"], tr["y"], tr["x"] + tr["w"], tr["y"] + tr["h"]], axis=1).astype(np.float32)


def prep_tracks(tr, ds):
    """Same geometry/class/size treatment as the dataset's targets, keeping track ids."""
    tr = rescale_tracks(tr, ds.scale)
    tr = crop_tracks(tr, ds.width, ds.height)
    cls, cmask = map_classes(tr["class_id"], ds.class_remapping)
    tr = tr.copy(); tr["class_id"] = cls
    keep = cmask & filter_small_bboxes(tr["w"], tr["h"], MIN_H, MIN_DIAG)
    return tr[keep]


STRICT = os.environ.get("LT_STRICT", "0") == "1"


def interp_tracks(tr0, tr1, t):
    """Per-track linear interpolation between the labels at t0 and t1.

    A track labelled at t1 but not at t0 has no box before t1. In the default mode it keeps its
    t1 box for every t in (t0, t1], which credits the event model with a match before the first
    label while the frame-only model can only be scored at t0 (where the same detection would be
    a false positive). With LT_STRICT=1 such a track is only scored at t == t1, so both models get
    their first chance at the same instant and a tie is reported as 0 ms rather than -10 ms.
    """
    if len(tr1) == 0:
        return tr1
    out = tr1.copy()
    t1 = float(tr1["t"][0])
    if len(tr0) == 0:
        return out if (not STRICT or t >= t1) else out[:0]
    t0 = float(tr0["t"][0])
    r = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
    id0 = {int(i): k for k, i in enumerate(tr0["track_id"])}
    keep = np.ones(len(tr1), bool)
    for k, tid in enumerate(tr1["track_id"]):
        j = id0.get(int(tid))
        if j is None:
            if STRICT and t < t1:
                keep[k] = False
            continue
        for f in "xywh":
            out[f][k] = tr0[f][j] * (1 - r) + tr1[f][k] * r
    return out[keep]


def match(det, gt, iou_thr, score_thr):
    """Greedy by score. Returns (matched track ids, n_fp, n_det)."""
    boxes = det["boxes"].cpu().numpy(); scores = det["scores"].cpu().numpy(); labels = det["labels"].cpu().numpy()
    sel = scores >= score_thr
    boxes, scores, labels = boxes[sel], scores[sel], labels[sel]
    order = np.argsort(-scores)
    gtb = gt_struct_to_xyxy(gt); used = np.zeros(len(gt), bool)
    ious = iou_xyxy(boxes, gtb)
    matched, fp = [], 0
    for i in order:
        best, bj = 0.0, -1
        for j in range(len(gt)):
            if used[j] or int(gt["class_id"][j]) != int(labels[i]):
                continue
            if ious[i, j] > best:
                best, bj = ious[i, j], j
        if bj >= 0 and best >= iou_thr:
            used[bj] = True; matched.append(int(gt["track_id"][bj]))
        else:
            fp += 1
    return matched, fp, int(sel.sum())


def main():
    args = FLAGS()
    steps_us = [int(float(s) * 1000) for s in os.environ.get("LT_STEPS_MS", "10,20,30,40,50").split(",")]
    score_thrs = [float(s) for s in os.environ.get("LT_SCORES", "0.3,0.5").split(",")]
    iou_thr = float(os.environ.get("LT_IOU", "0.5"))
    out = os.environ.get("LT_OUT", "lead_time")
    max_pairs = int(os.environ.get("LT_MAX_PAIRS", "0"))

    ds = DSEC(args.dataset_directory, "test", Augmentations.transform_testing, debug=False,
              min_bbox_diag=MIN_DIAG, min_bbox_height=MIN_H, demo=True)
    ds.no_eval = True  # skip the dataset's own label interpolation; labels are handled here with track ids
    seqs = [d.name for d in ds.dataset.subsequence_directories]
    print(f"[lt] sequences {seqs} pairs {len(ds)} steps_us {steps_us} scores {score_thrs}", flush=True)

    model = DAGR(args, height=ds.height, width=ds.width).cuda()
    ema = ModelEMA(model)
    ck = torch.load(args.checkpoint, map_location="cuda")
    ema.ema.load_state_dict(ck["ema"])
    ema.ema.cache_luts(radius=args.radius, height=ds.height, width=ds.width)
    net = ema.ema.eval()
    assert hasattr(net.head, "no_events"), "head has no no_events switch"

    # first labelled time per track (after the same filters), for reference
    first_label = {}
    for name in seqs:
        tr = prep_tracks(ds.dataset.directories[name].tracks.tracks, ds)
        for tid in np.unique(tr["track_id"]):
            first_label[(name, int(tid))] = {"t": int(tr["t"][tr["track_id"] == tid].min()),
                                             "cls": int(tr["class_id"][tr["track_id"] == tid][0])}

    first_hit = {thr: {"frame": {}, "event": {}} for thr in score_thrs}
    fp_count = {thr: {"frame": 0, "event": 0} for thr in score_thrs}
    det_count = {thr: {"frame": 0, "event": 0} for thr in score_thrs}
    n_eval = {"frame": 0, "event": 0}
    ms = {"frame": [], "event": []}
    fout = open(out + "_evals.jsonl", "w")

    def run(data, no_events):
        net.head.no_events = no_events
        b = Batch.from_data_list([data], follow_batch=["bbox", "bbox0"]).cuda(non_blocking=True)
        b = format_data(b)
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.no_grad():
            dets, _ = net(b.clone())
        torch.cuda.synchronize()
        return dets[0], (time.perf_counter() - t) * 1000.0

    n_pairs = len(ds) if not max_pairs else min(max_pairs, len(ds))
    t_start = time.time()
    for i in range(n_pairs):
        directory, pairs, track_mask, idx = ds.rel_index(i)
        name = directory.root.name
        i0, i1 = pairs[idx]
        t0, t1 = (int(x) for x in directory.images.timestamps[[i0, i1]])
        tr0 = prep_tracks(ds.dataset.get_tracks(i0, mask=track_mask, directory_name=name), ds)
        tr1 = prep_tracks(ds.dataset.get_tracks(i1, mask=track_mask, directory_name=name), ds)

        # frame-only: image at t0, labels at t0
        ds.set_num_us(steps_us[-1])
        data = ds[i]
        det, dt = run(data, True); ms["frame"].append(dt); n_eval["frame"] += 1
        rec = {"seq": name, "pair": i, "t0": t0, "model": "frame", "t": t0, "ms": round(dt, 1), "n_gt": len(tr0)}
        for thr in score_thrs:
            m, fp, nd = match(det, tr0, iou_thr, thr)
            fp_count[thr]["frame"] += fp; det_count[thr]["frame"] += nd
            for tid in m:
                first_hit[thr]["frame"].setdefault((name, tid), t0)
            rec[f"m@{thr}"] = m; rec[f"fp@{thr}"] = fp
        fout.write(json.dumps(rec) + "\n")

        # event+frame at t0 + d
        for d in steps_us:
            ds.set_num_us(d)
            data = ds[i]
            gt = interp_tracks(tr0, tr1, t0 + d)
            det, dt = run(data, False); ms["event"].append(dt); n_eval["event"] += 1
            rec = {"seq": name, "pair": i, "t0": t0, "model": "event", "t": t0 + d, "ms": round(dt, 1),
                   "n_events": int(data.num_nodes), "n_gt": len(gt)}
            for thr in score_thrs:
                m, fp, nd = match(det, gt, iou_thr, thr)
                fp_count[thr]["event"] += fp; det_count[thr]["event"] += nd
                for tid in m:
                    first_hit[thr]["event"].setdefault((name, tid), t0 + d)
                rec[f"m@{thr}"] = m; rec[f"fp@{thr}"] = fp
            fout.write(json.dumps(rec) + "\n")

        if i % 20 == 0:
            print(f"[hb] pair {i}/{n_pairs} {time.time()-t_start:.0f}s frame {np.mean(ms['frame']):.0f} ms "
                  f"event {np.mean(ms['event']):.0f} ms", flush=True)
        if i % 10 == 0:
            torch.cuda.empty_cache()
    fout.close()

    summary = {"sequences": seqs, "pairs": n_pairs, "steps_us": steps_us, "iou": iou_thr,
               "ms_mean": {k: float(np.mean(v)) for k, v in ms.items()}, "n_eval": n_eval,
               "tracks_total": {"pedestrian": sum(1 for v in first_label.values() if v["cls"] == 1),
                                "car": sum(1 for v in first_label.values() if v["cls"] == 0)},
               "by_score": {}}
    for thr in score_thrs:
        fh = first_hit[thr]
        rows = []
        for key, lab in first_label.items():
            tf, te = fh["frame"].get(key), fh["event"].get(key)
            rows.append({"seq": key[0], "track": key[1], "cls": "pedestrian" if lab["cls"] == 1 else "car",
                         "t_label": lab["t"],
                         "frame_delay_ms": None if tf is None else (tf - lab["t"]) / 1000.0,
                         "event_delay_ms": None if te is None else (te - lab["t"]) / 1000.0,
                         "lead_ms": None if (tf is None or te is None) else (tf - te) / 1000.0})
        def stats(vals):
            vals = [v for v in vals if v is not None]
            return None if not vals else {"n": len(vals), "median": float(np.median(vals)), "mean": float(np.mean(vals)),
                                          "p25": float(np.percentile(vals, 25)), "p75": float(np.percentile(vals, 75))}
        per_cls = {}
        for c in ("pedestrian", "car"):
            rr = [r for r in rows if r["cls"] == c]
            per_cls[c] = {"tracks": len(rr),
                          "found_frame": sum(r["frame_delay_ms"] is not None for r in rr),
                          "found_event": sum(r["event_delay_ms"] is not None for r in rr),
                          "found_event_only": sum(r["event_delay_ms"] is not None and r["frame_delay_ms"] is None for r in rr),
                          "frame_delay_ms": stats([r["frame_delay_ms"] for r in rr]),
                          "event_delay_ms": stats([r["event_delay_ms"] for r in rr]),
                          "lead_ms": stats([r["lead_ms"] for r in rr])}
        summary["by_score"][str(thr)] = {
            "per_class": per_cls,
            "precision": {k: (1.0 - fp_count[thr][k] / det_count[thr][k]) if det_count[thr][k] else None for k in ("frame", "event")},
            "detections": det_count[thr], "false_positives": fp_count[thr], "tracks": rows}
    json.dump(summary, open(out + "_summary.json", "w"), indent=1)
    for thr in score_thrs:
        pc = summary["by_score"][str(thr)]["per_class"]
        print(f"[lt] score>={thr}: ped lead {pc['pedestrian']['lead_ms']}  car lead {pc['car']['lead_ms']}  "
              f"precision {summary['by_score'][str(thr)]['precision']}", flush=True)
    print(f"[lt] done -> {out}_summary.json", flush=True)


if __name__ == "__main__":
    main()
