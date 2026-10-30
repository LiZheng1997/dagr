"""YOLO as the frame detector of the operator-visible comparison.

For every image pair of a DSEC sequence, runs an Ultralytics YOLO model on the image at t0 (the same file and
the same 430-row crop the DAGR dataset uses, at full resolution) and matches its boxes to the labels exactly
like lead_time_dsec.py (same class, IoU >= 0.5, greedy by score). Writes lt2-format records in which the
"frame" model is YOLO; every other record (the event+frame detector at t0+d) is copied unchanged, so the
timeline script can be pointed at the new file (EV_LT2=<prefix>) and only the frame-detector arms change.
COCO classes map to the DSEC evaluation classes: person -> pedestrian, car/bus/truck -> car.

  YOLO_MODEL=yolo26n.pt YOLO_SEQS=zurich_city_09_b YOLO_ROOT=/dsec_night YOLO_TAG=y26n python3 yolo_frame_lt2.py
"""
import os, sys, json, time
sys.path.insert(0, "/output/yolo/pylib"); sys.path.insert(0, "/output/yolo"); sys.path.insert(0, "/opt/dagr/src")
SEQS = os.environ["YOLO_SEQS"].split(","); ROOT = os.environ.get("YOLO_ROOT", "/dsec_night")
MODEL = os.environ.get("YOLO_MODEL", "yolo26n.pt"); TAG = os.environ.get("YOLO_TAG", "y26n")
IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640")); THRS = [0.3, 0.5]; IOU = 0.5
sys.argv = ["x", "--config", "config/dagr-s-dsec.yaml", "--use_image", "--img_net", "resnet50", "--checkpoint",
            "data/dagr_s_50.pth", "--batch_size", "1", "--n_nodes", "20000", "--dataset_directory", ROOT,
            "--output_directory", "/tmp/x"]
import numpy as np, torch
from pathlib import Path
from dagr.data.dsec_data import DSEC
from dagr.data.augment import Augmentations
from lead_time_dsec import prep_tracks, match, gt_list, MIN_DIAG, MIN_H
from ultralytics import YOLO

COCO_TO_EVAL = {0: 1, 2: 0, 5: 0, 7: 0}          # person -> pedestrian(1); car, bus, truck -> car(0)
model = YOLO("/output/yolo/" + MODEL)
times = []
for seq in SEQS:
    ds = DSEC(Path(ROOT + "/one_" + seq), "test", Augmentations.transform_testing, debug=False,
              min_bbox_diag=MIN_DIAG, min_bbox_height=MIN_H, demo=True)
    ds.no_eval = True
    orig = [json.loads(l) for l in open("/output/lt2_%s_evals.jsonl" % seq)]
    frame_t0 = {r["pair"]: r["t0"] for r in orig if r["model"] == "frame"}
    out = open("/output/lt2%s_%s_evals.jsonl" % (TAG, seq), "w")
    H = ds.scale * ds.height
    for i in range(len(ds)):
        directory, pairs, track_mask, idx = ds.rel_index(i)
        name = directory.root.name; i0, _ = pairs[idx]
        t0 = int(directory.images.timestamps[i0])
        assert frame_t0.get(i) == t0, ("pair/time mismatch with the lt2 file", seq, i)
        tr0 = prep_tracks(ds.dataset.get_tracks(i0, mask=track_mask, directory_name=name), ds)
        img = ds.dataset.get_image(i0, directory_name=name)[:H]          # BGR, full resolution, same crop
        torch.cuda.synchronize(); t = time.perf_counter()
        r = model.predict(img, imgsz=IMGSZ, conf=0.05, classes=list(COCO_TO_EVAL), half=True, verbose=False)[0]
        torch.cuda.synchronize(); dt = (time.perf_counter() - t) * 1000.0
        if i >= 5: times.append(dt)
        b = r.boxes
        xyxy = b.xyxy.cpu().numpy() / ds.scale
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, ds.width); xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, ds.height)
        det = {"boxes": torch.from_numpy(xyxy.astype(np.float32)), "scores": b.conf.cpu().float(),
               "labels": torch.tensor([COCO_TO_EVAL[int(c)] for c in b.cls.cpu().numpy()], dtype=torch.long)}
        rec = {"seq": name, "pair": i, "t0": t0, "model": "frame", "t": t0, "ms": round(dt, 1), "n_gt": len(tr0),
               "detector": MODEL}
        for thr in THRS:
            m, fp, nd, mb = match(det, tr0, IOU, thr)
            rec["m@%s" % thr] = m; rec["fp@%s" % thr] = fp; rec["mb@%s" % thr] = mb
        rec["gt"] = gt_list(tr0)
        out.write(json.dumps(rec) + "\n")
    for r in orig:                                   # event+frame records unchanged
        if r["model"] != "frame":
            out.write(json.dumps(r) + "\n")
    out.close()
    S = json.load(open("/output/lt2_%s_summary.json" % seq)); S["frame_detector"] = MODEL
    json.dump(S, open("/output/lt2%s_%s_summary.json" % (TAG, seq), "w"))
    t = np.array(times)
    print("[yolo] %s %s pairs %d | predict ms p50 %.1f p90 %.1f" % (MODEL, seq, len(ds), np.median(t), np.percentile(t, 90)), flush=True)
