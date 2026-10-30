"""Operator-visible box error on labelled DSEC sequences, for the cue pipelines compared in the paper.

At every operator time Q (10 ms grid) and for every labelled track, each pipeline shows the newest box that
has reached the operator; its error is the distance to the label at Q, i.e. to where the object is at the
moment the operator looks. Timing is modelled with on-vehicle measurements:
  frame         detector without the event branch, every image, box available LAT_A ms after the image
  frame+cv      same boxes, extrapolated to Q at the velocity of the last two boxes of the track
  frame+events  detector with the event branch, every PERIOD_B-th image, available LAT_B ms after the image
  ours          frame+events boxes as anchors, moved by events alone every 10 ms; output AGE_C ms old
  frame+track   frame boxes as anchors, moved by events alone (isolates the tracker from the event branch)
  video         where the object appears in the displayed video frame (VIDEO_MS behind)
Cue pipelines add NET_MS of cue channel to the operator. Boxes come from lt2 *_evals.jsonl (score 0.3,
matched to labels, so a pipeline has a box only for tracks its detector found). Output: summary JSON + jsonl."""
import sys, os, json, time
sys.argv = ["x", "--config", "config/dagr-s-dsec.yaml", "--use_image", "--img_net", "resnet50", "--checkpoint",
            "data/dagr_s_50.pth", "--batch_size", "1", "--n_nodes", "20000", "--dataset_directory", "/dsec_night",
            "--output_directory", "/tmp/x"]
sys.path.insert(0, "/opt/dagr/src")
import numpy as np, cv2
from pathlib import Path
from dagr.utils.args import FLAGS
from dagr.data.dsec_data import DSEC
from dagr.data.augment import Augmentations
SEQS = os.environ.get("EV_SEQS", "zurich_city_09_b,zurich_city_09_c,zurich_city_09_d,zurich_city_10_b").split(",")
ROOT = os.environ.get("EV_ROOT", "/dsec_night")
MS = 1000
LAT_A = int(os.environ.get("LAT_A_MS", "53")) * MS; LAT_B = int(os.environ.get("LAT_B_MS", "116")) * MS
PERIOD_B = int(os.environ.get("PERIOD_B", "2")); AGE_C = int(os.environ.get("AGE_C_MS", "10")) * MS
NET = int(os.environ.get("NET_MS", "135")) * MS; VIDEO = int(os.environ.get("VIDEO_MS", "285")) * MS
HOR = 250 * MS; STEP = 10 * MS; XWIN = 10 * MS; SEARCH = 6; BLUR = 1.0; MINEV = 12; KEY = 30 * MS; THR = "0.3"
MAXP = int(os.environ.get("EV_MAX_PAIRS", "0")); TAG = os.environ.get("EV_TAG", "")
LT2 = os.environ.get("EV_LT2", "lt2")   # input prefix: lt2 (DAGR frame branch) or e.g. lt2y26n (YOLO as the frame detector)
NOISE_HZ = float(os.environ.get("EV_NOISE_HZ", "0")); RNG = np.random.default_rng(0)
ARMS = ["video", "frame", "frame+cv", "frame+events", "frame+events+cv", "frame+track", "frame+track+pred", "ours", "ours+pred"]
args = FLAGS()


def cerr(a, b): return float(np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2))


def hist(xx, yy, x0, y0, W, H):
    xi = (xx - x0).astype(np.int32); yi = (yy - y0).astype(np.int32); k = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    h = np.bincount(yi[k] * W + xi[k], minlength=H * W).reshape(H, W).astype(np.float32)
    return cv2.GaussianBlur(h, (0, 0), BLUR)


def shift(ref, cur):
    S = SEARCH; H, W = ref.shape; tpl = ref[S:H - S, S:W - S]
    if tpl.shape[0] < 2 or tpl.shape[1] < 2: return None
    sc = cv2.matchTemplate(cur, tpl, cv2.TM_CCORR_NORMED); iy, ix = np.unravel_index(int(np.argmax(sc)), sc.shape)
    if sc[iy, ix] - sc[S, S] < 0.02: return 0.0, 0.0
    def sub(a, b, c):
        d = a - 2 * b + c
        return 0.0 if abs(d) < 1e-9 else 0.5 * (a - c) / d
    fx = sub(sc[iy, ix - 1], sc[iy, ix], sc[iy, ix + 1]) if 0 < ix < sc.shape[1] - 1 else 0.0
    fy = sub(sc[iy - 1, ix], sc[iy, ix], sc[iy + 1, ix]) if 0 < iy < sc.shape[0] - 1 else 0.0
    return ix - S + fx, iy - S + fy


def propagate(anchor_boxes, t_a, ev):
    """move boxes from time t_a with events only; returns {t: {tid: box}} on the 10 ms grid up to t_a+HOR"""
    x, y, ta = ev
    st = {}
    a, b = np.searchsorted(ta, [t_a, t_a + XWIN])
    for tid, bx in anchor_boxes.items():
        w = bx[2] - bx[0]; h = bx[3] - bx[1]; m = SEARCH + 2
        X0, Y0 = int(np.floor(bx[0])) - m, int(np.floor(bx[1])) - m; W, H = int(np.ceil(w)) + 2 * m, int(np.ceil(h)) + 2 * m
        st[tid] = {"box": list(bx), "ref": hist(x[a:b], y[a:b], X0, Y0, W, H), "X0": X0, "Y0": Y0, "W": W, "H": H,
                   "kt": t_a + XWIN // 2, "kbox": list(bx)}
    out = {t_a: {k: list(v) for k, v in anchor_boxes.items()}, t_a + STEP: {k: list(v) for k, v in anchor_boxes.items()}}
    for d in range(2 * XWIN, HOR + 1, STEP):
        tc = t_a + d; a, b = np.searchsorted(ta, [tc - XWIN, tc]); xx, yy = x[a:b], y[a:b]
        for tid, s in st.items():
            if s["ref"].sum() < MINEV: continue
            cur = hist(xx, yy, s["X0"], s["Y0"], s["W"], s["H"])
            if cur.sum() < MINEV: continue
            sh = shift(s["ref"], cur)
            if sh is None: continue
            f = (tc - s["kt"]) / max(1, (tc - XWIN // 2) - s["kt"]); kb = s["kbox"]
            s["box"] = [kb[0] + sh[0] * f, kb[1] + sh[1] * f, kb[2] + sh[0] * f, kb[3] + sh[1] * f]
            if tc - s["kt"] >= KEY:
                bx = s["box"]; m = SEARCH + 2
                s["X0"], s["Y0"] = int(np.floor(bx[0])) - m, int(np.floor(bx[1])) - m
                s["ref"] = hist(xx, yy, s["X0"], s["Y0"], s["W"], s["H"]); s["kt"] = tc - XWIN // 2; s["kbox"] = list(bx)
        out[tc] = {k: list(s["box"]) for k, s in st.items()}
    return out



def ext(bx, v, dt):
    return [bx[0] + v[0] * dt, bx[1] + v[1] * dt, bx[2] + v[0] * dt, bx[3] + v[1] * dt]


def prop_lookup(P, ti, qv, tid):
    """tracker output that has left the vehicle by qv: box, its info time, and a velocity from the last 50 ms of the track"""
    tq = ((qv - AGE_C - ti) // STEP) * STEP + ti
    tq = min(max(tq, ti), max(P))
    if tid not in P[tq]: return None
    b = P[tq][tid]; tp = tq - 50 * MS; v = None
    if tp in P and tid in P[tp]:
        pb = P[tp][tid]; v = ((b[0] + b[2] - pb[0] - pb[2]) / 2 / 0.05, (b[1] + b[3] - pb[1] - pb[3]) / 2 / 0.05)
    t_info = qv - AGE_C if tq < max(P) else tq
    return b, t_info, v

summary = {"params": {k: v for k, v in dict(LAT_A=LAT_A, LAT_B=LAT_B, PERIOD_B=PERIOD_B, AGE_C=AGE_C, NET=NET, VIDEO=VIDEO,
                                             NOISE_HZ=NOISE_HZ).items()}, "seqs": {}}
for seq in SEQS:
    t_start = time.time()
    recs = {}; gt_at = {}
    for l in open("/output/%s_%s_evals.jsonl" % (LT2, seq)):
        r = json.loads(l)
        recs.setdefault(r["pair"], {})[r["model"] if r["model"] == "frame" else r["t"] - r["t0"]] = r
        gt_at[r["t"]] = {g[0]: g[1:] for g in r["gt"]}
    S_ = json.load(open("/output/%s_%s_summary.json" % (LT2, seq)))
    cls_of = {int(r["track"]): r["cls"] for r in S_["by_score"][THR]["tracks"]}
    ds = DSEC(Path(ROOT + "/one_" + seq), "test", Augmentations.transform_testing, debug=False,
              min_bbox_diag=15, min_bbox_height=10, demo=True)
    ds.no_eval = True; ds.set_num_us(50 * MS)
    cache = {}

    def window(i):
        if i not in cache:
            if len(cache) > 12: cache.pop(min(cache))
            d = ds[i]; tk = int(d.t0)
            x = d.pos[:, 0].numpy().astype(np.float32); y = d.pos[:, 1].numpy().astype(np.float32)
            t = tk + 50 * MS - (1_000_000 - d.t.numpy().astype(np.int64))
            if NOISE_HZ > 0:
                nn = RNG.poisson(NOISE_HZ * 640 * 480 * 0.05 * (ds.width * ds.height) / (320 * 240))
                x = np.concatenate([x, RNG.uniform(0, ds.width, nn).astype(np.float32)])
                y = np.concatenate([y, RNG.uniform(0, ds.height, nn).astype(np.float32)])
                t = np.concatenate([t, RNG.integers(tk, tk + 50 * MS, nn)])
            cache[i] = (tk, x, y, t)
        return cache[i]

    def events_from(i, n):
        xs, ys, ts = [], [], []; t0 = None
        for k in range(n):
            if i + k >= len(ds): break
            tk, x, y, t = window(i + k)
            if t0 is None: t0 = tk
            if tk - t0 > k * 50 * MS + 5 * MS: break
            xs.append(x); ys.append(y); ts.append(t)
        x = np.concatenate(xs); y = np.concatenate(ys); t = np.concatenate(ts); o = np.argsort(t, kind="stable")
        return x[o], y[o], t[o]

    pairs = sorted(recs)
    if MAXP: pairs = pairs[:MAXP]
    t_img = {i: recs[i]["frame"]["t0"] for i in pairs if "frame" in recs[i]}
    # ---- published boxes per pipeline: list of (available_at, info_time_fn, boxes_fn)
    pubA, pubB, propA, propB = [], [], {}, {}
    prevA = {}; prevB = {}
    for i in pairs:
        if i not in t_img: continue
        tk = t_img[i]
        fa = {b[0]: b[1:] for b in recs[i]["frame"]["mb@" + THR]}
        vel = {}
        for tid, bx in fa.items():
            if tid in prevA and 0 < tk - prevA[tid][0] <= 120 * MS:
                dt = (tk - prevA[tid][0]) / 1e6; pb = prevA[tid][1]
                vel[tid] = ((bx[0] + bx[2] - pb[0] - pb[2]) / 2 / dt, (bx[1] + bx[3] - pb[1] - pb[3]) / 2 / dt)
            prevA[tid] = (tk, bx)
        pubA.append((tk + LAT_A, tk, fa, vel, i))
        if i % PERIOD_B == 0 and 10 * MS in recs[i]:
            fb = {b[0]: b[1:] for b in recs[i][10 * MS]["mb@" + THR]}
            velB = {}
            for tid, bx in fb.items():
                if tid in prevB and 0 < tk - prevB[tid][0] <= 250 * MS:
                    dt = (tk - prevB[tid][0]) / 1e6; pb = prevB[tid][1]
                    velB[tid] = ((bx[0] + bx[2] - pb[0] - pb[2]) / 2 / dt, (bx[1] + bx[3] - pb[1] - pb[3]) / 2 / dt)
                prevB[tid] = (tk, bx)
            pubB.append((tk + LAT_B, tk + 10 * MS, fb, velB, i))
    need = {p[4] for p in pubA if p[2]} | {p[4] for p in pubB if p[2]}
    for i in sorted(need):
        ev = events_from(i, HOR // (50 * MS) + 1)
        tk = t_img[i]
        fa = {b[0]: b[1:] for b in recs[i]["frame"]["mb@" + THR]}
        if fa: propA[i] = propagate(fa, tk, ev)
        if i % PERIOD_B == 0 and 10 * MS in recs[i]:
            fb = {b[0]: b[1:] for b in recs[i][10 * MS]["mb@" + THR]}
            if fb: propB[i] = propagate(fb, tk + 10 * MS, ev)
    # ---- operator timeline
    gts = np.array(sorted(gt_at)); rows = []
    availA = np.array([p[0] for p in pubA]); availB = np.array([p[0] for p in pubB]) if pubB else np.zeros(0)
    q0 = gts[0] + 400 * MS; q1 = gts[-1]
    for Q in range(int(q0), int(q1) + 1, STEP):
        j = np.searchsorted(gts, Q); cand = [gts[k] for k in (j - 1, j) if 0 <= k < len(gts)]
        tg = min(cand, key=lambda c: abs(c - Q)) if cand else None
        if tg is None or abs(tg - Q) > 2 * MS: continue
        G = gt_at[tg]
        qv = Q - NET                              # newest cue that reached the operator left the vehicle by qv
        kA = np.searchsorted(availA, qv, side="right") - 1
        kB = np.searchsorted(availB, qv, side="right") - 1
        # video: frame shown at Q is the newest image older than Q - VIDEO
        tv = Q - VIDEO; jv = np.searchsorted(gts, tv, side="right") - 1
        Gv = gt_at[gts[jv]] if jv >= 0 else {}
        for tid, gb in G.items():
            c = cls_of.get(tid, "?")
            if c not in ("pedestrian", "car"): continue
            e = {"Q": Q, "tid": tid, "cls": c}
            if tid in Gv: e["video"] = (cerr(Gv[tid], gb) * 2, (Q - gts[jv]) / 1e3)
            if kA >= 0:
                av, ti, fa, vel, i = pubA[kA]
                if tid in fa:
                    e["frame"] = (cerr(fa[tid], gb) * 2, (Q - ti) / 1e3)
                    if tid in vel:
                        dt = (Q - ti) / 1e6; bx = fa[tid]
                        cb = [bx[0] + vel[tid][0] * dt, bx[1] + vel[tid][1] * dt, bx[2] + vel[tid][0] * dt, bx[3] + vel[tid][1] * dt]
                    else:
                        cb = fa[tid]
                    e["frame+cv"] = (cerr(cb, gb) * 2, (Q - ti) / 1e3)
                    P = propA.get(i)
                    if P:
                        L = prop_lookup(P, ti, qv, tid)
                        if L:
                            b, t_info, v = L
                            e["frame+track"] = (cerr(b, gb) * 2, (Q - t_info) / 1e3)
                            vv = v if v is not None else vel.get(tid)
                            e["frame+track+pred"] = (cerr(ext(b, vv, (Q - t_info) / 1e6) if vv else b, gb) * 2, (Q - t_info) / 1e3)
            if kB >= 0:
                av, ti, fb, velB, i = pubB[kB]
                if tid in fb:
                    e["frame+events"] = (cerr(fb[tid], gb) * 2, (Q - ti) / 1e3)
                    e["frame+events+cv"] = (cerr(ext(fb[tid], velB[tid], (Q - ti) / 1e6) if tid in velB else fb[tid], gb) * 2, (Q - ti) / 1e3)
                    P = propB.get(i)
                    if P:
                        L = prop_lookup(P, ti, qv, tid)
                        if L:
                            b, t_info, v = L
                            e["ours"] = (cerr(b, gb) * 2, (Q - t_info) / 1e3)
                            vv = v if v is not None else velB.get(tid)
                            e["ours+pred"] = (cerr(ext(b, vv, (Q - t_info) / 1e6) if vv else b, gb) * 2, (Q - t_info) / 1e3)
            rows.append(e)
    fo = open("/output/evcue/timeline%s_%s.jsonl" % (TAG, seq), "w")
    for e in rows: fo.write(json.dumps(e) + "\n")
    fo.close()
    print("[timeline] %s pairs %d instants %d %.0fs" % (seq, len(pairs), len(rows), time.time() - t_start), flush=True)
