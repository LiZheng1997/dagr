"""Long-horizon variant of the event-cue evaluation: the frame-only anchor at image time t0 is moved by
events alone up to t0+150 ms (the regime of the deployed tracker, where the detector's next box arrives
~110 ms after its image), scored against the labels of the following images (interpolated at 10 ms
steps). Two template policies: fixed reference from [t0, t0+10 ms] and a keyframe renewed every KEY ms
around the moved box (the deployed tracker). Output: /output/evcue/evlong<TAG>_<seq>.jsonl"""
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
HOR = int(os.environ.get("EV_HOR_MS", "150")) * 1000; STEP = 10000; XWIN = 10000; SEARCH = 6; BLUR = 1.0; MINEV = 12
KEY = int(os.environ.get("EV_KEY_MS", "0")) * 1000; TAG = os.environ.get("EV_TAG", ""); THR = "0.3"
MAXP = int(os.environ.get("EV_MAX_PAIRS", "0"))
NOISE_HZ = float(os.environ.get("EV_NOISE_HZ", "0")); RNG = np.random.default_rng(0)
args = FLAGS()


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


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


for seq in SEQS:
    recs = {}; gt_at = {}
    for l in open("/output/lt2_%s_evals.jsonl" % seq):
        r = json.loads(l)
        recs.setdefault(r["pair"], {})[r["model"] if r["model"] == "frame" else r["t"] - r["t0"]] = r
        gt_at[r["t"]] = {g[0]: g[1:] for g in r["gt"]}
    gt_times = np.array(sorted(gt_at))
    S_ = json.load(open("/output/lt2_%s_summary.json" % seq))
    cls_of = {int(r["track"]): r["cls"] for r in S_["by_score"][THR]["tracks"]}
    ds = DSEC(Path("/dsec_night/one_" + seq), "test", Augmentations.transform_testing, debug=False,
              min_bbox_diag=15, min_bbox_height=10, demo=True)
    ds.no_eval = True; ds.set_num_us(50000)
    fo = open("/output/evcue/evlong%s_%s.jsonl" % (TAG, seq), "w"); n = 0; t_start = time.time()
    pairs = sorted(recs)
    if MAXP: pairs = pairs[:MAXP]
    for i in pairs:
        fr = recs[i].get("frame")
        if fr is None: continue
        anchors = {b[0]: b[1:] for b in fr["mb@" + THR]}
        if not anchors: continue
        # the loader caps the event count per window, so stitch the 50 ms windows of this and the next pairs
        xs, ys, ts = [], [], []; t0 = None
        for k in range(0, HOR // 50000):
            if i + k >= len(ds): break
            dk = ds[i + k]; tk = int(dk.t0)
            if t0 is None: t0 = tk
            if tk - t0 > k * 50000 + 5000: break          # gap in the image sequence
            xs.append(dk.pos[:, 0].numpy().astype(np.float32)); ys.append(dk.pos[:, 1].numpy().astype(np.float32))
            ts.append(tk + 50000 - (1_000_000 - dk.t.numpy().astype(np.int64)))
        x = np.concatenate(xs); y = np.concatenate(ys); ta = np.concatenate(ts)
        if NOISE_HZ > 0:   # v2e-style shot noise, uniform Poisson over the native 640x480 array (4 native px per eval px)
            span = int(ta.max() - t0) if len(ta) else HOR
            nn = RNG.poisson(NOISE_HZ * 640 * 480 * span * 1e-6 * (ds.width * ds.height) / (320 * 240))
            x = np.concatenate([x, RNG.uniform(0, ds.width, nn).astype(np.float32)]); y = np.concatenate([y, RNG.uniform(0, ds.height, nn).astype(np.float32)])
            ta = np.concatenate([ta, RNG.integers(t0, t0 + span, nn)])
        o = np.argsort(ta, kind="stable"); x, y, ta = x[o], y[o], ta[o]
        st = {}
        for tid, bx in anchors.items():
            w = bx[2] - bx[0]; h = bx[3] - bx[1]; m = SEARCH + 2
            X0, Y0 = int(np.floor(bx[0])) - m, int(np.floor(bx[1])) - m; W, H = int(np.ceil(w)) + 2 * m, int(np.ceil(h)) + 2 * m
            a, b = np.searchsorted(ta, [t0, t0 + XWIN])
            st[tid] = {"box": list(bx), "ref": hist(x[a:b], y[a:b], X0, Y0, W, H), "X0": X0, "Y0": Y0, "W": W, "H": H,
                       "kt": t0 + XWIN // 2, "kbox": list(bx)}
        for d in range(2 * XWIN, HOR + 1, STEP):
            tc = t0 + d; a, b = np.searchsorted(ta, [tc - XWIN, tc]); xx, yy = x[a:b], y[a:b]
            for tid, s in st.items():
                if s["ref"].sum() < MINEV: continue
                cur = hist(xx, yy, s["X0"], s["Y0"], s["W"], s["H"])
                if cur.sum() < MINEV: continue
                sh = shift(s["ref"], cur)
                if sh is None: continue
                f = (tc - s["kt"]) / max(1, (tc - XWIN // 2) - s["kt"])
                kb = s["kbox"]; s["box"] = [kb[0] + sh[0] * f, kb[1] + sh[1] * f, kb[2] + sh[0] * f, kb[3] + sh[1] * f]
                if KEY and tc - s["kt"] >= KEY:
                    bx = s["box"]; w = bx[2] - bx[0]; h = bx[3] - bx[1]; m = SEARCH + 2
                    s["X0"], s["Y0"] = int(np.floor(bx[0])) - m, int(np.floor(bx[1])) - m
                    s["ref"] = hist(xx, yy, s["X0"], s["Y0"], s["W"], s["H"]); s["kt"] = tc - XWIN // 2; s["kbox"] = list(bx)
            # score against the labels at tc (nearest labelled instant within 2 ms)
            j = np.searchsorted(gt_times, tc); cand = [gt_times[k] for k in (j - 1, j) if 0 <= k < len(gt_times)]
            tg = min(cand, key=lambda q: abs(q - tc)) if cand else None
            if tg is None or abs(tg - tc) > 2000: continue
            g = gt_at[tg]; g0 = {gg[0]: gg[1:] for gg in fr["gt"]}
            for tid, s in st.items():
                if tid not in g or tid not in g0: continue
                fo.write(json.dumps({"seq": seq, "pair": i, "d_ms": d // 1000, "tid": tid, "cls": cls_of.get(tid, "?"),
                                     "moved": cerr(g0[tid], g[tid]) * 2, "held": [cerr(anchors[tid], g[tid]) * 2, iou(anchors[tid], g[tid])],
                                     "xc": [cerr(s["box"], g[tid]) * 2, iou(s["box"], g[tid])]}) + "\n"); n += 1
    fo.close(); print("[evlong] %s pairs %d instants %d %.0fs" % (seq, len(pairs), n, time.time() - t_start), flush=True)
