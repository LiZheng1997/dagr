import json, sys, glob, numpy as np
ARMS = ["video", "frame", "frame+cv", "frame+events", "frame+events+cv", "frame+track", "frame+track+pred", "ours", "ours+pred"]
pat = sys.argv[1] if len(sys.argv) > 1 else "/home/lz/dagr_outputs/evcue/timeline_*.jsonl"
R = [json.loads(l) for f in sorted(glob.glob(pat)) for l in open(f)]
print("instants", len(R), "from", pat)
for c in ("pedestrian", "car"):
    S = [r for r in R if r["cls"] == c]
    if not S: continue
    common = [r for r in S if all(a in r for a in ARMS)]
    moving = [r for r in common if "video" in r and r["video"][0] >= 4]
    print("-- %s: %d track-instants, %d with every pipeline, %d of those moving (video error >= 4 px)" % (c, len(S), len(common), len(moving)))
    print("   %-14s %10s %12s %12s %10s" % ("pipeline", "coverage", "err p50 all", "err p50 mov", "age p50 ms"))
    for a in ARMS:
        have = [r for r in S if a in r]
        e_all = np.median([r[a][0] for r in common]) if common else float("nan")
        e_mov = np.median([r[a][0] for r in moving]) if moving else float("nan")
        age = np.median([r[a][1] for r in have]) if have else float("nan")
        print("   %-14s %9.0f%% %12.1f %12.1f %10.0f" % (a, 100 * len(have) / len(S), e_all, e_mov, age))
