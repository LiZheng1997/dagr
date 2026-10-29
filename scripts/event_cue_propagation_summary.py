import json, sys, glob, numpy as np
files = sys.argv[1:] or glob.glob("/home/lz/dagr_outputs/evcue/evcue_*.jsonl")
R = [json.loads(l) for f in files for l in open(f)]
keys = [k for k in ("held", "prop", "xc", "dagr") if k in R[0]]
for c in ("pedestrian", "car"):
    print("--", c, "(centre error px, full res, median; IoU median) over instants where DAGR also matched")
    for d in (10, 20, 30, 40, 50):
        S = [r for r in R if r["cls"] == c and r["d_ms"] == d and r["dagr"]]
        if not S:
            continue
        line = "  +%2d ms n=%5d " % (d, len(S))
        for k in keys:
            line += " %s %5.1f/%.2f" % (k, np.median([r[k][0] for r in S]), np.median([r[k][1] for r in S]))
        print(line)
    S = [r for r in R if r["cls"] == c and r["d_ms"] == 50 and r["dagr"] and r["moved"] >= 4]
    if S:
        line = "  +50 ms moved>=4px n=%d " % len(S)
        for k in keys:
            line += " %s %5.1f" % (k, np.median([r[k][0] for r in S]))
        print(line)
        for k in ("prop", "xc"):
            if k in keys:
                print("    %s closer than held in %.0f%% of instants" % (k, 100 * np.mean([r[k][0] < r["held"][0] for r in S])))
