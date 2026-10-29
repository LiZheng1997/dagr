import json, sys, glob, numpy as np
for tag in sys.argv[1:]:
    R = [json.loads(l) for f in glob.glob("/home/lz/dagr_outputs/evcue/evlong%s_*.jsonl" % tag) for l in open(f)]
    print("== variant '%s'  instants %d" % (tag, len(R)))
    for c in ("pedestrian", "car"):
        line = "  %-10s" % c
        for d in (20, 50, 80, 110, 150):
            S = [r for r in R if r["cls"] == c and r["d_ms"] == d]
            if S:
                line += "  +%3d ms held %4.1f xc %4.1f (n %d)" % (d, np.median([r["held"][0] for r in S]), np.median([r["xc"][0] for r in S]), len(S))
        print(line)
        S = [r for r in R if r["cls"] == c and r["d_ms"] == 110 and r["moved"] >= 4]
        if S:
            print("    +110 ms moved>=4px n=%d held %.1f xc %.1f, xc closer in %.0f%%" % (len(S), np.median([r["held"][0] for r in S]), np.median([r["xc"][0] for r in S]), 100 * np.mean([r["xc"][0] < r["held"][0] for r in S])))
