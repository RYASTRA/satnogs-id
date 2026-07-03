"""Container-side scaled eval: run the strf/rffit wrap over every pass in /data/eval/,
and report honest metrics -- top-1 identification accuracy with a Wilson 95% CI, the rank
distribution of the true object, the margin distribution, and a per-object breakdown."""

import numpy as np, os, glob, subprocess, math

from satnogs_id.id.dat import build_dat, site_line
from satnogs_id.shared.waterfall import load_waterfall

NAME = {
    64879: "Geoscan-6",
    64880: "Geoscan-1",
    64890: "Geoscan-2",
    64891: "Geoscan-5",
    64892: "Geoscan-4",
    64893: "Geoscan-3",
}


def make_dat(h5path, siteid, datpath):
    # Identical extraction to the shipped wrap; build_dat returns 0 (<10 usable points -> unusable),
    # in which case we skip the site line just as this eval loop originally did.
    wf = load_waterfall(h5path)
    n = build_dat(wf, siteid, datpath)
    if n == 0:
        return 0
    with open("/opt/strf/data/sites.txt", "a", encoding="utf-8") as g:
        g.write(site_line(wf, siteid))
    return n


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


rows_out, by_obj, ranks, margins = [], {}, [], []
for k, h5path in enumerate(sorted(glob.glob("/data/eval/*.h5"))):
    base = os.path.basename(h5path)
    oid = base.split("_")[0][3:]
    truenorad = int(base.split("_n")[1].split("_")[0])
    siteid = 9001 + (k % 900)
    datp = f"/data/eval/{oid}.dat"
    catp = f"/data/eval/soup_{oid}.tle"
    if not os.path.exists(catp):
        continue
    npts = make_dat(h5path, siteid, datp)
    if npts < 10:
        rows_out.append((oid, truenorad, None, None, npts))
        continue
    out = subprocess.run(
        ["/opt/strf/rffit", "-d", datp, "-c", catp, "-s", str(siteid), "-I"],
        capture_output=True,
        text=True,
        env={**os.environ, "ST_DATADIR": "/opt/strf"},
    )
    cand = []
    for l in out.stdout.splitlines():
        if "kHz" in l and ":" in l and l.split(":")[0].strip().isdigit():
            try:
                cand.append(
                    (float(l.split(":")[1].split("kHz")[0]), int(l.split(":")[0]))
                )
            except Exception:
                pass
    cand.sort()
    if not cand:
        rows_out.append((oid, truenorad, None, None, npts))
        continue
    pred = cand[0][1]
    rank = next((i + 1 for i, (r, n) in enumerate(cand) if n == truenorad), None)
    trms = next((r for r, n in cand if n == truenorad), None)
    bconf = next((r for r, n in cand if n != truenorad), None)
    correct = pred == truenorad
    rows_out.append((oid, truenorad, pred, rank, npts))
    by_obj.setdefault(truenorad, [0, 0])
    by_obj[truenorad][1] += 1
    by_obj[truenorad][0] += correct
    if rank:
        ranks.append(rank)
    if correct and trms is not None and bconf is not None:
        margins.append(bconf - trms)

scored = [r for r in rows_out if r[2] is not None]
ncorrect = sum(1 for o, t, p, rk, n in scored if p == t)
N = len(scored)
lo, hi = wilson(ncorrect, N)
print(
    f"=== Scaled eval: Geoscan cluster, {len(rows_out)} passes ({N} scored, {len(rows_out) - N} unusable) ==="
)
print(
    f"TOP-1 ACCURACY: {ncorrect}/{N} = {100 * ncorrect / max(N, 1):.1f}%  (95% Wilson CI {100 * lo:.0f}-{100 * hi:.0f}%)"
)
print(
    f"true-object rank distribution: "
    + ", ".join(f"rank{r}:{ranks.count(r)}" for r in sorted(set(ranks)))
)
if margins:
    a = np.array(margins)
    print(
        f"margin over best confuser (correct cases): median {np.median(a):.2f} kHz, min {a.min():.2f}, max {a.max():.2f}"
    )
print("per-object top-1:")
for n in sorted(by_obj):
    c, t = by_obj[n]
    print(f"  {NAME.get(n, n)} ({n}): {c}/{t}")
