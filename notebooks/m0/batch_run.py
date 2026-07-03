"""Container-side: run the full wrap on each batch .h5 -- extract near-vertical track,
un-correct with the obs TLE, write rffit .dat + site, run `rffit -I` against that obs's
epoch-matched catalog, and tabulate whether the TRUE object (from the filename) is the
top match. This is the 4-pass batch validation of the Milestone-0 result."""

import os, glob, subprocess

from satnogs_id.id.dat import build_dat, site_line
from satnogs_id.shared.waterfall import TrackParams, load_waterfall


def make_dat(h5path, siteid, datpath):
    # Same track extraction as the shipped wrap (80th-pct SNR gate, +-4 kHz window, sub-bin
    # parabolic peak, 3x MAD); min_points=1 keeps this diagnostic's original gate-less behaviour.
    wf = load_waterfall(h5path)
    n = build_dat(wf, siteid, datpath, TrackParams(min_points=1))
    with open("/opt/strf/data/sites.txt", "a", encoding="utf-8") as g:
        g.write(site_line(wf, siteid))
    return n


results = []
for k, h5path in enumerate(sorted(glob.glob("/data/batch/*.h5"))):
    base = os.path.basename(h5path)
    oid = base.split("_")[0][3:]
    truenorad = int(base.split("_n")[1].split("_")[0])
    siteid = 9001 + k
    datp = f"/data/strf/{oid}.dat"
    catp = f"/data/strf/soup_{oid}.tle"
    npts = make_dat(h5path, siteid, datp)
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
    pred = cand[0][1] if cand else None
    trms = next((r for r, n in cand if n == truenorad), None)
    trank = next((i + 1 for i, (r, n) in enumerate(cand) if n == truenorad), None)
    bconf = next((r for r, n in cand if n != truenorad), None)
    results.append((oid, truenorad, pred, trank, trms, bconf, npts))

print(
    f"\n{'obs':>9} {'true':>6} {'predicted':>9} {'rank':>4} {'true_rms':>8} {'conf_rms':>8} {'pts':>4}  result"
)
ncorrect = 0
for oid, tn, pr, rk, trms, brms, npts in results:
    ok = pr == tn
    ncorrect += ok
    print(
        f"{oid:>9} {tn:>6} {str(pr):>9} {str(rk):>4} {(trms or 0):>8.3f} {(brms or 0):>8.3f} {npts:>4}  {'CORRECT' if ok else 'WRONG'}"
    )
print(f"\n=== {ncorrect}/{len(results)} passes correctly identified ===")
