"""Container-side: run the full wrap on each batch .h5 -- extract near-vertical track,
un-correct with the obs TLE, write rffit .dat + site, run `rffit -I` against that obs's
epoch-matched catalog, and tabulate whether the TRUE object (from the filename) is the
top match. This is the 4-pass batch validation of the Milestone-0 result."""
import h5py, json, numpy as np, os, glob, subprocess
from datetime import datetime, timedelta, timezone
from skyfield.api import load, wgs84, EarthSatellite
C = 299792.458
ts = load.timescale(builtin=True)
EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)

def make_dat(h5path, siteid, datpath):
    f = h5py.File(h5path, 'r'); m = json.loads(f.attrs['metadata'])
    f0 = float(m['frequency']); loc = m['location']; t0 = m['tle'].strip().splitlines()
    wf = f['waterfall']; data = wf['data'][:]; freqax = wf['frequency'][:].astype(float)
    scale = wf['scale'][:].astype(float); offset = wf['offset'][:].astype(float); relt = wf['relative_time'][:].astype(float)
    _st = wf.attrs['start_time']; _st = _st.decode() if isinstance(_st, bytes) else _st
    start = datetime.fromisoformat(_st.replace('Z','+00:00'))
    T, F = data.shape
    dB = data.astype(np.float32)*scale[None,:] + offset[None,:]
    peak = np.argmax(dB, axis=1); snr = dB[np.arange(T), peak] - np.median(dB, axis=1)
    hi = snr >= np.percentile(snr, 85); carrier = float(np.median(freqax[peak[hi]]))
    idx = np.where(hi & (np.abs(freqax[peak]-carrier) < 6000))[0]
    st = wgs84.latlon(loc['latitude'], loc['longitude'], elevation_m=loc['altitude'])
    sat = EarthSatellite(t0[1], t0[2], 'x', ts)
    rows = []
    for i in idx:
        dt = start + timedelta(seconds=float(relt[i]))
        pos = (sat - st).at(ts.from_datetime(dt)); r = pos.position.km; v = pos.velocity.km_per_s
        rr = float(np.sum(r*v)/np.linalg.norm(r))
        rows.append(((dt-EPOCH).total_seconds()/86400.0, f0 + float(freqax[peak[i]]) - f0*rr/C))
    rows.sort()
    with open(datpath, 'w') as g:
        for mj, fr in rows: g.write(f'{mj:.6f}\t{fr:.2f}\t1.0\t{siteid}\n')
    with open('/opt/strf/data/sites.txt', 'a') as g:
        g.write(f"{siteid} GS {loc['latitude']:.4f} {loc['longitude']:.4f} {int(loc['altitude'])} obs\n")
    return len(rows)

results = []
for k, h5path in enumerate(sorted(glob.glob('/data/batch/*.h5'))):
    base = os.path.basename(h5path); oid = base.split('_')[0][3:]; truenorad = int(base.split('_n')[1].split('_')[0])
    siteid = 9001 + k; datp = f'/data/strf/{oid}.dat'; catp = f'/data/strf/soup_{oid}.tle'
    npts = make_dat(h5path, siteid, datp)
    out = subprocess.run(['/opt/strf/rffit','-d',datp,'-c',catp,'-s',str(siteid),'-I'],
                         capture_output=True, text=True, env={**os.environ,'ST_DATADIR':'/opt/strf'})
    cand = []
    for l in out.stdout.splitlines():
        if 'kHz' in l and ':' in l and l.split(':')[0].strip().isdigit():
            try: cand.append((float(l.split(':')[1].split('kHz')[0]), int(l.split(':')[0])))
            except Exception: pass
    cand.sort()
    pred = cand[0][1] if cand else None
    trms = next((r for r, n in cand if n == truenorad), None)
    trank = next((i+1 for i, (r, n) in enumerate(cand) if n == truenorad), None)
    bconf = next((r for r, n in cand if n != truenorad), None)
    results.append((oid, truenorad, pred, trank, trms, bconf, npts))

print(f"\n{'obs':>9} {'true':>6} {'predicted':>9} {'rank':>4} {'true_rms':>8} {'conf_rms':>8} {'pts':>4}  result")
ncorrect = 0
for oid, tn, pr, rk, trms, brms, npts in results:
    ok = pr == tn; ncorrect += ok
    print(f"{oid:>9} {tn:>6} {str(pr):>9} {str(rk):>4} {(trms or 0):>8.3f} {(brms or 0):>8.3f} {npts:>4}  {'CORRECT' if ok else 'WRONG'}")
print(f"\n=== {ncorrect}/{len(results)} passes correctly identified ===")
