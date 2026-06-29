"""Container-side scaled eval: run the strf/rffit wrap over every pass in /data/eval/,
and report honest metrics -- top-1 identification accuracy with a Wilson 95% CI, the rank
distribution of the true object, the margin distribution, and a per-object breakdown."""
import h5py, json, numpy as np, os, glob, subprocess, math
from datetime import datetime, timedelta, timezone
from skyfield.api import load, wgs84, EarthSatellite
C = 299792.458; ts = load.timescale(builtin=True); EPOCH = datetime(1858,11,17, tzinfo=timezone.utc)
NAME = {64879:'Geoscan-6',64880:'Geoscan-1',64890:'Geoscan-2',64891:'Geoscan-5',64892:'Geoscan-4',64893:'Geoscan-3'}

def make_dat(h5path, siteid, datpath):
    f = h5py.File(h5path,'r'); m = json.loads(f.attrs['metadata'])
    f0 = float(m['frequency']); loc = m['location']; t0 = m['tle'].strip().splitlines()
    wf = f['waterfall']; data = wf['data'][:]; freqax = wf['frequency'][:].astype(float)
    scale = wf['scale'][:].astype(float); offset = wf['offset'][:].astype(float); relt = wf['relative_time'][:].astype(float)
    _st = wf.attrs['start_time']; _st = _st.decode() if isinstance(_st, bytes) else _st
    start = datetime.fromisoformat(_st.replace('Z','+00:00')); T, F = data.shape
    dB = data.astype(np.float32)*scale[None,:] + offset[None,:]; dfbin = float(freqax[1]-freqax[0])
    peak0 = np.argmax(dB, axis=1); snr = dB[np.arange(T), peak0] - np.median(dB, axis=1)
    hi = np.where(snr >= np.percentile(snr, 80))[0]
    if len(hi) < 10: return 0
    cbin = int(np.argmax(dB[hi].mean(axis=0))); win = max(3, int(4000/dfbin))
    st = wgs84.latlon(loc['latitude'], loc['longitude'], elevation_m=loc['altitude']); sat = EarthSatellite(t0[1], t0[2], 'x', ts)
    pts = []
    for i in hi:
        lo, hib = max(1, cbin-win), min(F-1, cbin+win); p = lo + int(np.argmax(dB[i, lo:hib]))
        y0, y1, y2 = dB[i,p-1], dB[i,p], dB[i,p+1]; den = y0-2*y1+y2
        delta = float(np.clip(0.5*(y0-y2)/den, -1, 1)) if den != 0 else 0.0
        pts.append((float(relt[i]), freqax[p] + delta*dfbin))
    pts = np.array(pts)
    for _ in range(3):
        med = np.median(pts[:,1]); mad = np.median(np.abs(pts[:,1]-med)) + 1e-9
        pts = pts[np.abs(pts[:,1]-med) < 4*mad]
    rows = []
    for relt_i, foff in pts:
        dt = start + timedelta(seconds=float(relt_i)); pos = (sat-st).at(ts.from_datetime(dt))
        r = pos.position.km; v = pos.velocity.km_per_s; rr = float(np.sum(r*v)/np.linalg.norm(r))
        rows.append(((dt-EPOCH).total_seconds()/86400.0, f0 + float(foff) - f0*rr/C))
    rows.sort()
    with open(datpath,'w') as g:
        for mj, fr in rows: g.write(f'{mj:.6f}\t{fr:.2f}\t1.0\t{siteid}\n')
    with open('/opt/strf/data/sites.txt','a') as g:
        g.write(f"{siteid} GS {loc['latitude']:.4f} {loc['longitude']:.4f} {int(loc['altitude'])} obs\n")
    return len(rows)

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0, c-h), min(1, c+h))

rows_out, by_obj, ranks, margins = [], {}, [], []
for k, h5path in enumerate(sorted(glob.glob('/data/eval/*.h5'))):
    base = os.path.basename(h5path); oid = base.split('_')[0][3:]; truenorad = int(base.split('_n')[1].split('_')[0])
    siteid = 9001 + (k % 900); datp = f'/data/eval/{oid}.dat'; catp = f'/data/eval/soup_{oid}.tle'
    if not os.path.exists(catp): continue
    npts = make_dat(h5path, siteid, datp)
    if npts < 10: rows_out.append((oid, truenorad, None, None, npts)); continue
    out = subprocess.run(['/opt/strf/rffit','-d',datp,'-c',catp,'-s',str(siteid),'-I'],
                         capture_output=True, text=True, env={**os.environ,'ST_DATADIR':'/opt/strf'})
    cand = []
    for l in out.stdout.splitlines():
        if 'kHz' in l and ':' in l and l.split(':')[0].strip().isdigit():
            try: cand.append((float(l.split(':')[1].split('kHz')[0]), int(l.split(':')[0])))
            except Exception: pass
    cand.sort()
    if not cand: rows_out.append((oid, truenorad, None, None, npts)); continue
    pred = cand[0][1]; rank = next((i+1 for i,(r,n) in enumerate(cand) if n == truenorad), None)
    trms = next((r for r,n in cand if n == truenorad), None)
    bconf = next((r for r,n in cand if n != truenorad), None)
    correct = pred == truenorad
    rows_out.append((oid, truenorad, pred, rank, npts))
    by_obj.setdefault(truenorad, [0,0]); by_obj[truenorad][1] += 1; by_obj[truenorad][0] += correct
    if rank: ranks.append(rank)
    if correct and trms is not None and bconf is not None: margins.append(bconf - trms)

scored = [r for r in rows_out if r[2] is not None]
ncorrect = sum(1 for o,t,p,rk,n in scored if p == t); N = len(scored)
lo, hi = wilson(ncorrect, N)
print(f"=== Scaled eval: Geoscan cluster, {len(rows_out)} passes ({N} scored, {len(rows_out)-N} unusable) ===")
print(f"TOP-1 ACCURACY: {ncorrect}/{N} = {100*ncorrect/max(N,1):.1f}%  (95% Wilson CI {100*lo:.0f}-{100*hi:.0f}%)")
print(f"true-object rank distribution: " + ", ".join(f"rank{r}:{ranks.count(r)}" for r in sorted(set(ranks))))
if margins:
    a = np.array(margins); print(f"margin over best confuser (correct cases): median {np.median(a):.2f} kHz, min {a.min():.2f}, max {a.max():.2f}")
print("per-object top-1:")
for n in sorted(by_obj): c, t = by_obj[n]; print(f"  {NAME.get(n,n)} ({n}): {c}/{t}")
