"""Empirical correction test across MANY strong passes from DIFFERENT stations.
For each .h5: matched-filter profile vs carrier offset for two hypotheses --
  vertical line (constant offset)  -> Doppler-CORRECTED
  embedded-TLE Doppler curve       -> RAW/UNCORRECTED
Score per offset = mean of the top 25% along-line power (robust to intermittent
beacons). Prominence = (peak - median)/std of that profile. The hypothesis with the
sharper, more prominent peak reveals the signal's shape. No conclusion is asserted
here -- the per-obs table is what we read."""

import h5py, json, numpy as np, glob
from datetime import datetime, timedelta
from skyfield.api import load, wgs84, EarthSatellite

C = 299792.458
ts = load.timescale(builtin=True)


def topmean(x, frac=0.25):
    k = max(1, int(len(x) * frac))
    return float(np.mean(np.sort(x)[-k:]))


def analyze(path):
    f = h5py.File(path, "r")
    m = json.loads(f.attrs["metadata"])
    f0 = float(m["frequency"])
    loc = m["location"]
    tle = m["tle"].strip().splitlines()
    wf = f["waterfall"]
    data = wf["data"][:]
    freqax = wf["frequency"][:].astype(float)
    scale = wf["scale"][:].astype(float)
    offset = wf["offset"][:].astype(float)
    relt = wf["relative_time"][:].astype(float)
    _st = wf.attrs["start_time"]
    if isinstance(_st, bytes):
        _st = _st.decode()
    start = datetime.fromisoformat(_st.replace("Z", "+00:00"))
    T, F = data.shape
    dB = data.astype(np.float32) * scale[None, :] + offset[None, :]
    st = wgs84.latlon(loc["latitude"], loc["longitude"], elevation_m=loc["altitude"])
    tt = ts.from_datetimes([start + timedelta(seconds=float(r)) for r in relt])
    sat = EarthSatellite(tle[1], tle[2], tle[0], ts)
    pos = (sat - st).at(tt)
    r = pos.position.km
    v = pos.velocity.km_per_s
    pred = -np.sum(r * v, axis=0) / np.linalg.norm(r, axis=0) / C * f0
    grid = np.arange(-12000, 12000, 50.0)

    def profile(curve):
        out = np.empty(len(grid))
        for i, df in enumerate(grid):
            idx = np.clip(np.searchsorted(freqax, curve + df), 0, F - 1)
            acc = dB[np.arange(T), idx]
            for w in (1, 2, 3):
                acc = np.maximum(
                    acc,
                    np.maximum(
                        dB[np.arange(T), np.clip(idx + w, 0, F - 1)],
                        dB[np.arange(T), np.clip(idx - w, 0, F - 1)],
                    ),
                )
            out[i] = topmean(acc)
        return out

    vp, dp = profile(np.zeros(T)), profile(pred)
    prom = lambda p: (p.max() - np.median(p)) / (np.std(p) + 1e-9)
    return (
        m["observation_id"],
        prom(vp),
        prom(dp),
        grid[int(np.argmax(dp))],
        tle[0].strip(),
    )


print(
    f"{'obs':>9} {'object':>11} {'vert_prom':>9} {'dopp_prom':>9} {'dopp_off':>8}  reading"
)
rows = []
for path in sorted(glob.glob("/data/batch/*.h5")):
    oid, vprom, dprom, doff, obj = analyze(path)
    reading = (
        "UNCORRECTED (curve)"
        if dprom > vprom * 1.3
        else ("CORRECTED (vertical)" if vprom > dprom * 1.3 else "ambiguous")
    )
    print(f"{oid:>9} {obj:>11} {vprom:>9.1f} {dprom:>9.1f} {doff:>+8.0f}  {reading}")
    rows.append(reading)
print("\nper-obs readings:", rows)
