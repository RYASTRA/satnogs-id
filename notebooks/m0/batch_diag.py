"""Empirical correction test across MANY strong passes from DIFFERENT stations.
For each .h5: matched-filter profile vs carrier offset for two hypotheses --
  vertical line (constant offset)  -> Doppler-CORRECTED
  embedded-TLE Doppler curve       -> RAW/UNCORRECTED
Score per offset = mean of the top 25% along-line power (robust to intermittent
beacons). Prominence = (peak - median)/std of that profile. The hypothesis with the
sharper, more prominent peak reveals the signal's shape. No conclusion is asserted
here -- the per-obs table is what we read."""

import h5py, json, numpy as np, glob
from datetime import timedelta

from satnogs_id.shared import geometry
from satnogs_id.shared.waterfall import load_waterfall


def topmean(x, frac=0.25):
    k = max(1, int(len(x) * frac))
    return float(np.mean(np.sort(x)[-k:]))


def observation_id(path):
    """Read the SatNOGS observation id from an artifact's metadata attribute (typed)."""
    with h5py.File(path, "r") as f:
        meta = f.attrs["metadata"]
    assert isinstance(meta, (str, bytes, bytearray))
    return json.loads(meta)["observation_id"]


def analyze(path):
    wf = load_waterfall(path)
    f0 = wf.f0_hz
    freqax = wf.freqax_hz
    relt = wf.relative_time_s
    dB = wf.db
    T, F = dB.shape
    times = [wf.start + timedelta(seconds=float(r)) for r in relt]
    rr = geometry.range_rate_km_s(wf.tle[1], wf.tle[2], wf.station, times)
    pred = geometry.doppler_offset_hz(f0, rr)
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
        observation_id(path),
        prom(vp),
        prom(dp),
        grid[int(np.argmax(dp))],
        wf.tle[0].strip(),
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
