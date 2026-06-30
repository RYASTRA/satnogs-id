"""M0 geometry validation: extract the measured Doppler track from a SatNOGS .h5
waterfall and overlay skyfield's predicted Doppler from the embedded per-obs TLE.
If they agree, the TEME->topocentric->range-rate geometry is trustworthy (the §5
non-negotiable gate) before any scoring. Runs in the python:3.14-slim container."""

import h5py, json, numpy as np
from datetime import datetime, timezone, timedelta
from skyfield.api import load, wgs84, EarthSatellite
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_KM_S = 299792.458

f = h5py.File("/data/sample.h5", "r")
m = json.loads(f.attrs["metadata"])
f0 = float(m["frequency"])  # Hz, nominal downlink
loc = m["location"]
tle = m["tle"].strip().splitlines()
name, l1, l2 = tle[0].strip(), tle[1].strip(), tle[2].strip()

wf = f["waterfall"]
data = wf["data"][:]  # (T, F) uint8
freqax = wf["frequency"][:].astype(float)  # (F,)  offset from f0 (units TBD)
scale = wf["scale"][:].astype(float)  # (F,)
offset = wf["offset"][:].astype(float)  # (F,)
relt = wf["relative_time"][:].astype(float)  # (T,) seconds since start
start_dt = datetime.fromisoformat(wf.attrs["start_time"].replace("Z", "+00:00"))
T, F = data.shape
print(
    f"waterfall {data.shape}  freqax[{freqax.min():.0f}..{freqax.max():.0f}] step={freqax[1] - freqax[0]:.2f}"
)
print(f"start={start_dt.isoformat()}  reltime span={relt[-1] - relt[0]:.1f}s")

# --- extract measured track: peak frequency bin per time row, dB-normalised per bin ---
dB = data.astype(np.float32) * scale[None, :] + offset[None, :]
# FIX 1 (structured noise): background subtraction. A fixed-frequency interference
# line is constant in time -> subtract each bin's time-median to remove it; the
# Doppler-sweeping satellite is NOT constant in any bin, so it survives.
dB = dB - np.median(dB, axis=0, keepdims=True)
peak = np.argmax(dB, axis=1)
peakp = dB[np.arange(T), peak]
base = np.median(dB, axis=1)
snr = peakp - base
keep = snr > (np.median(snr) + 2.0 * np.std(snr))  # signal-present rows
mt = relt[keep]  # rel seconds
mf = freqax[peak[keep]]  # measured offset (Hz)
print(
    f"signal-present rows: {keep.sum()}/{T}; measured offset range [{mf.min():.0f}..{mf.max():.0f}]"
)

# --- predicted Doppler from the embedded TLE (skyfield) ---
ts = load.timescale(builtin=True)
sat = EarthSatellite(l1, l2, name, ts)
st = wgs84.latlon(loc["latitude"], loc["longitude"], elevation_m=loc["altitude"])
dts = [start_dt + timedelta(seconds=float(rt)) for rt in relt]
tt = ts.from_datetimes(dts)
pos = (sat - st).at(tt)
r = pos.position.km  # (3, T) topocentric
v = pos.velocity.km_per_s  # (3, T)
rng = np.linalg.norm(r, axis=0)
rrate = np.sum(r * v, axis=0) / rng  # km/s radial (range rate)
pred_off_hz = -rrate / C_KM_S * f0  # Doppler offset, Hz
print(
    f"predicted Doppler offset range [{pred_off_hz.min():.0f}..{pred_off_hz.max():.0f}] Hz; min range {rng.min():.0f} km"
)

# --- overlay (test both Hz and kHz interpretations of the freq axis) ---
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
trel = relt
for a, (scl, lab) in zip(ax, [(1.0, "freq axis as Hz"), (1000.0, "freq axis as kHz")]):
    a.scatter(mt, mf * scl, s=6, c="tab:blue", label="measured peak (track)")
    a.plot(
        trel, pred_off_hz, c="tab:red", lw=2, label="predicted Doppler (embedded TLE)"
    )
    a.set_title(lab)
    a.set_xlabel("time since start (s)")
    a.set_ylabel("offset from %.4f MHz (Hz)" % (f0 / 1e6))
    a.legend()
    a.grid(alpha=0.3)
fig.suptitle(
    f"Geoscan-1 / NORAD 64880 obs {m['observation_id']} — measured vs predicted Doppler"
)
fig.tight_layout()
fig.savefig("/data/overlay.png", dpi=90)
print("saved /data/overlay.png")
