"""M0 matched-filter detector: instead of extracting a noisy per-row track, integrate
the (background-subtracted) waterfall power ALONG a candidate's predicted Doppler curve,
searching the unknown carrier offset. Demonstrates the two layered noise fixes:
  FIX 1 background subtraction  -> removes fixed-frequency interference lines
  FIX 2 matched filter          -> robust to broadband noise (integrate along the curve)
Compares the Doppler-curve score against a flat zero-Doppler line: if the signal really
follows the Doppler shape, the curve wins (and that gap is the basis of separability)."""

import h5py, json, numpy as np
from datetime import datetime, timedelta
from skyfield.api import load, wgs84, EarthSatellite
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = 299792.458

f = h5py.File("/data/sample.h5", "r")
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
start = datetime.fromisoformat(wf.attrs["start_time"].replace("Z", "+00:00"))
T, F = data.shape
dB = data.astype(np.float32) * scale[None, :] + offset[None, :]
dB = dB - np.median(dB, axis=0, keepdims=True)  # FIX 1: background subtraction

ts = load.timescale(builtin=True)
st = wgs84.latlon(loc["latitude"], loc["longitude"], elevation_m=loc["altitude"])
tt = ts.from_datetimes([start + timedelta(seconds=float(r)) for r in relt])


def predicted_offset(l1, l2, name):
    sat = EarthSatellite(l1, l2, name, ts)
    pos = (sat - st).at(tt)
    r = pos.position.km
    v = pos.velocity.km_per_s
    return -np.sum(r * v, axis=0) / np.linalg.norm(r, axis=0) / C * f0  # Hz


pred = predicted_offset(tle[1], tle[2], tle[0])

# FIX 2: matched filter. For a carrier offset df, take the max power in a small window
# around the curve at each time, average over the pass.
df_grid = np.arange(-10000, 10000, 50.0)


def mf(curve, win=2):
    out = np.empty(len(df_grid))
    for i, df in enumerate(df_grid):
        idx = np.clip(np.searchsorted(freqax, curve + df), 0, F - 1)
        acc = dB[np.arange(T), idx]
        for w in range(1, win + 1):
            acc = np.maximum(
                acc,
                np.maximum(
                    dB[np.arange(T), np.clip(idx + w, 0, F - 1)],
                    dB[np.arange(T), np.clip(idx - w, 0, F - 1)],
                ),
            )
        out[i] = acc.mean()
    return out


s_dopp = mf(pred)  # Geoscan-1's real Doppler curve
s_flat = mf(np.zeros(T))  # a flat zero-Doppler line (null hypothesis)
best = df_grid[np.argmax(s_dopp)]
print(
    f"Doppler-curve matched-filter peak: {s_dopp.max():.3f} dB at carrier offset {best:+.0f} Hz"
)
print(f"Flat zero-Doppler line peak:       {s_flat.max():.3f} dB")
print(
    f"--> Doppler shape advantage: {s_dopp.max() - s_flat.max():+.3f} dB  "
    f"({'signal follows the Doppler curve (uncorrected)' if s_dopp.max() > s_flat.max() + 0.2 else 'signal ~flat -> waterfall is Doppler-CORRECTED'})"
)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
ax[0].plot(df_grid, s_dopp, label="Geoscan-1 Doppler curve")
ax[0].plot(df_grid, s_flat, "--", label="flat zero-Doppler line")
ax[0].axvline(best, color="k", lw=0.7)
ax[0].set_xlabel("carrier offset (Hz)")
ax[0].set_ylabel("mean power along curve (dB)")
ax[0].set_title("matched-filter score vs carrier offset")
ax[0].legend()
ax[0].grid(alpha=0.3)
ds = max(1, T // 1200)
img = dB[::ds]
ax[1].imshow(
    img,
    aspect="auto",
    extent=[freqax[0], freqax[-1], relt[::ds][-1], relt[::ds][0]],
    cmap="viridis",
    vmin=np.percentile(img, 60),
    vmax=np.percentile(img, 99.7),
)
ax[1].plot(pred + best, relt, "r", lw=1.2, label="best-fit Geoscan-1 curve")
ax[1].set_xlim(-15000, 15000)
ax[1].set_xlabel("offset from %.4f MHz (Hz)" % (f0 / 1e6))
ax[1].set_ylabel("time (s)")
ax[1].set_title("cleaned waterfall + matched curve")
ax[1].legend(loc="upper right")
fig.tight_layout()
fig.savefig("/data/matched.png", dpi=90)
print("saved /data/matched.png")
