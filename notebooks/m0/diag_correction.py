"""Decide quantitatively whether the waterfall is Doppler-corrected: compare a matched
filter for a VERTICAL line (constant offset = corrected) vs the embedded-TLE DOPPLER CURVE
(raw), both on the same raw dB. Whichever 'excess above a random line' is larger reveals the
signal's shape. Also emits a high-contrast image."""

import h5py, json, numpy as np
from datetime import datetime, timedelta
from skyfield.api import load, wgs84, EarthSatellite
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = 299792.458
f = h5py.File("/data/good.h5", "r")
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
dBb = dB - np.median(dB, axis=0, keepdims=True)

ts = load.timescale(builtin=True)
st = wgs84.latlon(loc["latitude"], loc["longitude"], elevation_m=loc["altitude"])
tt = ts.from_datetimes([start + timedelta(seconds=float(r)) for r in relt])
sat = EarthSatellite(tle[1], tle[2], tle[0], ts)
pos = (sat - st).at(tt)
r = pos.position.km
v = pos.velocity.km_per_s
pred = -np.sum(r * v, axis=0) / np.linalg.norm(r, axis=0) / C * f0


def score(curve, mat, grid, win=3):
    out = np.empty(len(grid))
    for i, df in enumerate(grid):
        idx = np.clip(np.searchsorted(freqax, curve + df), 0, F - 1)
        acc = mat[np.arange(T), idx]
        for w in range(1, win + 1):
            acc = np.maximum(
                acc,
                np.maximum(
                    mat[np.arange(T), np.clip(idx + w, 0, F - 1)],
                    mat[np.arange(T), np.clip(idx - w, 0, F - 1)],
                ),
            )
        out[i] = acc.mean()
    return out


grid = np.arange(-12000, 12000, 40.0)
vline = score(np.zeros(T), dB, grid)  # vertical (corrected) hypothesis on raw
dcurv = score(pred, dB, grid)  # doppler-curve (raw) hypothesis on raw
v_exc = vline.max() - np.median(vline)
d_exc = dcurv.max() - np.median(dcurv)
print(
    "VERTICAL line  : peak %.3f at %+5.0f Hz | excess above median %.3f"
    % (vline.max(), grid[np.argmax(vline)], v_exc)
)
print(
    "DOPPLER curve  : peak %.3f at %+5.0f Hz | excess above median %.3f"
    % (dcurv.max(), grid[np.argmax(dcurv)], d_exc)
)
verdict = (
    "DOPPLER-CORRECTED (signal is a vertical line)"
    if v_exc > d_exc * 1.3
    else (
        "RAW/UNCORRECTED (signal follows the Doppler curve)"
        if d_exc > v_exc * 1.3
        else "AMBIGUOUS"
    )
)
print("VERDICT:", verdict)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ds = max(1, T // 1400)
for a, (mat, title, ov) in zip(
    ax,
    [
        (dB[::ds], "RAW waterfall (high contrast)", None),
        (dBb[::ds], "background-subtracted + predicted Doppler", pred),
    ],
):
    a.imshow(
        mat,
        aspect="auto",
        extent=[freqax[0], freqax[-1], relt[::ds][-1], relt[::ds][0]],
        cmap="inferno",
        vmin=np.percentile(mat, 90),
        vmax=np.percentile(mat, 99.9),
    )
    if ov is not None:
        a.plot(ov, relt, "c", lw=1.2, label="predicted Doppler")
        a.legend(loc="upper right")
    a.set_xlim(-12000, 12000)
    a.set_xlabel("offset from %.4f MHz (Hz)" % (f0 / 1e6))
    a.set_ylabel("time (s)")
    a.set_title(title)
fig.suptitle(
    "Geoscan-2 obs %d — correction diagnosis (%s)" % (m["observation_id"], verdict)
)
fig.tight_layout()
fig.savefig("/data/diag.png", dpi=100)
print("saved /data/diag.png")
