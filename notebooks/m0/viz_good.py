"""Visual confirmation: is the SatNOGS waterfall Doppler-corrected? Show a strong,
vetted-good pass WITHOUT background subtraction and overlay the embedded-TLE predicted
Doppler curve. Vertical signal line + red curve sweeping away from it => corrected."""

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
print("obs", m["observation_id"], "|", tle[0], "| pass", round(relt[-1] - relt[0]), "s")
dB = (
    data.astype(np.float32) * scale[None, :] + offset[None, :]
)  # per-bin normalized; NO time-median subtraction

ts = load.timescale(builtin=True)
st = wgs84.latlon(loc["latitude"], loc["longitude"], elevation_m=loc["altitude"])
tt = ts.from_datetimes([start + timedelta(seconds=float(r)) for r in relt])
sat = EarthSatellite(tle[1], tle[2], tle[0], ts)
pos = (sat - st).at(tt)
r = pos.position.km
v = pos.velocity.km_per_s
pred = -np.sum(r * v, axis=0) / np.linalg.norm(r, axis=0) / C * f0
print(
    "predicted Doppler [%.0f..%.0f] Hz; min range %.0f km"
    % (pred.min(), pred.max(), np.linalg.norm(r, axis=0).min())
)

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ds = max(1, T // 1400)
img = dB[::ds]
ext = [freqax[0], freqax[-1], relt[::ds][-1], relt[::ds][0]]
for a, (title, overlay) in zip(
    ax,
    [
        ("raw waterfall (as stored)", False),
        ("raw waterfall + embedded-TLE predicted Doppler", True),
    ],
):
    a.imshow(
        img,
        aspect="auto",
        extent=ext,
        cmap="viridis",
        vmin=np.percentile(img, 70),
        vmax=np.percentile(img, 99.8),
    )
    if overlay:
        a.plot(pred, relt, "r", lw=1.2, label="predicted Doppler (embedded TLE)")
        a.legend(loc="upper right")
    a.set_xlim(-15000, 15000)
    a.set_xlabel("offset from %.4f MHz (Hz)" % (f0 / 1e6))
    a.set_ylabel("time (s)")
    a.set_title(title)
fig.suptitle("Geoscan-2 obs %d — 85deg pass, vetted with-signal" % m["observation_id"])
fig.tight_layout()
fig.savefig("/data/good_viz.png", dpi=95)
print("saved /data/good_viz.png")
