"""Visual confirmation: is the SatNOGS waterfall Doppler-corrected? Show a strong,
vetted-good pass WITHOUT background subtraction and overlay the embedded-TLE predicted
Doppler curve. Vertical signal line + red curve sweeping away from it => corrected."""

import h5py, json, numpy as np
from datetime import timedelta
from skyfield.api import load, wgs84, EarthSatellite
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from satnogs_id.shared.waterfall import load_waterfall

C = 299792.458
PATH = "/data/good.h5"
with h5py.File(PATH, "r") as _h:
    _meta = _h.attrs["metadata"]
assert isinstance(_meta, (str, bytes, bytearray))
obs_id = json.loads(_meta)["observation_id"]

wf = load_waterfall(PATH)
f0 = wf.f0_hz
freqax = wf.freqax_hz
relt = wf.relative_time_s
dB = wf.db  # per-bin normalized; NO time-median subtraction
T, F = dB.shape
print("obs", obs_id, "|", wf.tle[0], "| pass", round(relt[-1] - relt[0]), "s")

ts = load.timescale(builtin=True)
st = wgs84.latlon(wf.station.lat, wf.station.lon, elevation_m=wf.station.alt_m)
tt = ts.from_datetimes([wf.start + timedelta(seconds=float(rt)) for rt in relt])
sat = EarthSatellite(wf.tle[1], wf.tle[2], wf.tle[0], ts)
pos = (sat - st).at(tt)
r = pos.position.km
v = pos.velocity.km_per_s
assert isinstance(r, np.ndarray) and isinstance(v, np.ndarray)
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
fig.suptitle("Geoscan-2 obs %d — 85deg pass, vetted with-signal" % obs_id)
fig.tight_layout()
fig.savefig("/data/good_viz.png", dpi=95)
print("saved /data/good_viz.png")
