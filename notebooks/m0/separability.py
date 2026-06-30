"""M0 separability test (grounded in the verified fact that the waterfall is Doppler-
corrected with the embedded TLE T0). The stored signal sits at s(t)=carrier+(true-T0)
Doppler. For each candidate c, the expected track if c were the emitter is
track_c(t)=df+(c_doppler-T0_doppler). Matched-filter that track against the stored
waterfall. The TRUE object (c=T0=Geoscan-2) -> flat track -> best fit; near-identical
siblings -> a sloped residual -> worse fit. Margin(true - best sibling) = separability."""

import h5py, json, numpy as np
from datetime import datetime, timedelta
from skyfield.api import load, wgs84, EarthSatellite

C = 299792.458
f = h5py.File("/data/good.h5", "r")
m = json.loads(f.attrs["metadata"])
f0 = float(m["frequency"])
loc = m["location"]
t0 = m["tle"].strip().splitlines()
wf = f["waterfall"]
data = wf["data"][:]
freqax = wf["frequency"][:].astype(float)
scale = wf["scale"][:].astype(float)
offset = wf["offset"][:].astype(float)
relt = wf["relative_time"][:].astype(float)
_st = wf.attrs["start_time"]
_st = _st.decode() if isinstance(_st, bytes) else _st
start = datetime.fromisoformat(_st.replace("Z", "+00:00"))
T, F = data.shape
dB = data.astype(np.float32) * scale[None, :] + offset[None, :]
ts = load.timescale(builtin=True)
st = wgs84.latlon(loc["latitude"], loc["longitude"], elevation_m=loc["altitude"])
tt = ts.from_datetimes([start + timedelta(seconds=float(r)) for r in relt])


def dopp(l1, l2):
    pos = (EarthSatellite(l1, l2, "x", ts) - st).at(tt)
    r = pos.position.km
    v = pos.velocity.km_per_s
    return -np.sum(r * v, axis=0) / np.linalg.norm(r, axis=0) / C * f0


T0 = dopp(t0[1], t0[2])  # correction TLE actually applied (Geoscan-2)
soup = json.load(open("/data/soup_tles.json"))
dfbin = float(freqax[1] - freqax[0])


def score(diff_track):
    # incoherent integration: shift each row by -(candidate differential) so that, IF the
    # candidate is the emitter, its signal aligns vertically; sum over time. True object ->
    # sharp peak; near-identical siblings (different phase) -> smeared -> no peak.
    sh = np.round(diff_track / dfbin).astype(int)
    integ = np.zeros(F)
    cnt = np.zeros(F)
    base = np.arange(F)
    for t in range(T):
        src = base + sh[t]
        v = (src >= 0) & (src < F)
        integ[v] += dB[t, src[v]]
        cnt[v] += 1
    integ = integ / np.maximum(cnt, 1)
    integ -= np.median(integ)
    return float(
        integ.max() / (np.std(integ) + 1e-9)
    )  # peak prominence (SNR) of the aligned signal


GEOSCANS = {64879, 64880, 64890, 64891, 64892, 64893}
res = []
for n, d in soup.items():
    track = dopp(d["tle1"], d["tle2"]) - T0  # differential vs the applied correction
    res.append(
        (
            score(track),
            int(n),
            d.get("tle0", "").strip(),
            d.get("days_off"),
            float(np.ptp(track)),
        )
    )
res.sort(reverse=True)
print(
    f"{'rk':>3} {'norad':>6} {'object':>20} {'score':>7} {'diff_span_Hz':>12} {'epoch_d':>7}"
)
for i, (s, n, nm, doff, span) in enumerate(res):
    tag = (
        "  <-- TRUE (Geoscan-2)"
        if n == 64890
        else ("  [sibling]" if n in GEOSCANS else "")
    )
    print(f"{i + 1:>3} {n:>6} {nm:>20} {s:>7.3f} {span:>12.0f} {doff:>7}{tag}")
true = next(r for r in res if r[1] == 64890)
sibs = [r for r in res if r[1] in GEOSCANS and r[1] != 64890]
bestsib = max(sibs)
print(f"\nTRUE Geoscan-2: score {true[0]:.3f}, rank {res.index(true) + 1}/{len(res)}")
print(
    f"best sibling  : norad {bestsib[1]} score {bestsib[0]:.3f}, differential span over pass {bestsib[4]:.0f} Hz"
)
print(f"SEPARABILITY MARGIN (true - best sibling): {true[0] - bestsib[0]:+.3f} dB")
print(f"sibling differential spans (Hz): {sorted(int(r[4]) for r in sibs)}")
