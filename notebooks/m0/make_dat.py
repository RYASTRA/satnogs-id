"""Build strf inputs from a SatNOGS .h5 (Geoscan-2, obs 12945145), following the
authoritative satnogs_waterfall_tabulation_helper.py recipe:
  1. extract the near-vertical signal track from the (Doppler-CORRECTED) waterfall,
  2. UN-correct it with the obs TLE: freq_recv = f_center + offset - f_center*range_rate/c,
  3. write rffit's .dat (MJD, freq_recv, weight, site), the candidate catalog, and the site.
Then rffit's identify ('i') matches the recovered Doppler curve against the catalog."""

import json, os, numpy as np
from datetime import timedelta

from satnogs_id.shared import geometry
from satnogs_id.shared.waterfall import load_waterfall

SITE = 9001
wf = load_waterfall("/data/good.h5")
f0 = wf.f0_hz
freqax = wf.freqax_hz
relt = wf.relative_time_s
dB = wf.db
T, F = dB.shape

# 1. extract near-vertical signal track: per-row peak, keep high-SNR rows near the carrier column
peak = np.argmax(dB, axis=1)
peakp = dB[np.arange(T), peak]
base = np.median(dB, axis=1)
snr = peakp - base
hi = snr >= np.percentile(snr, 85)
carrier = float(np.median(freqax[peak[hi]]))
keep = hi & (np.abs(freqax[peak] - carrier) < 6000)
idx = np.where(keep)[0]
print(f"extracted {len(idx)}/{T} track points; carrier offset ~{carrier:.0f} Hz")

# 2. un-correct with the obs TLE via the shared geometry helpers, exactly per the recipe
times = [wf.start + timedelta(seconds=float(relt[i])) for i in idx]
rr = geometry.range_rate_km_s(
    wf.tle[1], wf.tle[2], wf.station, times
)  # km/s, + = receding
recv = geometry.uncorrect(f0, freqax[peak[idx]], rr)
rows = sorted((geometry.mjd(times[k]), float(recv[k])) for k in range(len(idx)))
os.makedirs("/data/strf", exist_ok=True)
with open("/data/strf/geoscan.dat", "w") as g:
    for mj, fr in rows:
        g.write(f"{mj:.6f}\t{fr:.2f}\t1.0\t{SITE}\n")
fr_lo, fr_hi = min(r[1] for r in rows), max(r[1] for r in rows)
print(
    f"wrote geoscan.dat: {len(rows)} pts, recv-freq span {(fr_hi - fr_lo) / 1e3:.1f} kHz around {f0 / 1e6:.4f} MHz "
    f"(expect ~Doppler swing if extraction good)"
)

# 3. candidate catalog (3LE) + site
soup = json.load(open("/data/soup_tles.json"))
with open("/data/strf/soup.tle", "w") as g:
    for n, d in soup.items():
        g.write(f"{d.get('tle0') or '0 OBJECT'}\n{d['tle1']}\n{d['tle2']}\n")
print(f"wrote soup.tle: {len(soup)} candidates (true = 64890 / Geoscan-2)")
st = wf.station
with open("/opt/strf/data/sites.txt", "a") as g:
    g.write(f"{SITE} GS {st.lat:.4f} {st.lon:.4f} {int(st.alt_m)} GeoscanStation\n")
print(f"appended site {SITE} ({st.lat},{st.lon}) to sites.txt")
