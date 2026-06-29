"""Host-side: for each batch .h5, build a candidate catalog (the 2025-155 soup,
NORAD 64876-64895) with each candidate's TLE epoch-matched to THAT observation's date,
pulled from SatNOGS per-obs TLEs. Writes /scratch/strf/soup_<obsid>.tle."""
import json, os, glob, urllib.request
from datetime import datetime
SC = "/private/tmp/claude-501/-Users-ryan-GitHub-satnogs-id/e9fc3766-e6dd-4352-9d46-489818e4c3a6/scratchpad"
def getj(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={'Accept':'application/json'}), timeout=30))
soup = list(range(64876, 64896))
cand_obs = {n: getj(f'https://network.satnogs.org/api/observations/?norad_cat_id={n}&format=json') for n in soup}
os.makedirs(SC+'/strf', exist_ok=True)
for p in sorted(glob.glob(SC+'/batch/*.h5')):
    oid = os.path.basename(p).split('_')[0][3:]
    tdate = datetime.fromisoformat(getj(f'https://network.satnogs.org/api/observations/?id={oid}&format=json')[0]['start'][:10])
    lines = []; n_match = 0
    for n in soup:
        best = None; bd = 1e9
        for o in cand_obs[n]:
            if not o.get('tle1'): continue
            dd = abs((datetime.fromisoformat(o['start'][:10]) - tdate).days)
            if dd < bd: bd = dd; best = o
        if best:
            lines += [(best.get('tle0') or '0 OBJECT').strip(), best['tle1'], best['tle2']]; n_match += 1
    open(f'{SC}/strf/soup_{oid}.tle', 'w').write('\n'.join(lines)+'\n')
    print(f'obs {oid} date {tdate.date()}: soup_{oid}.tle ({n_match} candidates)')
