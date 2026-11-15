"""Datasets. MIAS: patient-level split (161 patients -> 113/24/24), diagnostic-region mask from Info.txt. Generic folder loader for other datasets.
Synthetic data (for smoke tests only) is available with --synthetic."""
import os, re, glob, numpy as np, torch, cv2
from torch.utils.data import Dataset

def _read(path):
    if path.lower().endswith('.dcm'):
        import pydicom; a = pydicom.dcmread(path).pixel_array.astype(np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5]); a = np.clip((a-lo)/(hi-lo+1e-8), 0, 1)*255
        return a.astype(np.uint8)
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a.ndim == 3: a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if a.dtype != np.uint8: a = (a.astype(np.float32)/a.max()*255).astype(np.uint8)
    return a
def _prep(a, size=256): return cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)

def mias_records(root):
    """returns list of dict(name, patient, path, abn=[(x,y,r)])  (x,y,r in the 1024 x 1024 frame; y measured from the bottom edge)"""
    info = {}
    p = os.path.join(root, 'Info.txt')
    if os.path.exists(p):
        for ln in open(p):
            f = ln.split()
            if len(f) >= 7 and f[0].startswith('mdb') and f[3] in ('B', 'M'):
                try: info.setdefault(f[0], []).append((float(f[4]), float(f[5]), float(f[6])))
                except ValueError: pass
    recs = []
    for path in sorted(glob.glob(os.path.join(root, 'mdb*.pgm'))+glob.glob(os.path.join(root, 'mdb*.png'))):
        name = os.path.splitext(os.path.basename(path))[0]; n = int(re.findall(r'\d+', name)[0])
        recs.append(dict(name=name, patient=(n+1)//2, path=path, abn=info.get(name, [])))   # mdb001/mdb002 = left/right breast of patient 1
    return recs

def patient_split(recs, seed=0, frac=(0.70, 0.15, 0.15)):
    pats = sorted({r['patient'] for r in recs}); rng = np.random.default_rng(seed); rng.shuffle(pats)
    n = len(pats); a = int(round(frac[0]*n)); b = a+int(round(frac[1]*n))
    sets = dict(train=set(pats[:a]), val=set(pats[a:b]), test=set(pats[b:]))
    return {k: [r for r in recs if r['patient'] in v] for k, v in sets.items()}

class ImageSet(Dataset):
    def __init__(self, recs, size=256, train=False):
        self.items, self.train = [], train
        for r in recs:
            img = _prep(_read(r['path']), size); m = np.zeros((size, size), np.uint8); s = size/1024
            for (x, y, rad) in r.get('abn', []): cv2.circle(m, (int(x*s), int((1024-y)*s)), max(2, int(rad*s)), 1, -1)
            self.items.append((img, m))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        img, m = self.items[i]
        if self.train and np.random.rand() < .5: img, m = img[:, ::-1].copy(), m[:, ::-1].copy()
        return torch.from_numpy(img).float().unsqueeze(0)/255, torch.from_numpy(m).float().unsqueeze(0)

def folder_records(root, exts=('png', 'jpg', 'jpeg', 'tif', 'tiff', 'dcm'), limit=None):
    files = sorted(sum([glob.glob(os.path.join(root, '**', '*.'+e), recursive=True) for e in exts], []))
    files = files[:limit] if limit else files
    return [dict(name=os.path.basename(f), patient=hash(os.path.dirname(f)+os.path.basename(f)) % 10**9, path=f, abn=[]) for f in files]

class SyntheticSet(Dataset):                 # smoke test only: smooth random textures with blobs
    def __init__(self, n=64, size=256, seed=0):
        rng = np.random.default_rng(seed); self.items = []
        for _ in range(n):
            a = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), rng.uniform(2, 8)); a = (a-a.min())/(a.max()-a.min())
            m = np.zeros((size, size), np.uint8); cv2.circle(m, tuple(rng.integers(40, size-40, 2).tolist()), 20, 1, -1)
            a = np.clip(a*0.8+0.2*m, 0, 1); self.items.append(((a*255).astype(np.uint8), m))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): img, m = self.items[i]; return torch.from_numpy(img).float().unsqueeze(0)/255, torch.from_numpy(m).float().unsqueeze(0)
