import sys; sys.path.insert(0, '.')
import torch, torch.nn as nn, numpy as np, onnxruntime as ort, io, warnings; warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)
from models import *
from export_onnx import load_fp32
sysm,meta=load_fp32('runs/smoke'); enc=sysm.enc.eval(); torch.manual_seed(0)
img=torch.rand(1,1,256,256); m=(torch.rand(1,511)>.5).float(); rk=torch.sign(torch.randn(1,1,128,128))
def chk(name, mod, *inp):
    f=io.BytesIO(); torch.onnx.export(mod, inp, f, opset_version=20, dynamo=False); s=ort.InferenceSession(f.getvalue())
    y=s.run(None,{i.name:x.numpy() for i,x in zip(s.get_inputs(),inp)})[0]; mod.eval(); r=mod(*inp).numpy()
    print(f'RESULT {name:20s} max_abs_diff {np.abs(y-r).max():.2e}', flush=True)
from export_onnx import EncW
chk('enc.msg', enc.msg, m)
chk('stem', enc.stem, torch.rand(1,7,128,128))
chk('stem+blocks', nn.Sequential(enc.stem, enc.blocks), torch.rand(1,7,128,128))
chk('enc full', EncW(enc), img, m, rk)
print('RESULT training flag', enc.training, any(mm.training for mm in enc.modules()))
