"""Secure payload chain (Section 3.3): metadata -> SHA-256 -> AES-128-CBC (or HMAC variant) -> BCH(511,385) -> key permutation;
plus the key-derived reference pattern R_K (AES-CTR keystream)."""
import os, hmac, hashlib, numpy as np, galois
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

N, K_BCH, T = 511, 385, 14
_bch = galois.BCH(N, K_BCH)           # t = 14
assert _bch.t == T, _bch.t

def _kdf(key, label): return hmac.new(key, label, hashlib.sha256).digest()[:16]
def derive(key):
    return dict(enc=_kdf(key, b'enc'), mac=_kdf(key, b'mac'), perm=_kdf(key, b'perm'), rk=_kdf(key, b'rk'))
def bits(b): return np.unpackbits(np.frombuffer(b, np.uint8))
def unbits(a): return np.packbits(a.astype(np.uint8)).tobytes()

def permutation(kperm):
    ks = Cipher(algorithms.AES(kperm), modes.CTR(b'\0'*16)).encryptor().update(b'\0'*N*4)
    return np.argsort(np.frombuffer(ks, np.uint32)[:N])
def reference_pattern(krk, iv, size=128):
    ks = Cipher(algorithms.AES(krk), modes.CTR(iv)).encryptor().update(b'\0'*(size*size//8))
    return bits(ks).reshape(size, size).astype(np.float32)*2-1

def make_payload_bits(key, metadata, iv, variant='cbc'):
    kd = derive(key); d = hashlib.sha256(metadata).digest()
    if variant == 'cbc':
        enc = Cipher(algorithms.AES(kd['enc']), modes.CBC(iv)).encryptor(); body = enc.update(d)+enc.finalize()
    else:
        body = hmac.new(kd['mac'], iv+metadata, hashlib.sha256).digest()
    return np.concatenate([bits(iv+body), [0]]).astype(np.uint8)         # 384 bits + 1 pad = 385

def encode(key, metadata, iv, variant='cbc'):
    msg = make_payload_bits(key, metadata, iv, variant)
    cw = np.asarray(_bch.encode(galois.GF2(msg)), dtype=np.uint8)
    return cw[permutation(derive(key)['perm'])]

def encode_batch(msgs):
    """vectorised BCH encode of (n,385) message array -> (n,511)"""
    return np.asarray(_bch.encode(galois.GF2(msgs.astype(np.uint8))), dtype=np.uint8)

def decode_bits(received, key):
    """received: 511 hard bits. returns (ok, iv, body) or (False,None,None) when BCH fails"""
    perm = permutation(derive(key)['perm']); inv = np.empty_like(perm); inv[perm] = np.arange(N)
    cw = received.astype(np.uint8)[inv]
    msg, nerr = _bch.decode(galois.GF2(cw), errors=True)
    if int(nerr) < 0: return False, None, None
    b = unbits(np.asarray(msg, np.uint8)[:384]); return True, b[:16], b[16:]

def verify(received, key, metadata, variant='cbc'):
    ok, iv, body = decode_bits(received, key)
    if not ok: return 'unreadable'
    kd = derive(key)
    if variant == 'cbc':
        dec = Cipher(algorithms.AES(kd['enc']), modes.CBC(iv)).decryptor(); d = dec.update(body)+dec.finalize()
        return 'authentic' if d == hashlib.sha256(metadata).digest() else 'metadata mismatch'
    return 'authentic' if hmac.compare_digest(body, hmac.new(kd['mac'], iv+metadata, hashlib.sha256).digest()) else 'metadata mismatch'

class PayloadFactory:
    """Fresh key, IV and metadata for every sample (training and evaluation)."""
    def __init__(self, variant='cbc', seed=0, coded=True):
        self.variant, self.rng, self.coded = variant, np.random.default_rng(seed), coded
    def batch(self, n):
        ms, rks, aux = [], [], []
        for _ in range(n):
            key = self.rng.bytes(16); iv = self.rng.bytes(16); md = b'PID%08d|DEV%04d|T%010d' % (int(self.rng.integers(0, 10**8)), int(self.rng.integers(0, 9999)), int(self.rng.integers(0, 10**9)))
            ms.append(make_payload_bits(key, md, iv, self.variant)); rks.append(reference_pattern(derive(key)['rk'], iv))
            aux.append((key, md, iv))
        if not self.coded:                      # ablation (ii): no BCH; 385 payload bits + random filler
            out = np.concatenate([np.stack(ms), self.rng.integers(0, 2, (n, N-K_BCH))], 1)
        else:
            cw = encode_batch(np.stack(ms))
            out = np.stack([cw[i][permutation(derive(aux[i][0])['perm'])] for i in range(n)])
        return out.astype(np.float32), np.stack(rks)[:, None], aux

if __name__ == '__main__':
    import time
    f = PayloadFactory(); t = time.time(); m, rk, aux = f.batch(16); print('batch(16) %.2fs' % (time.time()-t), m.shape, rk.shape)
    key, md, iv = aux[0]
    for var in ('cbc', 'hmac'):
        cw = encode(key, md, iv, var)
        print(var, 'clean ->', verify(cw, key, md, var), '| wrong key ->', verify(cw, os.urandom(16), md, var), '| other metadata ->', verify(cw, key, b'other', var))
        r = cw.copy(); idx = np.random.default_rng(1).choice(N, 14, replace=False); r[idx] ^= 1; print('   14 errors ->', verify(r, key, md, var), end=' | ')
        r = cw.copy(); idx = np.random.default_rng(1).choice(N, 40, replace=False); r[idx] ^= 1; print('40 errors ->', verify(r, key, md, var))
