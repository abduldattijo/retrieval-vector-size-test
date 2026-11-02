import numpy as np
SAMPLE_N, SEED = 2000, 0
def fit_sample(D, seed=SEED, n=SAMPLE_N):
    rng = np.random.default_rng(seed); return D[rng.choice(len(D), size=min(n, len(D)), replace=False)]
def fit_pca(sample):
    mu = sample.mean(0); _, _, Vt = np.linalg.svd(sample - mu, full_matrices=False); return mu, Vt  # rows = components
def project(X, mu, Vt, k):
    Z = (X - mu) @ Vt[:k].T; return Z / np.linalg.norm(Z, axis=1, keepdims=True)
def truncate(X, k):
    Z = X[:, :k]; return Z / np.linalg.norm(Z, axis=1, keepdims=True)
def quantize_docs(Dm, sample, kind):
    """Documents are stored at the lower precision, queries stay FP32. Returns (scored float32 matrix, stored bytes, extra bytes)."""
    if kind == "fp32": return Dm.astype(np.float32), Dm.astype(np.float32).nbytes, 0
    if kind == "fp16": a = Dm.astype(np.float16); return a.astype(np.float32), a.nbytes, 0
    if kind == "int8":
        scale = np.abs(sample).max(0) + 1e-12; q = np.clip(np.round(Dm / scale * 127), -127, 127).astype(np.int8)
        return (q.astype(np.float32) / 127 * scale), q.nbytes, scale.astype(np.float32).nbytes
    if kind == "bit1":
        bits = (Dm > 0); packed = np.packbits(bits, axis=1)
        return np.where(bits, 1.0, -1.0).astype(np.float32), packed.nbytes, 0
