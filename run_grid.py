"""Full grid. Frozen rule from PLAN.md: mean loss <= 0.01 nDCG@10, upper end of 95% paired bootstrap range of loss <= 0.02,
and at most 5% of queries losing >= 0.2 against their own full vector score."""
import json, numpy as np
from lib import *; from compress import *
DATASETS = ["scifact", "nfcorpus", "fiqa"]; ENC = {"small": [256, 128, 64, 32, 16], "base": [512, 256, 128, 64, 32, 16]}
PRECS = ["fp32", "fp16", "int8", "bit1"]; NB, BSEED = 10000, 12345
out = open("results/grid.jsonl", "w"); perq = {}
def passes(mean_loss, hi, share): return bool(mean_loss <= 0.01 and hi <= 0.02 and share <= 0.05)
for ds in DATASETS:
    for enc, dims in ENC.items():
        ids, qids, qrels, D, Q = load(ds, enc); nq = len(qids); d_full = D.shape[1]
        rng = np.random.default_rng(BSEED); idx = rng.integers(0, nq, size=(NB, nq))
        base = ndcg10(topk(Q @ D.T), ids, qids, qrels); perq[f"{ds}_{enc}_full"] = base
        def record(name, arm, dim, prec, Dm, Qm, doc_bytes, extra_bytes, fit_seed):
            n = ndcg10(topk(Qm @ Dm.T), ids, qids, qrels); loss = base - n
            boots = loss[idx].mean(1); lo, hi = np.percentile(boots, [2.5, 97.5]); share = float((loss >= 0.2).mean())
            row = dict(dataset=ds, encoder=enc, arm=arm, dim=dim, precision=prec, fit_seed=fit_seed, ndcg10=float(n.mean()), loss=float(loss.mean()),
                       loss_lo=float(lo), loss_hi=float(hi), share_lose_02=share, doc_bytes=int(doc_bytes), extra_bytes=int(extra_bytes),
                       full_doc_bytes=int(D.nbytes), n_docs=len(ids), n_queries=nq, passes_rule=passes(loss.mean(), hi, share))
            out.write(json.dumps(row) + "\n"); out.flush(); perq[f"{ds}_{enc}_{name}"] = n
        samp = fit_sample(D); mu, Vt = fit_pca(samp)
        def proj(X, k):
            Z = X @ Vt[:k].T; return Z / np.linalg.norm(Z, axis=1, keepdims=True)   # no centring (decided from pilot)
        # B: precision at full dimensions
        for p in PRECS:
            Dq, b, x = quantize_docs(D, samp, p); record(f"full_{p}", "B_precision", d_full, p, Dq, Q, b, x, 0)
        # A and C: dimensions, then dimensions with lower precision
        for k in dims:
            Dk, Qk = proj(D, k), proj(Q, k); sk = proj(samp, k); pca_bytes = k * d_full * 4
            for p in PRECS:
                Dq, b, x = quantize_docs(Dk, sk, p); record(f"pca{k}_{p}", "A_dims" if p == "fp32" else "C_both", k, p, Dq, Qk, b, x + pca_bytes, 0)
            Tk, TQ = truncate(D, k), truncate(Q, k); record(f"trunc{k}_fp32", "A_truncation", k, "fp32", Tk, TQ, Tk.astype(np.float32).nbytes, 0, 0)
            for s in (1, 2):   # does a different fitting sample change the answer
                m2, V2 = fit_pca(fit_sample(D, seed=s)); Z = (D @ V2[:k].T); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
                ZQ = (Q @ V2[:k].T); ZQ /= np.linalg.norm(ZQ, axis=1, keepdims=True); record(f"pca{k}_fp32_seed{s}", "A_dims_seed", k, "fp32", Z, ZQ, Z.astype(np.float32).nbytes, pca_bytes, s)
        # the trap: centred PCA at full dimensions
        Dc = (D - mu) @ Vt.T; Dc /= np.linalg.norm(Dc, axis=1, keepdims=True); Qc = (Q - mu) @ Vt.T; Qc /= np.linalg.norm(Qc, axis=1, keepdims=True)
        record("centred_full", "trap_centred", d_full, "fp32", Dc, Qc, Dc.astype(np.float32).nbytes, d_full * d_full * 4, 0)
        print(ds, enc, "done", flush=True)
np.savez_compressed("results/per_query_ndcg.npz", **perq); print("GRID_DONE")
