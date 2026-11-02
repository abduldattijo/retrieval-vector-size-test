"""Embed one BEIR dataset with one frozen encoder. Saves float32, L2 normalised vectors.
usage: python embed.py <dataset> <encoder short name: small|base>"""
import sys, json, numpy as np, torch
from sentence_transformers import SentenceTransformer
ds, enc = sys.argv[1], sys.argv[2]
MODEL = {"small": "BAAI/bge-small-en-v1.5", "base": "BAAI/bge-base-en-v1.5"}[enc]
QP = "Represent this sentence for searching relevant passages: "
docs = [json.loads(l) for l in open(f"data/{ds}/corpus.jsonl")]
qs = [json.loads(l) for l in open(f"data/{ds}/queries.jsonl")]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = SentenceTransformer(MODEL, device=dev); m.max_seq_length = 512
# sort by length so batches pad little; restore order after
texts = [(d["title"] + " " + d["text"]).strip() for d in docs]
order = np.argsort([len(t) for t in texts])
E = m.encode([texts[i] for i in order], batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
D = np.empty_like(E); D[order] = E
Q = m.encode([QP + q["text"] for q in qs], batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
np.save(f"data/{ds}/docs_{enc}.npy", D.astype(np.float32)); np.save(f"data/{ds}/queries_{enc}.npy", Q.astype(np.float32))
json.dump({"model": MODEL, "device": dev, "docs": len(docs), "queries": len(qs), "dim": int(D.shape[1]), "max_seq_length": 512, "query_prefix": QP}, open(f"data/{ds}/embed_{enc}.json", "w"))
print(ds, enc, D.shape, Q.shape, dev)
