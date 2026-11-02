import json, numpy as np, csv
def load(ds, enc):
    ids = [json.loads(l)["id"] for l in open(f"data/{ds}/corpus.jsonl")]
    qids = [json.loads(l)["id"] for l in open(f"data/{ds}/queries.jsonl")]
    qrels = {}
    for r in csv.DictReader(open(f"data/{ds}/qrels_test.tsv"), delimiter="\t"):
        if int(r["score"]) > 0: qrels.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
    D = np.load(f"data/{ds}/docs_{enc}.npy"); Q = np.load(f"data/{ds}/queries_{enc}.npy")
    return ids, qids, qrels, D, Q
def topk(S, k=10):
    """Exact top k per row, ties broken by lower document index (deterministic)."""
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    out = np.empty_like(idx)
    for i in range(S.shape[0]):
        c = idx[i]; order = np.lexsort((c, -S[i, c])); out[i] = c[order]
    return out
def ndcg10(top, ids, qids, qrels):
    """Per query nDCG@10, linear gain (rel), log2(rank+1) discount, like trec_eval ndcg_cut."""
    disc = 1.0 / np.log2(np.arange(2, 12)); res = np.zeros(len(qids))
    for i, q in enumerate(qids):
        rel = qrels.get(q, {}); 
        dcg = sum(rel.get(ids[d], 0) * disc[r] for r, d in enumerate(top[i]))
        ideal = sorted(rel.values(), reverse=True)[:10]
        idcg = sum(g * disc[r] for r, g in enumerate(ideal))
        res[i] = dcg / idcg if idcg > 0 else 0.0
    return res
