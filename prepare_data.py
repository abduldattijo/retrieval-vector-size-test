"""Download BEIR SciFact, NFCorpus, FiQA test splits and save plain files. Prints counts."""
import json, os, pandas as pd
from huggingface_hub import hf_hub_download
OUT = "data"
for name in ["scifact", "nfcorpus", "fiqa"]:
    d = f"{OUT}/{name}"; os.makedirs(d, exist_ok=True)
    corpus = pd.read_parquet(hf_hub_download(f"BeIR/{name}", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset"))
    queries = pd.read_parquet(hf_hub_download(f"BeIR/{name}", "queries/queries-00000-of-00001.parquet", repo_type="dataset"))
    qrels = pd.read_csv(hf_hub_download(f"BeIR/{name}-qrels", "test.tsv", repo_type="dataset"), sep="\t")
    qrels.columns = ["query-id", "corpus-id", "score"]
    qrels["query-id"] = qrels["query-id"].astype(str); qrels["corpus-id"] = qrels["corpus-id"].astype(str)
    test_q = set(qrels["query-id"])
    queries["_id"] = queries["_id"].astype(str)
    queries = queries[queries["_id"].isin(test_q)]
    with open(f"{d}/corpus.jsonl", "w") as f:
        for r in corpus.itertuples():
            f.write(json.dumps({"id": str(r._1), "title": r.title, "text": r.text}) + "\n")
    with open(f"{d}/queries.jsonl", "w") as f:
        for r in queries.itertuples():
            f.write(json.dumps({"id": r._1, "text": r.text}) + "\n")
    qrels.to_csv(f"{d}/qrels_test.tsv", sep="\t", index=False)
    print(name, "docs", len(corpus), "test queries with labels", len(queries), "qrel rows", len(qrels), "graded levels", sorted(qrels["score"].unique()))
