import json, numpy as np
from lib import load
P=np.load("results/per_query_ndcg.npz")
def qinfo(ds):
    qs={json.loads(l)["id"]:json.loads(l)["text"] for l in open(f"data/{ds}/corpus.jsonl")} if False else None
    q=[json.loads(l) for l in open(f"data/{ds}/queries.jsonl")]
    _,qids,qrels,_,_=load(ds,"small")
    return [x["text"] for x in q], np.array([len(x["text"].split()) for x in q]), np.array([len(qrels[i]) for i in qids]), qids
SET=[("pca128_fp32","PCA 128 dims"),("pca256_fp32","PCA 256 dims"),("full_bit1","1 bit, full size"),("full_int8","int8, full size")]
out={}
for enc in ("small","base"):
  print(f"\n===== bge-{enc} =====")
  for ds in ("scifact","nfcorpus","fiqa"):
    texts,qlen,nrel,qids=qinfo(ds); base=P[f"{ds}_{enc}_full"]
    for key,label in SET:
        n=P[f"{ds}_{enc}_{key}"]; d=base-n; tot=d[d>0].sum()
        lose=(d>=0.2).mean(); gain=(d<=-0.2).mean(); mean=d.mean()
        order=np.argsort(-d); top10=d[order[:max(1,len(d)//10)]].sum()
        # who loses: short vs long queries, one vs many relevant docs, hard vs easy at full size
        med=np.median(qlen); sh=d[qlen<=med].mean(); lo=d[qlen>med].mean()
        one=d[nrel==1].mean() if (nrel==1).any() else float("nan"); many=d[nrel>1].mean() if (nrel>1).any() else float("nan")
        easy=d[base>=0.8].mean(); hard=d[base<0.4].mean()
        unchanged=(np.abs(d)<1e-9).mean()
        print(f"{ds:9s}{label:17s} mean {mean*100:5.1f} | lose>=0.2 {lose*100:4.1f}% gain>=0.2 {gain*100:4.1f}% unchanged {unchanged*100:4.1f}% | worst10% of queries carry {top10/ max(d.sum(),1e-9)*100:5.1f}% of net loss | short(<= {med:.0f}w) {sh*100:5.1f} long {lo*100:5.1f} | 1rel {one*100:5.1f} many {many*100:5.1f} | easy(full>=.8) {easy*100:5.1f} hard(full<.4) {hard*100:5.1f}")
