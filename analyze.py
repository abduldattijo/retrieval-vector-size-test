import json, numpy as np
R=[json.loads(l) for l in open("results/grid.jsonl")]
main=[r for r in R if r["arm"] in("A_dims","B_precision","C_both")]
print("OPERATING POINTS (smallest total bytes that passes the frozen rule; total = documents + PCA matrix + int8 scales)")
for ds in ["scifact","nfcorpus","fiqa"]:
  for enc in ["small","base"]:
    c=[r for r in main if r["dataset"]==ds and r["encoder"]==enc and r["passes_rule"]]
    b=min(c,key=lambda r:r["doc_bytes"]+r["extra_bytes"]); tot=b["doc_bytes"]+b["extra_bytes"]
    print(f"{ds:9s}{enc:6s} dim {b['dim']:4d} {b['precision']:5s} loss {b['loss']*100:5.2f} hi {b['loss_hi']*100:5.2f} share {b['share_lose_02']*100:4.1f}% bytes {tot/b['full_doc_bytes']*100:5.1f}% of full ({tot/1e6:.2f} MB of {b['full_doc_bytes']/1e6:.1f} MB)")
print("\nNEAR MISSES: the first FAILING setup just below the operating point, and which condition failed")
for ds in ["scifact","nfcorpus","fiqa"]:
  for enc in ["small","base"]:
    f=[r for r in main if r["dataset"]==ds and r["encoder"]==enc and not r["passes_rule"] and r["precision"] in("fp32","int8")]
    f=sorted(f,key=lambda r:-(r["doc_bytes"]+r["extra_bytes"]))
    # largest failing setup
    r=[x for x in f if x["precision"]=="fp32" and x["arm"]=="A_dims"]; r=sorted(r,key=lambda x:-x["dim"])[0]
    why=[n for n,ok in (("mean loss>1.0",r["loss"]>0.01),("upper>2.0",r["loss_hi"]>0.02),("share>5%",r["share_lose_02"]>0.05)) if ok]
    print(f"{ds:9s}{enc:6s} first failing dim (fp32): {r['dim']} loss {r['loss']*100:.2f} hi {r['loss_hi']*100:.2f} share {r['share_lose_02']*100:.1f}% -> {', '.join(why)}")
print("\nPCA vs TRUNCATION mean nDCG@10 difference (PCA minus truncation, points), by dim")
for enc in ["small","base"]:
  for ds in ["scifact","nfcorpus","fiqa"]:
    rows=[]
    for r in R:
      if r["dataset"]==ds and r["encoder"]==enc and r["arm"]=="A_dims":
        t=[x for x in R if x["dataset"]==ds and x["encoder"]==enc and x["arm"]=="A_truncation" and x["dim"]==r["dim"]][0]; rows.append((r["dim"],(r["ndcg10"]-t["ndcg10"])*100))
    print(ds,enc," ".join(f"d{d}:{v:+.1f}" for d,v in rows))
print("\nFITTING SAMPLE CHECK: largest change in mean nDCG@10 across 3 fitting samples (points)")
mx=0
for ds in ["scifact","nfcorpus","fiqa"]:
  for enc in ["small","base"]:
    for k in sorted({r["dim"] for r in R if r["arm"]=="A_dims" and r["dataset"]==ds and r["encoder"]==enc}):
      v=[r["ndcg10"] for r in R if r["dataset"]==ds and r["encoder"]==enc and r["dim"]==k and r["arm"] in("A_dims","A_dims_seed")]
      d=(max(v)-min(v))*100; mx=max(mx,d)
      if d>0.5: print("  ",ds,enc,k,f"{d:.2f}")
print("largest overall:",round(mx,2))
