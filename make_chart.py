import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=[json.loads(l) for l in open("results/grid.jsonl")]
ds,enc="fiqa","base"; rows=[r for r in R if r["dataset"]==ds and r["encoder"]==enc and r["arm"] in("A_dims","B_precision","C_both")]
full=[r for r in rows if r["arm"]=="B_precision" and r["precision"]=="fp32"][0]["ndcg10"]
sty={"fp32":("#1f77b4","o","FP32 numbers"),"int8":("#d55e00","s","int8 numbers (1 byte each)"),"bit1":("#6a3d9a","^","1 bit numbers")}
fig,ax=plt.subplots(figsize=(10,5.6),dpi=200)
for p,(c,m,l) in sty.items():
    pts=sorted([(((r["doc_bytes"]+r["extra_bytes"])/r["full_doc_bytes"])*100,r["ndcg10"]*100,r["dim"]) for r in rows if r["precision"]==p],key=lambda t:-t[2])
    ax.plot([x for x,_,_ in pts],[y for _,y,_ in pts],color=c,marker=m,lw=1.6,ms=6,label=l)
    if p=="int8":
        for x,y,d in pts:
            if d in (512,256,128,64): ax.annotate(f"{d} dims",(x,y),textcoords="offset points",xytext=(-4,9) if d!=768 else (4,-14),fontsize=8,color=c)
ax.axhline((full-0.01)*100,color="#444",ls="--",lw=1); ax.text(0.075,(full-0.01)*100+0.6,"1 point below the full vector score",fontsize=8.5,color="#444")
ax.set_xscale("log"); ax.set_xlabel("Index size, percent of the full FP32 vector index (log scale)"); ax.set_ylabel("nDCG@10, out of 100 (higher is better)")
ax.set_title("FiQA with BGE base: quality against actual index size",fontsize=12,loc="left")
ax.grid(alpha=.25); ax.legend(frameon=False,loc="lower right"); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/featured_chart.png"); print("saved")
