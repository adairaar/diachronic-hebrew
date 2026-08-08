"""Uncalibrated vs variance-matched: the trade-off, computed rather than asserted."""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import numpy as np, pandas as pd, importlib.util, json
from scipy import stats
pt = importlib.util.spec_from_file_location("pt",DH.script("predict_targets.py"))
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)
Dd = pd.read_csv(DH.f("big_features_500.csv"))
feats=[c for c in Dd.columns if c not in PT.META]
Xa=Dd[feats].astype(float); keep=(Xa.std()>0)&(Xa.isna().mean()<.2)
feats=list(np.array(feats)[keep.values]); med=Dd[feats].astype(float).median()
X=Dd[feats].astype(float).fillna(med).values
y=Dd.date_bce.values.astype(float); g=Dd.unit.values
books=list(pd.unique(g)); bdate={b:y[g==b][0] for b in books}
LAM=PT.LAM

def fit_all(Xtr,ytr,wtr,Xte):
    mu=Xtr.mean(0); sd=Xtr.std(0); sd=np.where(sd>0,sd,1.)
    A=(Xtr-mu)/sd; B=(Xte-mu)/sd
    yb=np.average(ytr,weights=wtr); s=np.sqrt(wtr)
    Aw=A*s[:,None]; yw=(ytr-yb)*s
    ev,V=np.linalg.eigh(Aw@Aw.T); Vty=V.T@yw; BAt=(B@Aw.T)@V
    return {lam: BAt@(Vty/(ev+lam))+yb for lam in LAM}

t,p=[],[]
for b in books:
    te=g==b; tr=~te; inner=[x for x in books if x!=b]
    wb=dict(zip(inner,PT.wts([bdate[x] for x in inner])))
    wv=np.array([wb[u] for u in g[tr]])
    err={l:[] for l in LAM}
    for bb in inner:
        m=g[tr]!=bb
        pr=fit_all(X[tr][m],y[tr][m],wv[m],X[tr][~m])
        for l in LAM: err[l].append(abs(np.median(pr[l])-bdate[bb]))
    bl=min(LAM,key=lambda l:np.mean(err[l]))
    t.append(bdate[b]); p.append(float(np.median(fit_all(X[tr],y[tr],wv,X[te])[bl])))
t=np.array(t); p=np.array(p)
out={}
for name,cal in [("uncal",p),("var",t.mean()+ (t.std()/p.std())*(p-p.mean()))]:
    r=t-cal; pre=t>586; post=t<586
    out[name]=dict(mae=float(np.abs(r).mean()),
                   rho=float(stats.spearmanr(t,cal)[0]),
                   span=float(cal.max()-cal.min()),
                   pre_ok=int((cal[pre]>586).sum()), n_pre=int(pre.sum()),
                   post_ok=int((cal[post]<586).sum()), n_post=int(post.sum()))
    print(name, out[name])
out["true_span"]=float(t.max()-t.min())
json.dump(out,open(DH.f("uncal_compare.json"),"w"),indent=2)
