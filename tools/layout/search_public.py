"""Deterministic public layout search, without loading any endpoint trace.

Candidate family and objective are fixed in this script before execution.
The complete public count range, budget, horizon and privacy budget enter.
"""
from pathlib import Path
import importlib.util,json,hashlib,sys,time
import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parent
from . import canonical_layouts as old
U=2097152;K=393;EPS=1.;SIG=old.sigma(K,EPS)
GRID=np.unique(np.rint(np.geomspace(1,U,257)))
DENSE=np.unique(np.rint(np.geomspace(1,U,4097)))
CAPS=[1.05,1.1,1.25,1.5,2.,4.]

def allocate(bits,w,weights):
    words=bits//64
    weights=np.asarray(weights,dtype=float)
    available=words-w
    real=1+available*weights/weights.sum()
    floors=np.floor(real).astype(np.int64)
    left=words-int(floors.sum())
    if left:
        order=np.argsort(-(real-floors),kind='stable')
        floors[order[:left]]+=1
    return floors*64

def candidates(mem):
    bits=mem*8192; seen=set(); out=[]
    def add(w,q,rows,origin):
        rows=np.asarray(rows,dtype=np.int64)
        key=(q,tuple(rows))
        if key in seen:return
        seen.add(key)
        out.append(dict(w=int(w),q16=int(q),rows=rows.tolist(),origin=origin))
    orig=old.profile(mem,U,K,EPS,'payload')
    add(orig['w'],orig['q16'],orig['rows'],'canonical')
    for w in range(1,17):
        add(w,16,allocate(bits,w,np.ones(w)),'uniform-dyadic')
        if w==1:continue
        for q in range(1,17):
            # Uniform rows and probability-proportional/softened rows.
            probs=old.probabilities(w,q)
            for power in [0,.25,.5,.75,1.,1.5,2.]:
                add(w,q,allocate(bits,w,probs**power),f'probability-power-{power}')
            # Continuous head size at logarithmic fractions, dyadic tail.
            for frac in [1/512,1/256,1/128,1/64,1/32,1/16,1/8,1/4,1/3,1/2,2/3,3/4,7/8]:
                for tailpower in [0,.5,1.]:
                    tail=probs[1:]**tailpower
                    weights=np.r_[frac,(1-frac)*tail/tail.sum()]
                    add(w,q,allocate(bits,w,weights),f'head-{frac}-tailpower-{tailpower}')
    return orig,out

def score(mem):
    orig,spec=candidates(mem)
    gm=orig['gmod_width'];gr=orig['gmod_rows']
    # The reference is the single public GMod layout used in actual execution.
    ref=old.risks(GRID,.5*old.probabilities(gm),np.full(gm,gr),SIG,False)
    rows=[];vectors=[]
    start=time.time()
    for i,p in enumerate(spec):
        risk=old.risks(GRID,old.probabilities(p['w'],p['q16']),np.asarray(p['rows']),SIG)
        ratio=risk/ref
        vectors.append(risk)
        rows.append(dict(index=i,w=p['w'],q16=p['q16'],origin=p['origin'],
                         max_ratio=float(ratio.max()),mean_log_ratio=float(np.log(ratio).mean()),
                         square_mean_ratio=float(np.exp(logsumexp(2*np.log(ratio))-np.log(len(ratio)))),
                         max_relative_std=float(np.sqrt(risk).max()),
                         capacity_rse=float(np.sqrt(risk[-1])/U),
                         rows=';'.join(map(str,p['rows']))))
        if i%1000==0:print(mem,i,'/',len(spec),round(time.time()-start,1),flush=True)
    table=pd.DataFrame(rows)
    table.to_csv(ROOT/f'candidates_{mem}.csv',index=False)
    selections=[]
    indices={0}
    for cap in CAPS:
        valid=table[table.max_ratio<=cap]
        if valid.empty:continue
        for objective in ['mean_log_ratio','square_mean_ratio']:
            i=int(valid.sort_values([objective,'w','q16'],kind='stable').iloc[0]['index'])
            indices.add(i)
            selections.append(dict(cap=cap,objective=objective,index=i))
    for w in [1,2,3,4]:
        for q in [8,16]:
            chosen=table[(table.w==w)&(table.q16==q)&table.origin.eq('uniform-dyadic')]
            indices.update(chosen['index'].astype(int))
    profiles=[];curves=[]
    for i in sorted(indices):
        p=spec[i].copy();p.update(table.iloc[i].drop(labels=['rows','index']).to_dict())
        p.update(gmod_width=gm,gmod_rows=gr,capacity=U,K=K,epsilon=EPS,
                 memory_kib=mem,mode='payload',sigma=SIG,selector_rule='public-grid-candidate-evaluation-v20261110')
        p['profile_id']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:16]
        p['index']=i
        dense=old.risks(DENSE,old.probabilities(p['w'],p['q16']),np.asarray(p['rows']),SIG)
        dense_ref=old.risks(DENSE,.5*old.probabilities(gm),np.full(gm,gr),SIG,False)
        p['dense_max_ratio']=float(np.max(dense/dense_ref))
        p['public_load_metrics']={str(int(n)):dict(rmse=float(np.sqrt(old.risks([n],old.probabilities(p['w'],p['q16']),np.asarray(p['rows']),SIG)[0])),
                                     ratio=float(old.risks([n],old.probabilities(p['w'],p['q16']),np.asarray(p['rows']),SIG)[0]/old.risks([n],.5*old.probabilities(gm),np.full(gm,gr),SIG,False)[0]))
                                  for n in [U/256,U/64,U/16,U/4,U]}
        profiles.append(p)
        for n,a,b in zip(DENSE,dense,dense_ref):curves.append(dict(index=i,profile_id=p['profile_id'],n=int(n),rmse=float(np.sqrt(a)),rse=float(np.sqrt(a)/n),gmod_rmse=float(np.sqrt(b)),risk_ratio=float(a/b)))
    report=dict(memory_kib=mem,candidates=len(spec),grid_points=len(GRID),dense_grid_points=len(DENSE),
                selections=selections,profiles=profiles,seconds=time.time()-start)
    (ROOT/f'profiles_{mem}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame(curves).to_csv(ROOT/f'public_risk_curves_{mem}.csv',index=False)
    print('DONE',mem,'profiles',len(profiles),flush=True)
    for p in profiles:print(mem,p['index'],p['w'],p['q16'],p['rows'],round(p['max_ratio'],3),round(p['mean_log_ratio'],3),p['public_load_metrics'][str(U//16)],flush=True)
    return report

if __name__=='__main__':
    ROOT.mkdir(exist_ok=True,parents=True)
    for mem in [64,128,256]:score(mem)
