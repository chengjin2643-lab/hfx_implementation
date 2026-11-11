"""Public balanced-risk candidates; no input trace or trial result is loaded."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
from . import canonical_layouts as prior
from .search_public import allocate

RULE='hfx-public-balanced-risk-v20261110-w32'
MAX_VARIANCE_RATIO=2.0
SCORE_POINTS=257
GUARD_POINTS=4097

def profile(mem,U,K,eps=1.,mode='payload',save_candidates=False):
    start=__import__('time').time()
    cap=int(mem)*1024
    old=prior.profile(mem,U,K,eps,mode).copy()
    sig=old['sigma']
    grid=np.unique(np.rint(np.geomspace(1,U,SCORE_POINTS)))
    dense=np.unique(np.rint(np.geomspace(1,U,GUARD_POINTS)))
    gm=old['gmod_width'];gr=old['gmod_rows']
    ref=prior.risks(grid,.5*prior.probabilities(gm),np.full(gm,gr),sig,False)
    dense_ref=prior.risks(dense,.5*prior.probabilities(gm),np.full(gm,gr),sig,False)
    specs=[];seen=set();scores=[]
    def bits_for(w):return (cap-int(prior.GEOM.loc[w,'lean_base_bytes']))//8*64 if mode=='full' else cap*8
    def add(w,q,rows,origin):
        rows=np.asarray(rows,dtype=np.int64)
        if rows.min()<4 or rows.sum()>bits_for(w):return
        key=(int(q),tuple(rows))
        if key in seen:return
        seen.add(key)
        risk=prior.risks(grid,prior.probabilities(w,q),rows,sig)
        ratio=risk/ref
        specs.append(dict(w=int(w),q16=int(q),rows=rows.tolist(),origin=origin))
        scores.append(dict(index=len(specs)-1,w=w,q16=q,origin=origin,
                           max_variance_ratio=float(ratio.max()),mean_log_variance_ratio=float(np.log(ratio).mean())))
    add(old['w'],old['q16'],old['rows'],'canonical-public-layout')
    for w in range(1,33):
        bits=bits_for(w)
        add(w,16,allocate(bits,w,np.ones(w)),'uniform-dyadic')
        if w==gm+1 and bits-gm*gr>=4:
            add(w,8,np.r_[bits-gm*gr,np.full(gm,gr)],'retained-GMod-coordinates')
        if w==1:continue
        for q in range(1,17):
            probs=prior.probabilities(w,q)
            for power in [0,.25,.5,.75,1.,1.5,2.]:
                add(w,q,allocate(bits,w,probs**power),f'probability-power-{power}')
            for fraction in [1/512,1/256,1/128,1/64,1/32,1/16,1/8,1/4,1/3,1/2,2/3,3/4,7/8]:
                for tailpower in [0,.5,1.]:
                    tail=probs[1:]**tailpower
                    add(w,q,allocate(bits,w,np.r_[fraction,(1-fraction)*tail/tail.sum()]),
                        f'head-{fraction}-tailpower-{tailpower}')
    for c in prior.CORE:
        w=c['w'];bits=bits_for(w);free=bits-64*w
        shape=np.maximum(np.asarray(c['rows'])-64,1.)
        real=64+free*shape/shape.sum();words=np.floor(real/64).astype(np.int64)
        left=bits//64-words.sum();order=np.argsort(-(real/64-words),kind='stable');words[order[:left]]+=1
        add(w,c['q16'],words*64,'scaled-previous-public-shape')
    table=pd.DataFrame(scores)
    valid=table[table.max_variance_ratio<=MAX_VARIANCE_RATIO*(1+1e-12)].sort_values(
        ['mean_log_variance_ratio','w','q16'],kind='stable')
    checks=[];best=None
    # Coarse objective, dense guard: deterministic; no observed stream values.
    for _,s in valid.iterrows():
        i=int(s['index']);p=specs[i]
        rr=prior.risks(dense,prior.probabilities(p['w'],p['q16']),np.asarray(p['rows']),sig)/dense_ref
        dense_max=float(rr.max());checks.append(dict(index=i,dense_max_variance_ratio=dense_max))
        if dense_max<=MAX_VARIANCE_RATIO*(1+1e-10):
            best=p.copy();best.update(s.to_dict());best['dense_max_variance_ratio']=dense_max;break
    if best is None:raise ValueError('No candidate satisfies the public guard')
    # Resolve numerically equivalent optima by smaller width, then q, then rows.
    objective=best['mean_log_variance_ratio']
    tied=valid[valid.mean_log_variance_ratio<=objective+1e-10].sort_values(['w','q16'],kind='stable')
    for _,s in tied.iterrows():
        i=int(s['index']);p=specs[i]
        rr=prior.risks(dense,prior.probabilities(p['w'],p['q16']),np.asarray(p['rows']),sig)/dense_ref
        if float(rr.max())<=MAX_VARIANCE_RATIO*(1+1e-10):
            best=p.copy();best.update(s.to_dict());best['dense_max_variance_ratio']=float(rr.max());break
    best.update(memory_kib=int(mem),capacity=int(U),K=int(K),epsilon=float(eps),mode=mode,
                gmod_width=gm,gmod_rows=gr,gmod_state_bytes=old['gmod_state_bytes'],sigma=sig,
                state_bytes=int(prior.GEOM.loc[best['w'],'lean_base_bytes'])+(sum(best['rows'])+63)//64*8,
                selector_rule=RULE,variance_ratio_cap=MAX_VARIANCE_RATIO,
                score_grid_points=len(grid),guard_grid_points=len(dense),candidate_count=len(specs),
                elapsed_seconds=__import__('time').time()-start)
    def signature(p):return p['q16'],tuple(p['rows']),p['gmod_width'],p['gmod_rows']
    best['changed_vs_canonical']=signature(best)!=signature(old)
    best['prior_layout']={k:old[k] for k in ['w','q16','rows','profile_id']}
    best['profile_id']=hashlib.sha256(json.dumps({k:v for k,v in best.items() if k!='elapsed_seconds'},sort_keys=True).encode()).hexdigest()[:16]
    if mode=='full':assert max(best['state_bytes'],best['gmod_state_bytes'])<=cap
    if save_candidates:
        table.to_csv(ROOT/f'balanced_candidates_m{mem}_u{U}_k{K}_{mode}.csv',index=False)
    return best

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('memory',type=int);a.add_argument('--capacity',type=int,default=2097152);a.add_argument('--K',type=int,default=393);a.add_argument('--mode',default='payload');opts=a.parse_args()
    p=profile(opts.memory,opts.capacity,opts.K,mode=opts.mode,save_candidates=True)
    out=ROOT/f'balanced_m{opts.memory}_u{opts.capacity}_k{opts.K}_{opts.mode}.json';out.write_text(json.dumps(p,indent=2),encoding='utf-8')
    print(json.dumps(p,indent=2),flush=True)
