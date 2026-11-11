"""Public-only heterogeneous cell allocation. No stream values enter selection."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr
ROOT=Path(__file__).resolve().parent

def sigma(K,eps=1,delta=1e-6):
 f=lambda mu:ndtr(mu/2-eps/mu)-np.exp(eps)*ndtr(-mu/2-eps/mu)-delta
 return np.sqrt(K)/brentq(f,1e-6,5,xtol=1e-14)

def probabilities(w,q16=16):
 if q16==16:return 2.**(-np.minimum(np.arange(1,w+1),w-1))
 tail=2.**(-np.minimum(np.arange(1,w),w-2))
 return np.r_[1-q16/16.,q16/16.*tail]

def rows_for(bits,w,head_power):
 words=bits//64
 if head_power==0:
  words_each=words//w
  return np.full(w,words_each*64,dtype=np.int64)
 head=max(1,words//(2**head_power));tail=(words-head)//(w-1)
 return np.r_[head*64,np.full(w-1,tail*64,dtype=np.int64)]

def risks(grid,p,m,sig,weighted=True):
 # n and (cell j,cell k) arrays; finite-n covariance, not independent bits.
 n=np.asarray(grid)[:,None];r=p/m;l=np.log1p(-2*r);b=1-2*r
 g=-.5*m*l*np.exp(n*l);d=-.25*m*np.expm1(2*n*l)
 corr=np.log1p(-4*r[:,None]*r[None,:]/(b[:,None]*b[None,:]))
 C=.25*m[:,None]*m[None,:]*np.exp(n[:,:,None]*(l[:,None]+l[None,:]))*np.expm1(n[:,:,None]*corr)
 j=np.arange(len(p));C[:,j,j]=d+np.diagonal(C,axis1=1,axis2=2)*(m-1)/m
 a=np.ones_like(g)
 if weighted:
  ratios=d/np.maximum(g,1e-290);order=np.argsort(ratios,axis=1,kind='stable')
  ds=np.cumsum(np.take_along_axis(d,order,axis=1),axis=1);gs=np.cumsum(np.take_along_axis(g,order,axis=1),axis=1)
  lam=(ds+sig*sig)/np.maximum(gs,1e-290);threshold=np.take_along_axis(ratios,order,axis=1)
  following=np.c_[threshold[:,1:],np.full(len(grid),np.inf)]
  index=np.argmax(lam<=following,axis=1);chosen=lam[np.arange(len(grid)),index]
  a=np.minimum(1,chosen[:,None]*g/np.maximum(d,1e-290))
 slope=np.sum(a*g,axis=1);var=np.einsum('ni,nij,nj->n',a,C,a)+sig*sig
 return np.minimum(1e290,np.maximum(var,1e-30)/np.maximum(slope,1e-145)**2)

def select(case,mem,U,K,mode,geometry,points=129):
 grid=np.unique(np.rint(np.geomspace(1,U,points)));sig=sigma(K);spec=[];values=[]
 for w in range(1,17):
  base=int(geometry.loc[w,'base_bytes']);bits=(mem*1024-base)//8*64 if mode=='complete' else mem*8192
  if bits<64*w:continue
  # Include the original all-ID dyadic layout and a small public head stratum.
  for q16,hp in [(16,0)]+([(q,h) for q in [4,6,8,10,12,14] for h in [3,4,5,6,7,8]] if w>=2 else []):
   rows=rows_for(bits,w,hp);p=probabilities(w,q16)
   value=risks(grid,p,rows,sig)
   spec.append(dict(case=case,memory_kib=mem,capacity=U,K=K,memory_mode=mode,w=w,q16=q16,head_power=hp,rows=rows.tolist(),state_bytes=base+int(rows.sum())//8,sigma=sig));values.append(value)
 matrix=np.asarray(values);oracle=matrix.min(axis=0);regret=matrix/oracle
 score=np.max(regret,axis=1);logscore=np.mean(np.log(regret),axis=1)
 tie=lambda i:(spec[i]['w'],spec[i]['q16']!=16,spec[i]['head_power'],spec[i]['q16'])
 imin=min(np.flatnonzero(score<=score.min()*(1+1e-9)),key=tie)
 ilog=min(np.flatnonzero(logscore<=logscore.min()+1e-9),key=tie)
 best=spec[imin];best['objective']='minimax public pointwise local-risk regret';best['score']=float(score[imin])
 logbest=spec[ilog].copy();logbest['objective']='mean log public local-risk regret';logbest['score']=float(logscore[ilog])
 records=[]
 for i,s in enumerate(spec):
  records.append({k:v for k,v in s.items() if k not in ['rows','objective','score']}|dict(rows=';'.join(map(str,s['rows'])),max_regret=score[i],log_regret=logscore[i],risk_at_half_capacity=float(np.interp(U/2,grid,matrix[i])),risk_at_capacity=float(matrix[i,-1])))
 return best,logbest,records

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--points',type=int,default=129);args=parser.parse_args()
 geometry=pd.read_csv(ROOT/'data/measured_geometry.csv').set_index('w')
 jobs=[('Active32',32,2000000,200,'complete'),('Active128',128,2000000,200,'complete')]+[(f'Azure{m}',m,2097152,393,'payload') for m in [8,16,32,64,128,256,512,1024]]
 choices=[];logs=[];allrecords=[]
 for job in jobs:
  best,logbest,records=select(*job,geometry,args.points);choices.append(best);logs.append(logbest);allrecords.extend(records)
  print(job[0],json.dumps(best),flush=True)
 (ROOT/'data/public_layout_minimax.json').write_text(json.dumps(choices,indent=2))
 (ROOT/'data/public_layout_logmean.json').write_text(json.dumps(logs,indent=2))
 pd.DataFrame(allrecords).to_csv(ROOT/'data/public_layout_candidates.csv',index=False)
if __name__=='__main__':main()
