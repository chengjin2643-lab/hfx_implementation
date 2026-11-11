"""Frozen density-aware public selector; no trace or measured error is read."""
from pathlib import Path
from copy import deepcopy
import hashlib,json
from .balanced_selector import prior,profile as balanced_profile,MAX_VARIANCE_RATIO

ROOT=Path(__file__).resolve().parent
RULE_ID='hfx-density-aware-v20261110-final'
DENSITY_THRESHOLD=4.0
_CACHE={}

def execution_signature(p):
    return dict(w=int(p['w']),q16=int(p['q16']),rows=list(map(int,p['rows'])),
                gmod_width=int(p['gmod_width']),gmod_rows=int(p['gmod_rows']),
                capacity=int(p['capacity']),K=int(p['K']),epsilon=float(p['epsilon']),
                mode=p['mode'],memory_kib=int(p['memory_kib']),sigma=float(p['sigma']))

def _cache_filename(mem,U,K,eps,mode):
    base=f'balanced_m{int(mem)}_u{int(U)}_k{int(K)}_{mode}'
    if float(eps)!=1.:base+='_e'+str(float(eps)).replace('.','p')
    return ROOT/(base+'.json')

def select_layout(memory_kib,capacity,releases,epsilon=1.,mode='payload',
                  objective='controlled',payload_bits=None):
    """Select all parameters before processing; objective is explicit.

    real_accuracy: declared density above four uses the existing minimax
    profile; lower density uses balanced log-risk with the fixed 2x numerical
    variance guard. controlled: always retain the original minimax profile.
    The threshold is a fixed engineering choice, not an optimality theorem.
    """
    if objective not in {'real_accuracy','controlled'}:
        raise ValueError('objective must be real_accuracy or controlled')
    if mode not in {'payload','full'}:raise ValueError('invalid memory mode')
    mem,U,K,eps=int(memory_kib),int(capacity),int(releases),float(epsilon)
    original=deepcopy(prior.profile(mem,U,K,eps,mode))
    if payload_bits is None:
        # Full-state callers may explicitly declare available bitmap bits.
        # Otherwise use the original, publicly determined bitmap allocation.
        payload_bits=mem*8192 if mode=='payload' else sum(original['rows'])
    bits=int(payload_bits)
    if bits<=0:raise ValueError('payload bits must be positive')
    key=(mem,U,K,eps,mode,objective,bits,DENSITY_THRESHOLD,
         MAX_VARIANCE_RATIO,RULE_ID)
    if key in _CACHE:return deepcopy(_CACHE[key])
    density=U/bits
    selected='minimax' if objective=='controlled' or density>DENSITY_THRESHOLD else 'balanced'
    source='canonical_layouts'
    if selected=='minimax':p=deepcopy(original)
    else:
        filename=_cache_filename(mem,U,K,eps,mode)
        if filename.exists():
            p=json.loads(filename.read_text(encoding='utf-8'))
            assert p['selector_rule']=='hfx-public-balanced-risk-v20261110-w32'
            assert p['variance_ratio_cap']==MAX_VARIANCE_RATIO
            assert (p['memory_kib'],p['capacity'],p['K'],p['epsilon'],p['mode'])==(mem,U,K,eps,mode)
            source=filename.name
        else:p=balanced_profile(mem,U,K,eps,mode)
    signature=execution_signature(p);prior_signature=execution_signature(original)
    p.update(objective=objective,selection_criterion=selected,
             declared_payload_bits=bits,declared_density=density,
             density_threshold=DENSITY_THRESHOLD,variance_ratio_cap=MAX_VARIANCE_RATIO,
             final_selector_rule=RULE_ID,profile_source=source,
             changed_vs_canonical=signature!=prior_signature,
             canonical_profile_id=original['profile_id'],execution_signature=signature,
             execution_signature_id=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:16])
    p['source_profile_id']=p['profile_id']
    p['profile_id']=hashlib.sha256(json.dumps(dict(rule=RULE_ID,objective=objective,
        density_threshold=DENSITY_THRESHOLD,variance_ratio_cap=MAX_VARIANCE_RATIO,
        signature=signature),sort_keys=True).encode()).hexdigest()[:16]
    _CACHE[key]=p
    return deepcopy(p)
