#pragma once
#include <set>
// Include after the original Mechanism, ParityState, and TreeNoise definitions.
// Public finite-K tuning of the same subtraction-tree family. The naive AJS
// factorization is intentionally represented by the existing Exact-DP control.
static double ajs_exact_squared_sensitivity(uint64_t K,int b){
    TreeNoise t(K,b,1);set<pair<uint64_t,uint64_t>>used;
    for(uint64_t j=1;j<=K;++j)for(auto[key,sign]:t.terms(j))if(sign)used.insert(key);
    constexpr int64_t neg=-1000000000000ll;
    function<array<int64_t,4>(uint64_t,uint64_t)>dp=[&](uint64_t start,uint64_t size)->array<int64_t,4>{
        // The difference stream is padded by zeros: every endpoint after K
        // holds membership q_K, rather than resetting it to zero.
        if(start>=K)return{0,neg,neg,0};
        bool own=used.count({start,size});
        if(size==1)return{0,own?1:0,own?1:0,0};
        array<int64_t,4>comb{0,neg,neg,0};uint64_t sub=size/b;
        for(int i=0;i<b;++i){auto child=dp(start+i*sub,sub);array<int64_t,4>next{neg,neg,neg,neg};
            for(int l=0;l<2;++l)for(int r=0;r<2;++r)for(int mid=0;mid<2;++mid)
                next[2*l+r]=max(next[2*l+r],comb[2*l+mid]+child[2*mid+r]);
            comb=next;
        }
        if(own){++comb[1];++comb[2];}return comb;
    };
    auto v=dp(0,t.leaves);return (double)max(v[0],v[1]); // q_0=0, q_K free
}
static int ajs_public_branch(uint64_t K){
    double best=numeric_limits<double>::infinity();int selected=3;
    for(int b=3;b<=31;b+=2){
        TreeNoise t(K,b,1);double norm=0;
        for(uint64_t j=1;j<=K;++j)for(auto[key,sign]:t.terms(j))norm+=sign*sign;
        double score=ajs_exact_squared_sensitivity(K,b)*norm/K;
        if(score<best){best=score;selected=b;}
    }
    return selected;
}

class PublicTunedAjs:public Mechanism {
    ParityState parity;TreeNoise tree;
public:
    PublicTunedAjs(const Config&c,uint64_t sn,int public_branch=0):Mechanism("ajs",c,sn),tree(c.K,public_branch?public_branch:ajs_public_branch(c.K),sn+193){
        sensitivity=sqrt(ajs_exact_squared_sensitivity(c.K,tree.b));sigma=analytic_gaussian_sigma(c.epsilon,c.delta,sensitivity);
        calibration="AJS_public_odd_b3to31_exact_S1K_sensitivity";
    }
    void update(uint64_t id,int)override{parity.toggle(id);}
    Estimate release(uint64_t j)override{
        double q=(double)parity.count(),x=q+sigma*tree.noise(j);return{bounded(x),q,x<0||x>c.nmax};
    }
    Estimate release_query(double q,uint64_t j){
        double x=q+sigma*tree.noise(j);return{bounded(x),q,x<0||x>c.nmax};
    }
    uint64_t bytes()const override{return sizeof(*this)+parity.bytes()-sizeof(parity)+tree.bytes()-sizeof(tree);}
    int branch()const{return tree.b;}
    double raw_statistic()const override{return (double)parity.count();}
};
