#pragma once
// Frozen bounded-convex weighted-scalar rule. No oracle population enters design.
// Original source copies differ only by a toggle-return readout hook.
#define ICDE_LIBRARY
#include "frozen/icde_audit.cpp"

class LaggedHfx:public HfxMethod {
    vector<double> counts_,a_,work_;double previous_private_=0;
    double prob(int j)const{return exp2(-(j==width()-1?width()-1:j+1));}
    double phi_weighted(double n)const{double s=0;for(int j=0;j<width();++j)s+=a_[j]*-expm1(n*log1p(-2*prob(j)/rows()));return .5*rows()*s;}
    double decode(double y)const{if(y<=0)return 0;if(y>=phi_weighted(c.nmax))return c.nmax;double lo=0,hi=c.nmax;for(int k=0;k<52;++k){double mid=.5*(lo+hi);if(phi_weighted(mid)<y)lo=mid;else hi=mid;}return .5*(lo+hi);}
public:
    LaggedHfx(const Config&c,uint64_t sh,uint64_t sn,int forced):HfxMethod("lagged",c,sh,sn,forced),counts_(width(),0),a_(width(),1),work_(width()*width()+2*width()+4,0){}
    uint64_t bytes()const override{return HfxMethod::bytes()+sizeof(LaggedHfx)-sizeof(HfxMethod)+(counts_.capacity()+a_.capacity()+work_.capacity())*sizeof(double);}
    void update_hashed(uint64_t x,uint64_t y){auto[j,increment]=HfxMethod::update_hashed_record(x,y);counts_[j]+=increment;}
    void update(uint64_t id,int)override{auto[x,y]=hash_pair(id,hashseed);update_hashed(x,y);}
    void design(double private_pilot){
        int w=width();if(w==1){a_[0]=1;return;}
        int grid=(int)round(log(clamp(private_pilot,1.,(double)c.nmax))/log((double)c.nmax)*255.);
        double n=exp(log((double)c.nmax)*grid/255.);
        double*C=work_.data(),*g=C+w*w,*trial=g+w;
        for(int j=0;j<w;++j){double pj=prob(j),lj=log1p(-2*pj/rows());g[j]=-.5*rows()*lj*exp(n*lj);
            for(int k=0;k<w;++k){double pk=prob(k),lk=log1p(-2*pk/rows());double lr=log1p(-4*(pj/rows())*(pk/rows())/((1-2*pj/rows())*(1-2*pk/rows())));
                double cov=.25*exp(n*(lj+lk))*expm1(n*lr);C[j*w+k]=(double)rows()*rows()*cov;
                if(j==k)C[j*w+k]=-.25*rows()*expm1(2*n*lj)+(double)rows()*(rows()-1)*cov;
            }
        }
        double best=1e300;
        // Enumerate the coordinate attaining max(a)=1. Each remaining coordinate
        // has an exact one-variable minimizer of the same fractional quadratic.
        for(int pivot=0;pivot<w;++pivot){
            fill(trial,trial+w,1.);double slope=accumulate(g,g+w,0.),quadratic=0;
            for(int j=0;j<w;++j)for(int k=0;k<w;++k)quadratic+=C[j*w+k];
            for(int iteration=0;iteration<2000;++iteration){double change=0;
                for(int j=0;j<w;++j)if(j!=pivot){double cj=0;for(int k=0;k<w;++k)cj+=C[j*w+k]*trial[k];
                    double old=trial[j],A=C[j*w+j],B=cj-A*old,e=slope-g[j]*old,D=quadratic-A*old*old-2*B*old+sigma*sigma;
                    double denominator=A*e-B*g[j],value=1.;
                    if(denominator>0)value=clamp((D*g[j]-B*e)/denominator,0.,1.);
                    else{double left=D/(e*e),right=(A+2*B+D)/((e+g[j])*(e+g[j]));value=left<right?0.:1.;}
                    double delta=value-old;quadratic+=2*delta*cj+delta*delta*A;slope+=delta*g[j];trial[j]=value;change=max(change,abs(delta));
                }
                if(change<1e-10)break;
                if(iteration==1999)throw runtime_error("coordinate solver did not converge");
            }
            double score=(quadratic+sigma*sigma)/(slope*slope);
            if(score<best){best=score;copy(trial,trial+w,a_.begin());}
        }
        for(double a:a_)if(a<0||a>1)throw runtime_error("sensitivity bound");
    }
    Estimate release(uint64_t j)override{
        if(j>1)design(previous_private_);double total=0;for(int k=0;k<width();++k)total+=a_[k]*counts_[k];
        double noisy=total+sigma*rng.normal(),estimate=decode(noisy);previous_private_=estimate;
        return {estimate,diagnostic_clean?decode(total):0.,noisy<0||noisy>phi_weighted(c.nmax)};
    }
    const vector<double>&weights()const{return a_;}
    double column_sum()const{return accumulate(counts_.begin(),counts_.end(),0.);}
};

struct Layout {int w;uint64_t rows,bytes;double slope;};
static Layout geometry_fixed(string name,Config c,uint64_t cap,bool candidate){
    Layout best{0,0,0,-1};
    // Candidate retains original HFX's public width; only rows pay the overhead.
    int fixed_w=0;
    if(candidate){fixed_w=geometry_fixed("hfx",c,cap,false).w;}
    for(int w=1;w<=32;++w){
        if(name=="hx_phi"&&w!=32)continue;if(candidate&&w!=fixed_w)continue;
        FULL_STATE_HFX_ROWS=8;PUBLIC_AUDIT_ROWS_OVERRIDE=8;FULL_STATE_GMOD_WIDTH=w;c.width=w;
        unique_ptr<Mechanism> probe=candidate?unique_ptr<Mechanism>(new LaggedHfx(c,0,0,w)):make_mechanism(name,c,0,0);
        uint64_t base=probe->bytes()-((8ull*w+63)/64)*8;
        if(cap<=base+8)continue;uint64_t rows=((cap-base)/8)*64/w;
        double slope=0;for(double p:column_probabilities(w)){if(name=="hfx"){double a=log1p(-2*p/rows);slope+=-.5*rows*a*exp(c.nmax*a);}else slope+=.5*p*exp(-(double)c.nmax*p/rows);}
        uint64_t bytes=base+((rows*w+63)/64)*8;if(bytes>cap)throw runtime_error("cap overflow");
        if(slope>best.slope)best={w,rows,bytes,slope};
    }
    FULL_STATE_HFX_ROWS=0;PUBLIC_AUDIT_ROWS_OVERRIDE=0;FULL_STATE_GMOD_WIDTH=0;
    return best;
}
