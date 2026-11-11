#pragma once
#include "reference.hpp"

// Same exact bounded covariance-aware scalar objective as the supplied decoder.
// Cache only the last public grid vector, not a private warm-start or full table.
class AdaptiveHfx:public HfxMethod {
    vector<double> counts_,a_,work_;double previous_private_=0;
    double prob(int j)const{return exp2(-(j==width()-1?width()-1:j+1));}
    void restore_logs(){for(int j=0;j<width();++j)work_[j]=log1p(-2*prob(j)/rows());}
    double response(double n,double*derivative=nullptr)const{
        double sum=0,grad=0;
        for(int j=0;j<width();++j){double l=work_[j],v=n*l;sum+=a_[j]*-expm1(v);if(derivative)grad-=a_[j]*l*exp(v);}
        if(derivative)*derivative=.5*rows()*grad;
        return .5*rows()*sum;
    }
    double decode(double y)const{
        if(y<=0)return 0;if(y>=response(c.nmax))return c.nmax;
        double lo=0,hi=c.nmax,x=clamp(previous_private_,lo,hi);
        for(int k=0;k<80;++k){
            double slope=0,v=response(x,&slope),error=v-y;
            if(slope>0&&abs(error/slope)<1e-7){
                // On a floating-point response plateau use the same canonical
                // guarded bisection as the reference, not an unstable Newton root.
                if(32*numeric_limits<double>::epsilon()*(1+abs(v))/slope>1e-4){
                    double l=0,h=c.nmax;for(int i=0;i<52;++i){double mid=.5*(l+h);if(response(mid)<y)l=mid;else h=mid;}return .5*(l+h);
                }
                return x;
            }
            if(error<0)lo=x;else hi=x;
            if(hi-lo<1e-7)return .5*(hi+lo);
            double next=slope>0?x-error/slope:.5*(lo+hi);
            if(!(next>lo&&next<hi)||next==x)next=.5*(lo+hi);
            x=next;
        }
        return .5*(lo+hi);
    }
public:
    AdaptiveHfx(string label,const Config&cfg,uint64_t sh,uint64_t sn,int forced):HfxMethod(label,cfg,sh,sn,forced),counts_(width(),0),a_(width(),1),work_(width()*width()+2*width()+4,0){work_.back()=-1;restore_logs();}
    uint64_t bytes()const override{return HfxMethod::bytes()+sizeof(AdaptiveHfx)-sizeof(HfxMethod)+(counts_.capacity()+a_.capacity()+work_.capacity())*sizeof(double);}
    void update_hashed(uint64_t x,uint64_t y){auto[j,inc]=HfxMethod::update_hashed_record(x,y);counts_[j]+=inc;}
    void update(uint64_t id,int)override{auto[x,y]=hash_pair(id,hashseed);update_hashed(x,y);}
    bool design(double pilot){
        int w=width();if(w==1){a_[0]=1;return false;}
        int grid=(int)round(log(clamp(pilot,1.,(double)c.nmax))/log((double)c.nmax)*255.);
        if(work_.back()==grid&&work_[work_.size()-2]==sigma)return false;
        double n=exp(log((double)c.nmax)*grid/255.);
        double*C=work_.data(),*g=C+w*w,*trial=g+w;
        for(int j=0;j<w;++j){double pj=prob(j),lj=log1p(-2*pj/rows());g[j]=-.5*rows()*lj*exp(n*lj);
            for(int k=0;k<w;++k){double pk=prob(k),lk=log1p(-2*pk/rows());double lr=log1p(-4*(pj/rows())*(pk/rows())/((1-2*pj/rows())*(1-2*pk/rows())));
                double cov=.25*exp(n*(lj+lk))*expm1(n*lr);C[j*w+k]=(double)rows()*rows()*cov;
                if(j==k)C[j*w+k]=-.25*rows()*expm1(2*n*lj)+(double)rows()*(rows()-1)*cov;
            }
        }
        double best=1e300;
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
                if(iteration==1999)throw runtime_error("adaptive solver did not converge");
            }
            double score=(quadratic+sigma*sigma)/(slope*slope);
            if(score<best){best=score;copy(trial,trial+w,a_.begin());}
        }
        for(double a:a_)if(a<0||a>1||!isfinite(a))throw runtime_error("adaptive sensitivity bound");
        restore_logs();work_.back()=grid;work_[work_.size()-2]=sigma;return true;
    }
    Estimate release(uint64_t j)override{
        if(j>1)design(previous_private_);double total=0;for(int k=0;k<width();++k)total+=a_[k]*counts_[k];
        double noisy=total+sigma*rng.normal(),estimate=decode(noisy);
        double diagnostic=diagnostic_clean?decode(total):0.;previous_private_=estimate;
        return {estimate,diagnostic,noisy<0||noisy>response(c.nmax)};
    }
    const vector<double>&weights()const{return a_;}
    double column_sum()const{return accumulate(counts_.begin(),counts_.end(),0.);}
    double decode_for_test(double y)const{return decode(y);}
    double response_for_test(double n)const{return response(n);}
};

static unique_ptr<Mechanism> adaptive_factory(string label,Config c,uint64_t rows,int w,uint64_t sh,uint64_t sn){
    FULL_STATE_HFX_ROWS=rows;
    unique_ptr<Mechanism> result=w==1?unique_ptr<Mechanism>(new HfxMethod(label,c,sh,sn,1)):unique_ptr<Mechanism>(new AdaptiveHfx(label,c,sh,sn,w));
    FULL_STATE_HFX_ROWS=0;return result;
}
static Layout adaptive_layout(Config c,uint64_t cap,int width,bool payload){
    auto probe=adaptive_factory("probe",c,8,width,0,0);uint64_t base=probe->bytes()-((8ull*width+63)/64)*8;
    uint64_t rows=payload?c.mem*8192ull/width:((cap-base)/8)*64/width;
    if(rows<4)throw runtime_error("insufficient adaptive state");
    return {width,rows,base+((rows*width+63)/64)*8,0};
}
