#pragma once
#include "heterogeneous.hpp"

// Research implementation. Counts and clean diagnostics must not be published.
// The release is one bounded-weight scalar plus the original Gaussian noise.
template<bool ParallelRows=false> class LeanHfx:public Mechanism {
    int w_,q16_;uint64_t hashseed_;
    vector<uint64_t> bits_;
    vector<uint32_t> geometry_; // rows and offsets
    vector<double> buffer_; // counts,weights,g/reciprocals,d,logs,grid
    double previous_=0;
    double* counts(){return buffer_.data();}
    double* weights(){return buffer_.data()+w_;}
    const double* weights()const{return buffer_.data()+w_;}
    double* logs(){return buffer_.data()+4*w_;}
    const double* logs()const{return buffer_.data()+4*w_;}
    void restore_reciprocals(){for(int j=0;j<w_;++j)buffer_[2*w_+j]=bit_cast<double>(UINT64_MAX/(uint64_t)geometry_[j]);}
    uint64_t reduce_row(uint64_t x,int j)const{
        uint64_t m=geometry_[j],reciprocal=bit_cast<uint64_t>(buffer_[2*w_+j]);
        uint64_t quotient=(uint64_t)(((unsigned __int128)x*reciprocal)>>64);
        uint64_t remainder=x-quotient*m;return remainder-(remainder>=m?m:0);
    }
    double probability(int j)const{
        if(q16_==16)return exp2(-(j==w_-1?w_-1:j+1));
        if(j==0)return 1-q16_/16.;int k=j-1,t=w_-1;
        return q16_/16.*exp2(-(k==t-1?t-1:k+1));
    }
    double response(double n,double*slope=nullptr)const{
        double sum=0,g=0;for(int j=0;j<w_;++j){double l=logs()[j],e=exp(n*l);sum+=weights()[j]*.5*geometry_[j]*-expm1(n*l);g-=weights()[j]*.5*geometry_[j]*l*e;}
        if(slope)*slope=g;return sum;
    }
    double decode(double y)const{
        if(y<=0)return 0;if(y>=response(c.nmax))return c.nmax;
        double lo=0,hi=c.nmax,x=clamp(previous_,lo,hi);
        for(int it=0;it<80;++it){double slope=0,v=response(x,&slope),e=v-y;
            if(slope>0&&abs(e/slope)<1e-7){
                if(32*numeric_limits<double>::epsilon()*(1+abs(v))/slope>1e-4){
                    double l=0,h=c.nmax;for(int k=0;k<52;++k){double mid=.5*(l+h);if(response(mid)<y)l=mid;else h=mid;}return .5*(l+h);}
                return x;
            }
            if(e<0)lo=x;else hi=x;if(hi-lo<1e-7)return .5*(hi+lo);
            double next=slope>0?x-e/slope:.5*(lo+hi);if(!(next>lo&&next<hi)||next==x)next=.5*(lo+hi);x=next;
        }return .5*(lo+hi);
    }
public:
    LeanHfx(string label,Config cfg,vector<uint64_t> rows,int q16,uint64_t sh,uint64_t sn):Mechanism(label,cfg,sn),w_(rows.size()),q16_(q16),hashseed_(sh),geometry_(2*rows.size(),0),buffer_(5*rows.size()+1,0){
        if(w_<1||w_>32||q16_<1||q16_>16||(q16_!=16&&w_<2))throw runtime_error("invalid lean geometry");
        uint64_t offset=0;for(int j=0;j<w_;++j){if(rows[j]<4||rows[j]>UINT32_MAX||offset+rows[j]>UINT32_MAX)throw runtime_error("unsupported public cell count");geometry_[j]=rows[j];geometry_[j+w_]=offset;offset+=rows[j];weights()[j]=1;logs()[j]=log1p(-2*probability(j)/rows[j]);}
        bits_.assign((offset+63)/64,0);buffer_.back()=-1;restore_reciprocals();
        sensitivity=sqrt((double)c.K);sigma=analytic_gaussian_sigma(c.epsilon,c.delta,sensitivity);
    }
    int width()const override{return w_;}
    uint64_t bytes()const override{return sizeof(*this)+bits_.capacity()*8+geometry_.capacity()*4+buffer_.capacity()*8;}
    uint64_t payload_bytes()const{return bits_.capacity()*8;}
    uint64_t row(int j)const{return geometry_[j];}
    double raw_statistic()const override{return accumulate(buffer_.begin(),buffer_.begin()+w_,0.);}
    void update_hashed(uint64_t x,uint64_t y){
        int j;uint64_t index;
        if(q16_==16){j=min(ctz64(y),w_-1);index=geometry_[j+w_]+reduce_row(x,j);}
        else{
            int tail=w_==2?1:1+min(ctz64(y),w_-2);
            uint64_t gate=splitmix64(y^0x8A5CD789635D2DFFull)>>60;
            uint64_t accepted=gate>=(uint64_t)(16-q16_);j=tail*(int)accepted;
            if constexpr(ParallelRows){
                uint64_t head=reduce_row(x,0),other=geometry_[tail+w_]+reduce_row(x,tail);
                index=head+accepted*(other-head);
            }else index=geometry_[j+w_]+reduce_row(x,j);
        }
        uint64_t word=index>>6,mask=1ull<<(index&63),old=(bits_[word]>>(index&63))&1;
        bits_[word]^=mask;counts()[j]+=1.-2.*(double)old;
    }
    void update(uint64_t id,int)override{auto[x,y]=hash_pair(id,hashseed_);update_hashed(x,y);}
    bool design(double pilot){
        if(w_==1)return false;
        int grid=(int)round(log(clamp(pilot,1.,(double)c.nmax))/log((double)c.nmax)*255.);
        if(buffer_.back()==grid)return false;double n=exp(log((double)c.nmax)*grid/255.);
        double*g=buffer_.data()+2*w_,*d=g+w_,sum_g=0,sum_d=0;
        for(int j=0;j<w_;++j){double m=geometry_[j],l=logs()[j];g[j]=-.5*m*l*exp(n*l);d[j]=-.25*m*expm1(2*n*l);weights()[j]=1;sum_g+=g[j];sum_d+=d[j];}
        double lambda=(sum_d+sigma*sigma)/sum_g;
        // Active-set water filling. Once d_j/g_j > lambda, removing that
        // saturated coordinate decreases lambda; it never needs reactivation.
        // Exact for the diagonal-minus-rank-one local covariance surrogate,
        // not claimed to minimize the full finite-n covariance risk.
        for(int iteration=0;iteration<=w_;++iteration){
            bool removed=false;for(int j=0;j<w_;++j)if(weights()[j]&&d[j]>lambda*g[j]){weights()[j]=0;removed=true;}
            if(!removed)break;
            // Recompute rather than subtract nearly equal sums (saturation).
            sum_g=0;sum_d=0;for(int j=0;j<w_;++j)if(weights()[j]){sum_g+=g[j];sum_d+=d[j];}
            if(!(sum_g>0))throw runtime_error("water filling lost every coordinate");
            lambda=(sum_d+sigma*sigma)/sum_g;
            if(iteration==w_)throw runtime_error("water filling did not converge");
        }
        for(int j=0;j<w_;++j){weights()[j]=d[j]>0?min(1.,lambda*g[j]/d[j]):1.;if(weights()[j]<0||weights()[j]>1||!isfinite(weights()[j]))throw runtime_error("unbounded weight");}
        restore_reciprocals();buffer_.back()=grid;return true;
    }
    Estimate release(uint64_t j)override{
        if(j>1)design(previous_);double q=0;for(int k=0;k<w_;++k)q+=weights()[k]*counts()[k];
        double noisy=q+sigma*rng.normal(),estimate=decode(noisy);previous_=estimate;
        return{estimate,diagnostic_clean?decode(q):0.,noisy<0||noisy>response(c.nmax)};
    }
    double test_response(double n)const{return response(n);}
    double test_decode(double y)const{return decode(y);}
    double test_weight(int j)const{return weights()[j];}
    double test_count(int j)const{return buffer_[j];}
    uint64_t test_bitmap_weight()const{uint64_t total=0;for(uint64_t word:bits_)total+=popcount(word);return total;}
    uint64_t test_remainder(uint64_t x,int j)const{return reduce_row(x,j);}
};
