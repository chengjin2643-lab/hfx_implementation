#pragma once
#include "heterogeneous.hpp"

// Research implementation. Counts and clean diagnostics must not be published.
// The release is one bounded-weight scalar plus the original Gaussian noise.
template<bool ParallelRows=false,uint64_t ScanWords=0,bool IntegerCounts=false> class LeanHfx:public Mechanism {
    int w_,q16_;uint64_t hashseed_;
    vector<uint64_t> bits_;
    vector<uint32_t> geometry_; // rows and offsets
    vector<double> buffer_; // counts,weights,g/reciprocals,d,logs,grid
    double previous_=0;
    static uint64_t low_mask(unsigned k){return k==64?UINT64_MAX:((1ull<<k)-1);}
    uint64_t scan_column(int j)const{
        uint64_t start=geometry_[j+w_],end=start+geometry_[j],word=start>>6,last=end>>6,total=0;
        unsigned offset=start&63;
        if(word==last)return popcount((bits_[word]>>offset)&low_mask((unsigned)(end-start)));
        if(offset){total+=popcount(bits_[word]&(~0ull<<offset));++word;}
        for(;word+4<=last;word+=4)total+=popcount(bits_[word])+popcount(bits_[word+1])+popcount(bits_[word+2])+popcount(bits_[word+3]);
        for(;word<last;++word)total+=popcount(bits_[word]);
        if(end&63)total+=popcount(bits_[last]&low_mask(end&63));return total;
    }
    bool scanned(int j)const{return ScanWords>0&&((uint64_t)geometry_[j]+63)/64<=ScanWords;}
    // Integer mode stores the count's uint64 representation in the existing
    // eight-byte slots. bit_cast preserves representation without aliasing UB.
    double stored_count(int j)const{if constexpr(IntegerCounts)return (double)bit_cast<uint64_t>(buffer_[j]);else return buffer_[j];}
    void set_count(int j,uint64_t v){if constexpr(IntegerCounts)buffer_[j]=bit_cast<double>(v);else buffer_[j]=(double)v;}
    void add_count(int j,uint64_t old){if constexpr(IntegerCounts){uint64_t v=bit_cast<uint64_t>(buffer_[j]);buffer_[j]=bit_cast<double>(v+1-2*old);}else buffer_[j]+=1.-2.*(double)old;}
    double current_count(int j)const{return scanned(j)?(double)scan_column(j):stored_count(j);}
    void refresh_counts(){if constexpr(ScanWords>0)for(int j=0;j<w_;++j)if(scanned(j))set_count(j,scan_column(j));}
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
    void toggle_bit(int column, uint64_t index) {
        uint64_t& word = bits_[index >> 6];
        const uint64_t mask = uint64_t{1} << (index & 63);
        const uint64_t before = word;
        word = before ^ mask;
        if constexpr (ScanWords != UINT64_MAX) {
            if (!scanned(column)) add_count(column, (before & mask) != 0);
        }
    }
    void update_single_column(uint64_t row_hash) {
        const uint64_t rows = geometry_[0];
        const uint64_t index = has_single_bit(rows)
            ? (row_hash & (rows - 1)) : reduce_row(row_hash, 0);
        toggle_bit(0, index);
    }
    double probability(int j)const{
        if(q16_==16)return exp2(-(j==w_-1?w_-1:j+1));
        if(j==0)return 1-q16_/16.;int k=j-1,t=w_-1;
        return q16_/16.*exp2(-(k==t-1?t-1:k+1));
    }
    double response(double n,double*slope=nullptr)const{
        double sum = 0, derivative = 0;
        for (int j = 0; j < w_; ++j) {
            const double l = logs()[j];
            const double scale = weights()[j] * .5 * geometry_[j];
            sum += scale * -expm1(n * l);
            if (slope) derivative -= scale * l * exp(n * l);
        }
        if (slope) *slope = derivative;
        return sum;
    }
    double decode(double y, double upper_response)const{
        if(y<=0)return 0;if(y>=upper_response)return c.nmax;
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
    double raw_statistic()const override{double total=0;for(int j=0;j<w_;++j)total+=current_count(j);return total;}
    void update_hashed(uint64_t x,uint64_t y){
        int j;uint64_t index;
        if (w_ == 1) {
            update_single_column(x);
            return;
        }
        if(q16_==16){j=min(ctz64(y),w_-1);index=geometry_[j+w_]+reduce_row(x,j);}
        else {
            const int tail = w_ == 2 ? 1 : 1 + min(ctz64(y), w_ - 2);
            const uint64_t gate = splitmix64(y ^ 0x8A5CD789635D2DFFull) >> 60;
            const bool use_tail = gate >= static_cast<uint64_t>(16 - q16_);
            j = use_tail ? tail : 0;
            if constexpr (ParallelRows || ScanWords == UINT64_MAX) {
                // Compute both addresses while the routing hash is in flight.
                const uint64_t head_rows = geometry_[0];
                const uint64_t head_index = has_single_bit(head_rows)
                    ? (x & (head_rows - 1)) : reduce_row(x, 0);
                const uint64_t tail_index = geometry_[tail + w_] + reduce_row(x, tail);
                // A full-word mask selects the address without branching on a random hash bit.
                const uint64_t tail_mask = uint64_t{0} - static_cast<uint64_t>(use_tail);
                index = (head_index & ~tail_mask) | (tail_index & tail_mask);
            } else {
                index = geometry_[j + w_] + reduce_row(x, j);
            }
        }
        toggle_bit(j, index);
    }
    void update(uint64_t id,int)override{
        if (w_ == 1) {
            update_single_column(hash_pair(id, hashseed_).first);
            return;
        }
        const auto [x, y] = hash_pair(id, hashseed_);
        update_hashed(x, y);
    }
    template<bool PowerOfTwo> void update_single_scan(uint64_t id) {
        static_assert(ScanWords==UINT64_MAX);
        const uint64_t x = splitmix64(id^(hashseed_*GOLDEN64));
        uint64_t index;
        if constexpr(PowerOfTwo) index = x & (uint64_t(geometry_[0])-1);
        else index = reduce_row(x,0);
        toggle_bit(0,index);
    }
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
        if(j>1)design(previous_);refresh_counts();double q=0;for(int k=0;k<w_;++k)q+=weights()[k]*stored_count(k);
        const double upper_response = response(c.nmax);
        const double noisy = q + sigma * rng.normal();
        const double estimate = decode(noisy, upper_response);
        previous_ = estimate;
        return {estimate, diagnostic_clean ? decode(q, upper_response) : 0.,
                noisy < 0 || noisy > upper_response};
    }
    double test_response(double n)const{return response(n);}
    double test_decode(double y)const{return decode(y, response(c.nmax));}
    double test_weight(int j)const{return weights()[j];}
    double test_count(int j)const{return current_count(j);}
    uint64_t test_bitmap_weight()const{uint64_t total=0;for(uint64_t word:bits_)total+=popcount(word);return total;}
    uint64_t test_remainder(uint64_t x,int j)const{return reduce_row(x,j);}
};
