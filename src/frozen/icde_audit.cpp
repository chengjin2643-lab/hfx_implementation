#define main snapshot_reference_main
#include "snapshot_reference.cpp"
#undef main
#include <map>
#include <array>
#include <functional>

// All benchmark mechanisms consume public fixed slots. Stored signs are used
// by the evaluator and direction-oracle split references, never by HFX/GMod.
struct SmallRng {
    uint64_t state; bool spare_ready=false; double spare=0;
    explicit SmallRng(uint64_t s):state(s){}
    uint64_t next(){ state+=GOLDEN64; uint64_t z=state; z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull; return z^(z>>31); }
    double uniform(){return ((next()>>12)+0.5)*0x1p-52;}
    double normal(){if(spare_ready){spare_ready=false;return spare;} double r=sqrt(-2*log(uniform())), a=6.283185307179586*uniform(); spare=r*sin(a);spare_ready=true;return r*cos(a);}
    double laplace(double b){double x=uniform()-0.5;return -b*copysign(log1p(-2*abs(x)),x);}
};
struct Config {
    string mode="synthetic", dataset="synthetic", pattern="fresh", schedule="fixed-k", methods="hfx,gmod,gmod_mec,hx_native,hx_packed,exact,hss,ajs,fm,ll,hll", input, output="results/run";
    uint64_t N=10000,T=2010000,K=200,B=10000,nmax=20000,hot=0,seed=46101601;
    int mem=32,width=0,trials=10,trial_start=0,cap_q=4;
    double epsilon=1,delta=1e-6,beta=0.025,completion=-1;
    bool trace=true;
};
struct Counters {
    long double sq=0,rel=0,clean_sq=0,clean_rel=0,service_sq=0;
    uint64_t all=0,pos=0,service_n=0;
    void add(double x,double c,uint64_t n){long double e=x-(long double)n, ec=c-(long double)n;sq+=e*e;clean_sq+=ec*ec;++all;if(n){rel+=e*e/n/n;clean_rel+=ec*ec/n/n;++pos;}}
    void service(double x,uint64_t count,long double sum,long double sqsum){long double y=x;service_sq+=max(0.L,count*y*y-2*y*sum+sqsum);service_n+=count;}
};
struct Estimate {double value=0,clean=0;bool clipped=false;};
class Mechanism {
public:
    string name,contract="remove_all_ID",calibration="analytic_gaussian";Config c;SmallRng rng;
    uint64_t releases=0,clipped=0,stopped=0,max_nodes=0,processed=0;bool resource_exceeded=false,diagnostic_clean=true;double held=0,sigma=0,sensitivity=0;
    Counters metrics;double seconds=0,release_seconds=0;uint64_t noise_seed;
    Mechanism(string n,const Config& cfg,uint64_t seed):name(n),c(cfg),rng(seed),noise_seed(seed){}
    virtual ~Mechanism()=default;
    virtual void update(uint64_t id,int sign)=0;
    virtual Estimate release(uint64_t j)=0;
    virtual uint64_t bytes()const=0;
    virtual int width()const{return 0;}
    virtual double raw_statistic()const{throw runtime_error("raw statistic unavailable");}
    virtual double decode_raw(double)const{throw runtime_error("raw decoder unavailable");}
    double bounded(double x)const{return clamp(x,0.,(double)c.nmax);}
};
static uint64_t FULL_STATE_HFX_ROWS=0;
static int FULL_STATE_GMOD_WIDTH=0;
class HfxMethod:public Mechanism {
    PackedGeometricHfx sketch;int w;
public:
    HfxMethod(string n,const Config& a,uint64_t sh,uint64_t sn,int forced=0):Mechanism(n,a,sn),sketch(a.mem,forced?forced:select_public_width(a.mem,a.nmax),a.nmax,FULL_STATE_HFX_ROWS),w(forced?forced:select_public_width(a.mem,a.nmax)){sensitivity=sqrt((double)a.K);sigma=analytic_gaussian_sigma(a.epsilon,a.delta,sensitivity);hashseed=sh;}
    uint64_t hashseed;
    void update(uint64_t id,int)override{auto [x,y]=hash_pair(id,hashseed);update_hashed(x,y);}
    void update_hashed(uint64_t x,uint64_t y){sketch.update_hashed(x,y);}
    pair<int,double> update_hashed_record(uint64_t x,uint64_t y){return sketch.update_hashed_record(x,y);}
    uint64_t rows()const{return sketch.rows();}
    Estimate release(uint64_t)override{double x=sketch.weight()+sigma*rng.normal();return {sketch.invert(x),diagnostic_clean?sketch.invert(sketch.weight()):0.,x<0||x>sketch.max_state()};}
    uint64_t bytes()const override{return sizeof(*this)+sketch.full_state_bytes()-sizeof(sketch);}
    int width()const override{return w;}
    double raw_statistic()const override{return sketch.weight();}
    double decode_raw(double x)const override{return sketch.invert(x);}
};
class ParityState {
    shared_ptr<AllocationTracker> tracker=make_shared<AllocationTracker>();
    using Set=unordered_set<uint64_t,hash<uint64_t>,equal_to<uint64_t>,CountingAllocator<uint64_t>>;
    Set active{0,hash<uint64_t>{},equal_to<uint64_t>{},CountingAllocator<uint64_t>{tracker}};
public:
    int toggle(uint64_t x){auto it=active.find(x);if(it==active.end()){active.insert(x);return 1;}active.erase(it);return -1;}
    uint64_t count()const{return active.size();}
    uint64_t bytes()const{return sizeof(*this)+sizeof(AllocationTracker)+tracker->peak;}
};
// AJS balanced b-ary decoding. Each node appears on one contiguous interval of
// release indices; discarding inactive noise keeps O(b log K) live mechanism state.
struct TreeNoise {
    using NodeKey=pair<uint64_t,uint64_t>;
    using NodeValue=pair<const NodeKey,double>;
    shared_ptr<AllocationTracker> tracker=make_shared<AllocationTracker>();
    int b;uint64_t leaves=1;int h=0;SmallRng rng;
    map<NodeKey,double,less<NodeKey>,CountingAllocator<NodeValue>> live{less<NodeKey>{},CountingAllocator<NodeValue>{tracker}};
    uint64_t peak=0;
    explicit TreeNoise(uint64_t K,int branch,uint64_t s):b(branch),rng(s){while(leaves<K){leaves*=b;++h;}}
    using Terms=map<pair<uint64_t,uint64_t>,int>;
    void prefix(Terms& z,uint64_t start,uint64_t size,uint64_t len,int sign){
        if(!len)return;
        if(len==size){z[{start,size}]+=sign;return;}
        if(len>size/2){z[{start,size}]+=sign;suffix(z,start,size,size-len,-sign);return;}
        uint64_t sub=size/b, full=len/sub, rem=len%sub;
        for(uint64_t i=0;i<full;++i)z[{start+i*sub,sub}]+=sign;
        if(rem)prefix(z,start+full*sub,sub,rem,sign);
    }
    void suffix(Terms& z,uint64_t start,uint64_t size,uint64_t len,int sign){
        if(!len)return;
        if(len==size){z[{start,size}]+=sign;return;}
        if(len>size/2){z[{start,size}]+=sign;prefix(z,start,size,size-len,-sign);return;}
        uint64_t sub=size/b,full=len/sub,rem=len%sub;
        for(uint64_t i=0;i<full;++i)z[{start+size-(i+1)*sub,sub}]+=sign;
        if(rem)suffix(z,start+size-(full+1)*sub,sub,rem,sign);
    }
    Terms terms(uint64_t j){Terms t;prefix(t,0,leaves,j,1);return t;}
    double noise(uint64_t j){auto t=terms(j);double v=0;for(auto [key,sign]:t){if(!sign)continue;auto it=live.find(key);if(it==live.end())it=live.emplace(key,rng.normal()).first;v+=sign*it->second;}for(auto it=live.begin();it!=live.end();){if(!t.count(it->first)||!t[it->first])it=live.erase(it);else ++it;}peak=max<uint64_t>(peak,live.size());return v;}
    // Exact maximum squared sensitivity for unrestricted sampled membership
    // q[0]=0, q[j] in {0,1}; increments after K are fixed to zero.
    array<int64_t,4> dp(uint64_t start,uint64_t size,uint64_t K,bool used){
        constexpr int64_t neg=-1000000000000ll;
        if(start>=K)return {0,neg,neg,0};
        if(size==1)return {0,used?1:0,used?1:0,0};
        uint64_t sub=size/b;array<int64_t,4> comb{0,neg,neg,0};
        for(int i=0;i<b;++i){uint64_t first=i<b/2?start+(i+1)*sub:start+size/2+1;auto child=dp(start+i*sub,sub,K,i!=b/2&&first<=K);array<int64_t,4> next{neg,neg,neg,neg};for(int l=0;l<2;++l)for(int r=0;r<2;++r)for(int k=0;k<2;++k)next[2*l+r]=max(next[2*l+r],comb[2*l+k]+child[2*k+r]);comb=next;}
        if(used){++comb[1];++comb[2];}return comb;
    }
    double sens(uint64_t K){auto x=dp(0,leaves,K,leaves/2+1<=K);return sqrt((double)max(x[0],x[1]));}
    uint64_t bytes()const{return sizeof(*this)+sizeof(AllocationTracker)+tracker->peak;}
};
class ExactMethod:public Mechanism {
    ParityState parity;TreeNoise tree;bool use_tree,svt;uint64_t sk=0,count=1;double eps1=0,tau=0,raw_out=0,threshold=0;
public:
    ExactMethod(string n,const Config& a,uint64_t sn):Mechanism(n,a,sn),tree(a.K,7,sn+193),use_tree(n=="ajs"),svt(n=="hss"){
        if(use_tree){sensitivity=tree.sens(a.K);sigma=analytic_gaussian_sigma(a.epsilon,a.delta,sensitivity);calibration="AJS_b7_subtraction_exact_S1K_sensitivity";}
        else if(svt){double e=min(a.epsilon,1.-1e-6);sk=(uint64_t)floor(pow((double)a.T*e/(36*sqrt(log(1/a.delta))*log((double)a.K/a.beta)),2./3.))+1;sk=max<uint64_t>(2,sk);eps1=e/(4*sqrt(2*sk*log(1/a.delta)));threshold=16/eps1*log(2*a.K/a.beta);tau=rng.laplace(2/eps1);raw_out=rng.laplace(1/eps1);held=bounded(raw_out);calibration="HSS_Algorithm1_genericQ_VequalsT";}
        else{sensitivity=sqrt((double)a.K);sigma=analytic_gaussian_sigma(a.epsilon,a.delta,sensitivity);}
    }
    void update(uint64_t id,int)override{parity.toggle(id);}
    Estimate release_query(double q,uint64_t j){
        double x;
        if(svt){if(!stopped){if(abs(raw_out-q)+rng.laplace(4/eps1)>threshold+tau){++count;tau=rng.laplace(2/eps1);raw_out=q+rng.laplace(1/eps1);}if(count>=sk)stopped=j;}x=raw_out;}
        else x=q+sigma*(use_tree?tree.noise(j):rng.normal());
        return {bounded(x),q,x<0||x>c.nmax};
    }
    Estimate release(uint64_t j)override{return release_query((double)parity.count(),j);}
    uint64_t bytes()const override{return sizeof(*this)+parity.bytes()-sizeof(parity)+(use_tree?tree.bytes()-sizeof(tree):0);}
    uint64_t svt_budget()const{return sk;} double svt_eps()const{return eps1;}double svt_threshold()const{return threshold;}
    uint64_t svt_refreshes()const{return count-1;}
    double raw_statistic()const override{return (double)parity.count();}
    double decode_raw(double x)const override{return bounded(x);}
};
class OfflineReference:public Mechanism {
public:
    uint64_t truth=0;
    OfflineReference(const Config&a,uint64_t sn):Mechanism("exact_query_offline",a,sn){calibration="offline_truth_analytic_Gaussian_reference";sensitivity=sqrt((double)a.K);sigma=analytic_gaussian_sigma(a.epsilon,a.delta,sensitivity);}
    void update(uint64_t,int sign)override{truth+=sign;}
    Estimate release(uint64_t)override{double x=truth+sigma*rng.normal();return{bounded(x),(double)truth,x<0||x>c.nmax};}
    uint64_t bytes()const override{return 0;}
};
// G=2 is sign-independent. All column zero-count statistics are released
// jointly with sensitivity sqrt(K), then the published MEC rule is applied.
static uint64_t PUBLIC_AUDIT_ROWS_OVERRIDE=0;
static int public_gmod_width(int mem,uint64_t nmax){
    int best=1;double slope=-1;
    for(int width=1;width<=32;++width){
        uint64_t rows=max<uint64_t>(4,mem*1024ull*8/width);double candidate=0;
        for(int j=0;j<width;++j){double p=exp2(-(j==width-1?width-1:j+1));candidate+=.5*p*exp(-(double)nmax*p/rows);}
        if(candidate>slope){slope=candidate;best=width;}
    }
    return best;
}
class ReversibleReference:public Mechanism {
    bool half,native,mec;int w=32;uint64_t m,hashseed;vector<uint64_t> packed;vector<uint32_t> raw,counters,seeds;vector<double> column;uint64_t occupied=0;ParityState adapter;
    bool bit(uint64_t i)const{return native?raw[i]!=0:((packed[i>>6]>>(i&63))&1);}
    void toggle(uint64_t i){if(native)raw[i]^=1;else packed[i>>6]^=1ull<<(i&63);}
    static double dyadic(int exponent){return bit_cast<double>((uint64_t)(1023-exponent)<<52);}
    double total_response(double n,double*derivative=nullptr)const{double p=0,g=0;for(int j=0;j<w;++j){double prob=dyadic(j==w-1?w-1:j+1),v=-n*prob/m;p+=-expm1(v);if(derivative)g+=.5*prob*exp(v);}if(derivative)*derivative=g;return p*m/2.;}
    double inverse_total(double weight)const{
        if(weight<=0)return 0.;if(weight>=total_response((double)c.nmax))return (double)c.nmax;
        if(w==1)return -(double)m*log1p(-2*weight/m);
        double lo=0,hi=c.nmax,x=clamp(held,lo,hi);
        for(int it=0;it<80;++it){double slope=0,value=total_response(x,&slope),error=value-weight;
            if(slope>0&&abs(error/slope)<1e-7){if(32*numeric_limits<double>::epsilon()*(1+abs(value))/slope<1e-4)return x;break;}
            if(error<0)lo=x;else hi=x;if(hi-lo<1e-7)return .5*(lo+hi);
            double next=slope>0?x-error/slope:.5*(lo+hi);if(!(next>lo&&next<hi)||next==x)next=.5*(lo+hi);x=next;
        }
        lo=0;hi=c.nmax;for(int it=0;it<52;++it){double mid=.5*(lo+hi);if(total_response(mid)<weight)lo=mid;else hi=mid;}return .5*(lo+hi);
    }
    double estimate(const vector<double>& z,double d,double noisevar)const{
        double total=accumulate(z.begin(),z.end(),0.),coarse=inverse_total(total);
        double lin=-(double)m*log(max(1e-300,1-clamp(d,0.,nextafter((double)m,0.))/(double)m));
        if(half&&!mec)return bounded(coarse<=3*m?lin:coarse);
        double num=0,den=0;
        for(int j=0;j<w;++j){double p=exp2(-(j==w-1?w-1:j+1));double y=clamp(z[j],0.,(double)m);if(y>=m*.5)continue;double a=1-2*y/m;double logterm=half?-p/m:log1p(-p/m);double nj=log(a)/logterm;
            double pilot_a=exp(logterm*max(1.,coarse));
            double derivative=-.5*m*logterm*pilot_a;double variance=(.25*m*(1-pilot_a*pilot_a)+noisevar)/max(1e-290,derivative*derivative);variance=max(1e-12,variance);num+=nj/variance;den+=1/variance;
        }
        double est=den?num/den:coarse;
        if(half){double a=exp(-min(est/(double)m,700.));double varlin=m*(1/a-est/m-1)+noisevar/max(1e-290,a*a);if(varlin<(den?1/den:1e300))est=lin;}
        return bounded(est);
    }
public:
    // Bitmap-only Half-Xor follows the published phi path, without its
    // optional EZ counter. At matched geometry it is the G=2 scalar path.
    ReversibleReference(string n,const Config& a,uint64_t sh,uint64_t sn):Mechanism(n,a,sn),half(n.rfind("hx_",0)==0&&n.rfind("hx_phi",0)!=0),native(n=="hx_native"),mec(n!="gmod"&&n!="hx_ez"&&n!="gmod_public"&&n!="hx_phi"&&n!="hx_phi_public"),w((n=="gmod_public"||n=="gmod_mec_public"||n=="hx_phi_public")?public_gmod_width(a.mem,a.nmax):32),hashseed(sh){
        if(FULL_STATE_GMOD_WIDTH)w=FULL_STATE_GMOD_WIDTH;
        uint64_t budget=a.mem*1024ull;m=native?budget/(4*w+4+12):(half?budget/(w/8+4):budget*8/w);m=max<uint64_t>(4,m);
        if(PUBLIC_AUDIT_ROWS_OVERRIDE)m=PUBLIC_AUDIT_ROWS_OVERRIDE;
        if(native){raw.assign(m*w,0);seeds.assign(3*m,0);}else packed.assign((m*w+63)/64,0);
        if(half)counters.assign(m,0);
        column.assign(w,0);sensitivity=sqrt(a.K*(half?2.:1.));sigma=analytic_gaussian_sigma(a.epsilon,a.delta,sensitivity);
        calibration=half?"central_DP_joint_D_and_bitmap_columns":"central_DP_joint_G2_columns";
        // Scalar mode does not use EZ occupied counts. Reuse that existing
        // field for a public exact-remainder reciprocal, with no added bytes.
        if(!half&&!mec)occupied=UINT64_MAX/m;
    }
    void update(uint64_t id,int)override{
        auto [h1,h2]=hash_pair(id,hashseed);if(!half){update_hashed(h1,h2);return;}uint64_t row=h1%m;int j=min(ctz64(h2),w-1);
        if(half){int sign=adapter.toggle(id);uint64_t r=splitmix64(h1^0x13579BDFull)%m;bool was=counters[r]!=0;if(sign>0)++counters[r];else --counters[r];bool now=counters[r]!=0;occupied+=now&&!was;occupied-=was&&!now;}
        if((splitmix64(h2^0x8A5CD789635D2DFFull)>>63)==0)return;
        uint64_t idx=row*w+j;bool old=bit(idx);toggle(idx);column[j]+=old?-1:1;
    }
    void update_hashed(uint64_t h1,uint64_t h2){
        if(half)throw runtime_error("hashed batch path is bitmap-only");
        if(!mec){
            uint64_t accepted=splitmix64(h2^0x8A5CD789635D2DFFull)>>63;
            uint64_t quotient=(uint64_t)(((unsigned __int128)h1*occupied)>>64),row=h1-quotient*m;row-=row>=m?m:0;
            int j=min(ctz64(h2),w-1)*(int)accepted;uint64_t idx=(row*w+j)*(uint64_t)accepted;
            uint64_t mask=(1ull<<(idx&63))*accepted,old=(packed[idx>>6]>>(idx&63))&1;packed[idx>>6]^=mask;
            column[j]+=(double)accepted*(1.-2.*(double)old);return;
        }
        if((splitmix64(h2^0x8A5CD789635D2DFFull)>>63)==0)return;
        uint64_t idx=(h1%m)*w+min(ctz64(h2),w-1);bool old=bit(idx);toggle(idx);column[idx%w]+=old?-1:1;
    }
    Estimate release(uint64_t)override{
        if(!mec&&!half){
            // Same scalar query and addition order; no temporary noisy vector.
            // Also record clipping before inversion, rather than after clamping.
            double noisy=column[0]+sigma*rng.normal();for(int j=1;j<w;++j)noisy+=column[j];
            double v=inverse_total(noisy),clean=diagnostic_clean?inverse_total(accumulate(column.begin(),column.end(),0.)):0.;
            held=bounded(v);return {held,bounded(clean),noisy<0||noisy>total_response(c.nmax)};
        }
        vector<double> noisy=column;double z=0;bool clip=false;
        // Scalar GMod base releases total weight; MEC releases all columns.
        if(!mec){z=sigma*rng.normal();noisy[0]+=z;}else for(double& x:noisy){x+=sigma*rng.normal();clip|=x<0||x>m;}
        double d=occupied+(half?sigma*rng.normal():0);
        double v=!mec&&!half?inverse_total(accumulate(noisy.begin(),noisy.end(),0.)):estimate(noisy,d,sigma*sigma);
        double clean=diagnostic_clean?(!mec&&!half?inverse_total(accumulate(column.begin(),column.end(),0.)):estimate(column,occupied,0)):0.;
        if(!half&&!mec)held=bounded(v);
        return {bounded(v),bounded(clean),clip||v<0||v>c.nmax};
    }
    uint64_t bytes()const override{return sizeof(*this)+packed.capacity()*8+raw.capacity()*4+counters.capacity()*4+seeds.capacity()*4+column.capacity()*8+(half?adapter.bytes()-sizeof(adapter):0);}
    int width()const override{return w;}
    uint64_t rows()const{return m;}
    double raw_statistic()const override{return accumulate(column.begin(),column.end(),0.);}
    double decode_raw(double x)const override{return bounded(inverse_total(x));}
    vector<double> audit_statistics()const{auto x=column;x.push_back(occupied);return x;}
    // Scalar G=2 engineering control: defer a public bitmap prefix and count
    // it at releases. No extra buffer, hash/layout, sensitivity or noise change.
    template<uint64_t ScanWords,bool IntegerCounts=false> void update_hashed_scan(uint64_t h1,uint64_t h2){
        if constexpr(ScanWords==0&&!IntegerCounts){update_hashed(h1,h2);return;}
        if(half||mec)throw runtime_error("prefix scan expects scalar GMod");
        uint64_t accepted=splitmix64(h2^0x8A5CD789635D2DFFull)>>63;
        uint64_t quotient=(uint64_t)(((unsigned __int128)h1*occupied)>>64),row=h1-quotient*m;row-=row>=m?m:0;
        int j=min(ctz64(h2),w-1)*(int)accepted;uint64_t idx=(row*w+j)*accepted,word=idx>>6;
        uint64_t mask=(1ull<<(idx&63))*accepted,before=packed[word];packed[word]=before^mask;
        if(word>=ScanWords){uint64_t old=(before>>(idx&63))&1;
            if constexpr(IntegerCounts){uint64_t v=bit_cast<uint64_t>(column[j]);column[j]=bit_cast<double>(v+accepted*(1-2*old));}
            else column[j]+=(double)accepted*(1.-2.*(double)old);}
    }
    template<uint64_t ScanWords,bool IntegerCounts=false> void update_id_scan(uint64_t id){auto[x,y]=hash_pair(id,hashseed);update_hashed_scan<ScanWords,IntegerCounts>(x,y);}
    template<uint64_t ScanWords> double prefix_weight()const{
        uint64_t words=min<uint64_t>(ScanWords,packed.size()),total=0,i=0;
        for(;i+4<=words;i+=4)total+=popcount(packed[i])+popcount(packed[i+1])+popcount(packed[i+2])+popcount(packed[i+3]);
        for(;i<words;++i)total+=popcount(packed[i]);return (double)total;
    }
    template<bool IntegerCounts> double scalar_column(int j)const{if constexpr(IntegerCounts)return (double)bit_cast<uint64_t>(column[j]);else return column[j];}
    template<uint64_t ScanWords,bool IntegerCounts=false> double raw_scan_statistic()const{double total=prefix_weight<ScanWords>();for(int j=0;j<w;++j)total+=scalar_column<IntegerCounts>(j);return total;}
    template<uint64_t ScanWords,bool IntegerCounts=false> Estimate release_scan(uint64_t j){
        if constexpr(ScanWords==0&&!IntegerCounts)return ReversibleReference::release(j);
        double prefix=prefix_weight<ScanWords>(),noisy=prefix+scalar_column<IntegerCounts>(0)+sigma*rng.normal();for(int k=1;k<w;++k)noisy+=scalar_column<IntegerCounts>(k);
        double v=inverse_total(noisy),clean=diagnostic_clean?inverse_total(raw_scan_statistic<ScanWords,IntegerCounts>()):0.;
        held=bounded(v);return {held,bounded(clean),noisy<0||noisy>total_response(c.nmax)};
    }
};
template<class Sketch>class SplitMethod:public Mechanism {
    Sketch sketch;uint64_t hashseed;double jump;
public:
    SplitMethod(string n,const Config&a,uint64_t sh,uint64_t sn,double s):Mechanism(n,a,sn),sketch(a.mem),hashseed(sh),jump(s){contract="fresh_interval_ID_direction_oracle";sensitivity=s*sqrt(2.*a.K);sigma=analytic_gaussian_sigma(a.epsilon,a.delta,sensitivity);}
    void update(uint64_t id,int sign)override{auto[x,y]=hash_pair(id,hashseed);sketch.update(x,y,sign);}
    Estimate release(uint64_t)override{double x=sketch.add_state()+sigma*rng.normal(),y=sketch.del_state()+sigma*rng.normal();return {bounded(sketch.invert(x,c.T)-sketch.invert(y,c.T)),diagnostic_clean?bounded(sketch.invert(sketch.add_state(),c.T)-sketch.invert(sketch.del_state(),c.T)):0.,x<0||y<0||x>sketch.max_state()||y>sketch.max_state()};}
    uint64_t bytes()const override{return sizeof(*this)+sketch.full_state_bytes()-sizeof(sketch);}
};
class CapTreeMethod:public Mechanism {
    uint64_t hashseed,m,R;int w=32,H,q;vector<uint32_t> bits,audit_bits;vector<uint64_t> load;vector<double> U,P;double weight=0,noisy=0,audit_weight=0;uint64_t t=0;
    uint64_t get_load(uint64_t idx)const{uint64_t bit=idx*q,word=bit>>6;int off=bit&63;uint64_t x=load[word]>>off;if(off+q>64)x|=load[word+1]<<(64-off);return x&R;}
    void set_load(uint64_t idx,uint64_t x){uint64_t bit=idx*q,word=bit>>6;int off=bit&63;load[word]=(load[word]&~(R<<off))|(x<<off);if(off+q>64){int high=off+q-64;uint64_t mask=(1ull<<high)-1;load[word+1]=(load[word+1]&~mask)|(x>>(64-off));}}
public:
    uint64_t filtered=0;
    CapTreeMethod(const Config&a,uint64_t sh,uint64_t sn):Mechanism("cap_tree_q"+to_string(a.cap_q),a,sn),hashseed(sh),m(max<uint64_t>(4,(a.mem*1024ull*8-2*(bit_width(a.T)+1)*64)/(32*(1+a.cap_q)))),R((1ull<<a.cap_q)-1),H(bit_width(a.T)),q(a.cap_q),bits(m,0),audit_bits(m,0),load((m*w*q+63)/64,0),U(H+1,0),P(H+1,0){contract="endpoint_add_remove_only_full_T_outputs";calibration="original_cap_tree_theorem";double rho=pow(sqrt(log(1/a.delta)+a.epsilon)-sqrt(log(1/a.delta)),2);sensitivity=sqrt(4.*(R+1)*H);sigma=sqrt(2.*(R+1)*H/rho);}
    void update(uint64_t id,int)override{++t;auto[x,y]=hash_pair(id,hashseed);uint64_t row=x%m;int j=min(ctz64(y),w-1);uint64_t idx=row*w+j;bool audit_old=(audit_bits[row]>>j)&1;audit_bits[row]^=1u<<j;audit_weight+=audit_old?-1:1;double inc=0;uint64_t v=get_load(idx);if(v<R){set_load(idx,v+1);bool old=(bits[row]>>j)&1;bits[row]^=1u<<j;inc=old?-1:1;weight+=inc;}else ++filtered;int level=countr_zero(t);for(int i=0;i<level;++i){inc+=U[i];noisy-=P[i];U[i]=P[i]=0;}U[level]=inc;P[level]=inc+sigma*rng.normal();noisy+=P[level];}
    double inverse(double z)const{double lo=0,hi=c.nmax;for(int it=0;it<52;++it){double n=(lo+hi)/2,p=0;for(int j=0;j<w;++j){double pr=exp2(-(j==w-1?w-1:j+1));p+=-expm1(n*log1p(-2*pr/m));}p*=m/2.;if(p<z)lo=n;else hi=n;}return (lo+hi)/2;}
    Estimate release(uint64_t)override{return{inverse(noisy),inverse(weight),noisy<0||noisy>m*w/2.};}
    uint64_t bytes()const override{return sizeof(*this)+bits.capacity()*4+load.capacity()*8+(U.capacity()+P.capacity())*8;}
    int width()const override{return w;}
    double state_bias()const{return weight-audit_weight;}
    uint64_t unmatched()const{uint64_t z=0;for(uint64_t i=0;i<m;++i)z+=popcount(bits[i]^audit_bits[i]);return z;}
    int depth()const{return H;}uint64_t cap()const{return R;}
};
class SignedTreeMethod:public Mechanism {
    vector<double> U,P;uint64_t t=0;double noisy=0,truth=0;
public:
    SignedTreeMethod(const Config&a,uint64_t sn):Mechanism("signed_count_tree",a,sn),U(bit_width(a.T)+1,0),P(bit_width(a.T)+1,0){contract="signed_event_add_remove_full_T_outputs";sensitivity=sqrt((double)bit_width(a.T));sigma=analytic_gaussian_sigma(a.epsilon,a.delta,sensitivity);}
    void update(uint64_t,int sign)override{++t;truth+=sign;double inc=sign;int l=countr_zero(t);for(int i=0;i<l;++i){inc+=U[i];noisy-=P[i];U[i]=P[i]=0;}U[l]=inc;P[l]=inc+sigma*rng.normal();noisy+=P[l];}
    Estimate release(uint64_t)override{return {bounded(noisy),truth,noisy<0||noisy>c.nmax};}
    uint64_t bytes()const override{return sizeof(*this)+(U.capacity()+P.capacity())*8;}
};
static unique_ptr<Mechanism> make_mechanism(string n,const Config&a,uint64_t sh,uint64_t sn){
    if(n=="hfx")return make_unique<HfxMethod>(n,a,sh,sn,a.width);
    if(n=="uniform")return make_unique<HfxMethod>(n,a,sh,sn,1);
    if(n=="fixed32")return make_unique<HfxMethod>(n,a,sh,sn,32);
    if(n=="exact"||n=="hss"||n=="ajs")return make_unique<ExactMethod>(n,a,sn);
    if(n=="exact_query_offline")return make_unique<OfflineReference>(a,sn);
    if(n=="gmod"||n=="gmod_mec"||n=="gmod_public"||n=="gmod_mec_public"||n=="hx_native"||n=="hx_packed"||n=="hx_ez"||n=="hx_phi"||n=="hx_phi_public")return make_unique<ReversibleReference>(n,a,sh,sn);
    if(n=="fm")return make_unique<SplitMethod<FMSplit>>(n,a,sh,sn,32);
    if(n=="ll")return make_unique<SplitMethod<LLSplit>>(n,a,sh,sn,64);
    if(n=="hll")return make_unique<SplitMethod<HLLSplit>>(n,a,sh,sn,1);
    if(n=="cap")return make_unique<CapTreeMethod>(a,sh,sn);
    if(n=="signed_count_tree")return make_unique<SignedTreeMethod>(a,sn);
    throw runtime_error("unknown method "+n);
}
struct Stream {
    Config c;ifstream input;uint64_t t=0,truth=0;unique_ptr<LifecycleGenerator> lifecycle;
    Stream(const Config&a):c(a){if(!a.input.empty()){input.open(a.input,ios::binary);char magic[8];uint64_t n;input.read(magic,8);input.read((char*)&n,8);if(!input||string(magic,8)!="HFXSTRM1"||n!=a.T)throw runtime_error("invalid real binary");}else if(a.completion>=0)lifecycle=make_unique<LifecycleGenerator>(a.T,a.completion);}
    Event next(){++t;if(input.is_open()){uint64_t id;int8_t sign;input.read((char*)&id,8);input.read((char*)&sign,1);if(!input)throw runtime_error("unexpected EOF");if(sign>0)++truth;else if(sign<0)--truth;return{id,(int)sign,truth};}if(lifecycle)return lifecycle->next();if(t<=c.N)return{public_permute(t),1,t};uint64_t j=t-c.N-1,p=j/2,slot=c.hot?p%c.hot:p%c.N;
        if(c.pattern=="fresh")return (j&1)?Event{public_permute(c.N+p+1),1,c.N}:Event{public_permute(p+1),-1,c.N-1};
        uint64_t generation=p/(c.hot?c.hot:c.N),id=slot+1;if(generation&1)id+=c.N;
        return{public_permute((j&1)?(id<=c.N?id+c.N:id-c.N):id),(j&1)?1:-1,(j&1)?c.N:c.N-1};
    }
};
static vector<uint64_t> schedule_for(Config&c){vector<uint64_t>s;uint64_t prep=c.input.empty()&&c.completion<0?c.N:0;if(c.schedule=="fixed-k"){for(uint64_t j=1;j<=c.K;++j)s.push_back(prep+(uint64_t)((unsigned __int128)(c.T-prep)*j/c.K));}else{for(uint64_t t=prep+c.B;t<c.T;t+=c.B)s.push_back(t);s.push_back(c.T);c.K=s.size();}if(s.empty()||s.front()==0||adjacent_find(s.begin(),s.end())!=s.end())throw runtime_error("invalid schedule");return s;}
static Config parse_config(int argc,char**argv){Config c;for(int i=1;i<argc;++i){string k=argv[i];auto v=[&](){if(++i>=argc)throw runtime_error("missing arg "+k);return string(argv[i]);};if(k=="--mode")c.mode=v();else if(k=="--dataset")c.dataset=v();else if(k=="--pattern")c.pattern=v();else if(k=="--schedule")c.schedule=v();else if(k=="--methods")c.methods=v();else if(k=="--input")c.input=v();else if(k=="--output")c.output=v();else if(k=="--N")c.N=stoull(v());else if(k=="--T")c.T=stoull(v());else if(k=="--K")c.K=stoull(v());else if(k=="--B")c.B=stoull(v());else if(k=="--nmax")c.nmax=stoull(v());else if(k=="--hot")c.hot=stoull(v());else if(k=="--seed")c.seed=stoull(v());else if(k=="--memory")c.mem=stoi(v());else if(k=="--width")c.width=stoi(v());else if(k=="--trials")c.trials=stoi(v());else if(k=="--trial-start")c.trial_start=stoi(v());else if(k=="--q")c.cap_q=stoi(v());else if(k=="--completion")c.completion=stod(v());else if(k=="--epsilon")c.epsilon=stod(v());else if(k=="--delta")c.delta=stod(v());else if(k=="--no-trace")c.trace=false;else throw runtime_error("unknown option "+k);}if(!c.input.empty()){ifstream f(c.input,ios::binary);f.seekg(8);f.read((char*)&c.T,8);if(!f)throw runtime_error("cannot read input header");}return c;}
static int selftest(){
    for(uint64_t K:{7,10,49,200}){TreeNoise tr(K,7,9);map<pair<uint64_t,uint64_t>,pair<int,int>> visits;for(uint64_t j=1;j<=K;++j){auto terms=tr.terms(j);vector<int> decoded(K,0);for(auto[key,s]:terms)if(s){for(uint64_t i=key.first;i<min(K,key.first+key.second);++i)decoded[i]+=s;auto&v=visits[key];if(v.second&&v.second!=(int)j-1)throw runtime_error("tree noise node revisited after eviction");v={v.first?v.first:(int)j,(int)j};}for(uint64_t i=0;i<K;++i)if(decoded[i]!=(i<j))throw runtime_error("tree decode mismatch");tr.noise(j);}
        if(K<=10){double best=0;for(uint64_t mask=0;mask<(1ull<<K);++mask){double total=0;for(auto[key,v]:visits){int left=key.first?((mask>>(key.first-1))&1):0;int right=(mask>>(min(K,key.first+key.second)-1))&1;total+=(left-right)*(left-right);}best=max(best,total);}if(abs(tr.sens(K)*tr.sens(K)-best)>1e-8)throw runtime_error("tree sensitivity mismatch");}
    }
    Config c;c.T=20000;c.N=128;c.nmax=256;c.K=64;c.pattern="reuse";Stream s(c);ParityState p;for(uint64_t t=1;t<=c.T;++t){auto e=s.next();if(p.toggle(e.item)!=e.op||p.count()!=e.truth)throw runtime_error("reuse generator parity mismatch");}
    for(string n:{"hfx","gmod","gmod_mec","hx_native","hx_packed","exact","ajs"}){auto m=make_mechanism(n,c,11,12);for(uint64_t id=1;id<50;++id){m->update(id,1);m->update(id,-1);}auto e=m->release(1);if(e.clean>1e-5)throw runtime_error("cancellation failed "+n);}
    double sig=analytic_gaussian_sigma(1,1e-6,sqrt(200.));if(abs(analytic_gaussian_delta(1,sqrt(200.)/sig)-1e-6)>1e-12)throw runtime_error("Gaussian mismatch");
    cout<<"PASS tree decoding; exact sensitivity enumeration; O(logK) cache; parity reuse; cancellation; analytic Gaussian\n";return 0;
}
#ifndef ICDE_LIBRARY
int main(int argc,char**argv){try{
    Config c=parse_config(argc,argv);if(c.mode=="selftest")return selftest();auto slots=schedule_for(c);filesystem::create_directories(filesystem::path(c.output).parent_path());ofstream summary(c.output+"_summary.csv"),trace,cap_diagnostics;
    if(c.methods.find("cap")!=string::npos){cap_diagnostics.open(c.output+"_cap_diagnostics.csv");cap_diagnostics<<"trial,slot,q,R,H,sigma_node,filtered,accepted,state_weight_bias,unmatched_coordinates\n";}
    summary<<"dataset,pattern,schedule,N,T,K,B,Nmax,memory_kib,method,privacy_unit,calibration,epsilon,delta,beta,trial,hash_seed,noise_seed,w,sensitivity,sigma,state_bytes,rmse,rmsd,clean_rmse,clean_rmsd,service_rmse,release_count,clipped,stopped_at,processed,total_seconds,release_seconds,svt_budget,svt_epsilon1,svt_threshold,resource_exceeded\n";
    if(c.trace){trace.open(c.output+"_releases.csv");trace<<"dataset,method,trial,slot,release_index,truth,estimate,clean_estimate,clipped,stopped_at,query\n";}
    for(int trial=c.trial_start;trial<c.trial_start+c.trials;++trial){uint64_t sh=c.seed+1000003ull*trial,sn=c.seed+7000003ull*trial+123;vector<unique_ptr<Mechanism>> methods;stringstream names(c.methods);string n;while(getline(names,n,',')){int index=n=="hfx"?0:(n=="gmod_public"||n=="hx_phi")?1:n=="exact"?5:n=="hss"?6:n=="ajs"?7:-1;if(index<0)throw runtime_error("undeclared main method");methods.push_back(make_mechanism(n,c,sh,sn+index*10007));}
        Stream stream(c);uint64_t ri=0,count=0;long double sum=0,sumsq=0;auto start=chrono::steady_clock::now();
        for(uint64_t t=1;t<=c.T;++t){auto e=stream.next();for(auto&m:methods){if(m->resource_exceeded)continue;m->update(e.item,e.op);++m->processed;if((c.mode=="resource"||c.mode=="limited")&&(t%1024==0)&&m->bytes()>(c.mode=="resource"?512ull*1024:512ull*1024*1024))m->resource_exceeded=true;}
            if(all_of(methods.begin(),methods.end(),[](const auto& m){return m->resource_exceeded;}))break;
            if(t==slots[ri]){for(auto&m:methods){if(m->resource_exceeded)continue;m->metrics.service(m->held,count,sum,sumsq);auto rs=chrono::steady_clock::now();Estimate x=m->release(ri+1);m->release_seconds+=chrono::duration<double>(chrono::steady_clock::now()-rs).count();m->held=x.value;m->metrics.add(x.value,x.clean,e.truth);m->metrics.service(x.value,1,e.truth,(long double)e.truth*e.truth);m->clipped+=x.clipped;++m->releases;if(c.trace)trace<<c.dataset<<','<<m->name<<','<<trial<<','<<t<<','<<ri+1<<','<<e.truth<<','<<number(x.value)<<','<<number(x.clean)<<','<<x.clipped<<','<<m->stopped<<','<<number(m->raw_statistic())<<'\n';}count=0;sum=sumsq=0;++ri;if(ri==slots.size())break;}
            else{++count;sum+=e.truth;sumsq+=(long double)e.truth*e.truth;}
        }
        double elapsed=chrono::duration<double>(chrono::steady_clock::now()-start).count();
        for(auto&m:methods){auto&z=m->metrics;auto*exact=dynamic_cast<ExactMethod*>(m.get());summary<<c.dataset<<','<<c.pattern<<','<<c.schedule<<','<<c.N<<','<<c.T<<','<<c.K<<','<<c.B<<','<<c.nmax<<','<<c.mem<<','<<m->name<<','<<m->contract<<','<<m->calibration<<','<<number(c.epsilon)<<','<<number(c.delta)<<','<<number(c.beta)<<','<<trial<<','<<sh<<','<<m->noise_seed<<','<<m->width()<<','<<number(m->sensitivity)<<','<<number(m->sigma)<<','<<m->bytes()<<','<<number(sqrt((double)(z.sq/max<uint64_t>(1,z.all))))<<','<<number(sqrt((double)(z.rel/max<uint64_t>(1,z.pos))))<<','<<number(sqrt((double)(z.clean_sq/max<uint64_t>(1,z.all))))<<','<<number(sqrt((double)(z.clean_rel/max<uint64_t>(1,z.pos))))<<','<<number(sqrt((double)(z.service_sq/max<uint64_t>(1,z.service_n))))<<','<<m->releases<<','<<m->clipped<<','<<m->stopped<<','<<m->processed<<','<<number(elapsed)<<','<<number(m->release_seconds)<<','<<(exact?exact->svt_budget():0)<<','<<number(exact?exact->svt_eps():0)<<','<<number(exact?exact->svt_threshold():0)<<','<<m->resource_exceeded<<'\n';}summary.flush();cout<<c.dataset<<" "<<c.pattern<<" N="<<c.N<<" T="<<c.T<<" mem="<<c.mem<<" trial="<<trial<<" methods="<<methods.size()<<" seconds="<<elapsed<<endl;
    }
    return 0;
}catch(const exception&e){cerr<<"ERROR: "<<e.what()<<endl;return 1;}}
#endif
