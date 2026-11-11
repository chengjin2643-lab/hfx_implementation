// Benchmark replay: truth, clean estimates and queries are evaluator diagnostics.
// Only noisy estimates are private outputs; never publish these CSVs as a service API.
#include <climits>
#include <cstdlib>
#define ICDE_LIBRARY
#include "../src/hybrid.hpp"

struct Job {Config c;string group,key,mode,id;int q=16,gw=1,first=0,trials=10;uint64_t gr=0;vector<uint64_t>rows;};

static uint64_t unsigned_field(const string& value) {
    if(value.empty() || value.find_first_not_of("0123456789") != string::npos)
        throw runtime_error("expected an unsigned integer in job.tsv");
    return stoull(value);
}

static int integer_field(const string& value) {
    auto result = unsigned_field(value);
    if(result > INT_MAX) throw runtime_error("job integer exceeds INT_MAX");
    return static_cast<int>(result);
}

static Job read_job(const string& path) {
    ifstream f(path);
    string line;
    if(!f || !getline(f,line) || !getline(f,line))
        throw runtime_error("cannot read the job.tsv header and job row");
    if(!line.empty() && line.back()=='\r') line.pop_back();
    stringstream ss(line);
    vector<string> x;
    string z;
    while(getline(ss,z,'\t')) x.push_back(z);
    if(x.size()!=23) throw runtime_error("job.tsv must have 23 fields");
    for(size_t i=0;i<x.size();++i)
        if(x[i].empty() || x[i].find_first_of("\r\n")!=string::npos ||
           (i!=16 && i!=19 && x[i].find(',')!=string::npos))
            throw runtime_error("job fields must be nonempty and contain no CSV separators");
    if(getline(f,line) && line.find_first_not_of("\r\n \t")!=string::npos)
        throw runtime_error("job.tsv must contain exactly one job row");
    Job j;
    j.mode=x[0];j.c.dataset=x[1];j.c.pattern=x[2];
    j.c.N=unsigned_field(x[3]);j.c.T=unsigned_field(x[4]);
    j.c.nmax=unsigned_field(x[5]);j.c.K=unsigned_field(x[6]);
    j.c.B=unsigned_field(x[7]);j.c.mem=integer_field(x[8]);
    size_t consumed=0;
    j.c.completion=stod(x[9],&consumed);
    if(consumed!=x[9].size()) throw runtime_error("invalid completion fraction");
    j.c.hot=unsigned_field(x[10]);j.c.seed=unsigned_field(x[11]);
    j.c.epsilon=stod(x[12],&consumed);
    if(consumed!=x[12].size()) throw runtime_error("invalid epsilon");
    j.first=integer_field(x[13]);j.trials=integer_field(x[14]);
    j.q=integer_field(x[15]);
    if(x[16].back()==',') throw runtime_error("empty final HFX row count");
    stringstream row_text(x[16]);
    while(getline(row_text,z,',')) j.rows.push_back(unsigned_field(z));
    j.gw=integer_field(x[17]);j.gr=unsigned_field(x[18]);
    if(x[19]!="-") j.c.input=x[19];
    j.group=x[20];j.key=x[21];j.id=x[22];
    j.c.schedule=j.group=="history"||j.group=="real"||j.group=="cadence"||j.group=="epsilon"?"fixed-b":"fixed-k";
    if(j.mode!="payload" && j.mode!="full" && j.mode!="theory")
        throw runtime_error("mode must be payload, full, or theory");
    if(!j.c.T || !j.c.K || !j.c.nmax || j.c.mem<1 || j.trials<1 ||
       j.first>INT_MAX-j.trials || !isfinite(j.c.epsilon) || j.c.epsilon<=0 ||
       !isfinite(j.c.completion) || (j.c.completion!=-1 &&
       (j.c.completion<0 || j.c.completion>1)))
        throw runtime_error("invalid job budget, schedule, trial count, or privacy parameters");
    if(j.q<1 || j.q>16 || j.rows.empty() || j.rows.size()>32 ||
       (j.q!=16 && j.rows.size()<2) || j.gw<1 || j.gw>32 || j.gr<4)
        throw runtime_error("invalid layout width, routing parameter, or GMod row count");
    uint64_t total=0;
    for(auto rows:j.rows) {
        if(rows<4 || rows>UINT32_MAX || total>UINT32_MAX-rows)
            throw runtime_error("invalid HFX row count or bitmap offset");
        total+=rows;
    }
    if(j.c.input.empty()) {
        if(!j.c.N || j.c.hot>j.c.N || (j.c.completion<0 && j.c.T<=j.c.N))
            throw runtime_error("synthetic stream requires a positive population and T>N");
    } else {
        ifstream input(j.c.input,ios::binary);
        char magic[8];uint64_t count=0;
        input.read(magic,8);input.read(reinterpret_cast<char*>(&count),8);
        if(!input || string(magic,8)!="HFXSTRM1" || count!=j.c.T)
            throw runtime_error("input must be an HFXSTRM1 file with exactly T endpoints");
        if(j.c.T>(UINT64_MAX-16)/9 || filesystem::file_size(j.c.input)!=16+9*j.c.T)
            throw runtime_error("binary endpoint stream has an unexpected size");
    }
    uint64_t prep=j.c.input.empty() && j.c.completion<0 ? j.c.N : 0;
    if((j.c.schedule=="fixed-k" && j.c.K>j.c.T-prep) ||
       (j.c.schedule=="fixed-b" && !j.c.B))
        throw runtime_error("release schedule exceeds available input slots");
    uint64_t payload=static_cast<uint64_t>(j.c.mem)*8192;
    if(total>payload || j.gr>payload/static_cast<uint64_t>(j.gw))
        throw runtime_error("layout exceeds the bitmap payload budget");
    return j;
}

// Full-state budgets deduct the same reference object's auxiliary state.
// The original payload replay did not need this allocation step.
static uint64_t half_xor_rows(const Config& c, bool full) {
    uint64_t cap=static_cast<uint64_t>(c.mem)*1024;
    if(!full) return cap*8/32;
    auto probe=make_hybrid_gmod(INT_MAX,"hx_phi",c,32,4,0,0);
    uint64_t overhead=probe->bytes()-16;
    if(cap<=overhead || (cap-overhead)/4<4)
        throw runtime_error("full-state budget cannot hold the Half-Xor layout");
    return (cap-overhead)/4;
}
class PublicPilotHfx:public LeanHfx<false,UINT64_MAX>{
public:
    PublicPilotHfx(Config c,const vector<uint64_t>&r,int q,uint64_t sh,uint64_t sn):LeanHfx("hfx",c,r,q,sh,sn){design(c.nmax/2.);}
    Estimate release(uint64_t)override{double query=0;for(int k=0;k<width();++k)query+=test_weight(k)*test_count(k);double y=query+sigma*rng.normal();return{test_decode(y),test_decode(query),y<0||y>test_response(c.nmax)};}
};
int main(int argc,char**argv){try{
    if(argc!=3)throw runtime_error("job.tsv output-prefix");Job j=read_job(argv[1]);auto c=j.c;auto slots=schedule_for(c);string out=argv[2];
    if(c.K!=j.c.K)throw runtime_error("release interval does not match declared K");
    ofstream s(out+"_summary.csv"),r(out+"_releases.csv");
    if(!s || !r) throw runtime_error("cannot open output files (create the output directory first)");
    const char* selected=getenv("HFX_RUN_METHODS");string methods=selected?selected:"hfx,gmod_public,hx_phi";
    stringstream method_text(methods);string method;
    if(methods.empty() || methods.back()==',') throw runtime_error("empty method selection");
    while(getline(method_text,method,','))
        if(method!="hfx" && method!="gmod_public" && method!="hx_phi")
            throw runtime_error("unknown HFX_RUN_METHODS entry: "+method);
    auto enabled=[&](string name){return (","+methods+",").find(","+name+",")!=string::npos;};
    const bool run_hfx=enabled("hfx")||j.mode=="theory",run_gmod=enabled("gmod_public")&&j.mode!="theory",run_hx=enabled("hx_phi")&&j.mode!="theory";
    uint64_t hx_noise_offset=getenv("HFX_HX_NOISE_OFFSET")?stoull(getenv("HFX_HX_NOISE_OFFSET")):10007ull;
    bool diagnostics=getenv("HFX_DIAGNOSTICS")!=nullptr;ofstream diag;
    if(diagnostics){diag.open(out+"_columns.csv");if(!diag)throw runtime_error("cannot open column diagnostics");diag<<"key,trial,release_index,slot,truth,prior_private,estimate,column,rows,probability,weight,popcount,slope_at_prior,slope_at_truth,weighted_query,response_upper\n";}
    s<<"group,key,dataset,pattern,schedule,N,T,K,B,Nmax,memory_kib,budget_kib,method,trial,hash_seed,noise_seed,w,state_bytes,sigma,rmse,rmsd,clean_rmse,clean_rmsd,service_rmse,release_count,clipped,processed,resource_exceeded,profile_id,geometry_mode\n";
    r<<"group,key,dataset,method,trial,slot,release_index,truth,estimate,clean_estimate,clipped,query\n";
    for(int trial=j.first;trial<j.first+j.trials;++trial){uint64_t sh=c.seed+1000003ull*trial,sn=c.seed+7000003ull*trial+123;unique_ptr<LeanHfx<false,UINT64_MAX>>h;
        if(j.mode=="theory")h=make_unique<PublicPilotHfx>(c,j.rows,j.q,sh,sn);else h=make_unique<LeanHfx<false,UINT64_MAX>>("hfx",c,j.rows,j.q,sh,sn);
        auto g=make_hybrid_gmod(INT_MAX,"gmod_public",c,j.gw,j.gr,sh,sn);auto*gp=dynamic_cast<HybridGmod<UINT64_MAX>*>(g.get());if(!gp)throw runtime_error("GMod type");
        auto hx=make_hybrid_gmod(INT_MAX,"hx_phi",c,32,half_xor_rows(c,j.mode=="full"),sh,sn+hx_noise_offset);auto*hxp=dynamic_cast<HybridGmod<UINT64_MAX>*>(hx.get());if(!hxp)throw runtime_error("Half-Xor type");
        vector<Mechanism*>ps;if(run_hfx)ps.push_back(h.get());if(j.mode!="theory"){if(run_gmod)ps.push_back(g.get());if(run_hx)ps.push_back(hx.get());}if(j.mode=="full")for(auto p:ps)if(p->bytes()>c.mem*1024ull)throw runtime_error("state cap exceeded");
        Stream stream(c);uint64_t count=0;long double sum=0,sumsq=0;size_t next=0;
        for(uint64_t step=1;step<=c.T;++step){auto e=stream.next();auto[x,y]=hash_pair(e.item,sh);if(run_hfx)h->update_hashed(x,y);if(j.mode!="theory"){if(run_gmod)gp->update_hashed(x,y);if(run_hx)hxp->update_hashed(x,y);}
            if(step!=slots[next]){++count;sum+=e.truth;sumsq+=(long double)e.truth*e.truth;continue;}
            for(auto p:ps){double prior=p->held;p->metrics.service(p->held,count,sum,sumsq);auto z=p->release(next+1);p->held=z.value;p->metrics.add(z.value,z.clean,e.truth);p->metrics.service(z.value,1,e.truth,(long double)e.truth*e.truth);++p->releases;p->clipped+=z.clipped;
                double query=0;if(p==h.get())for(int k=0;k<h->width();++k)query+=h->test_weight(k)*h->test_count(k);else query=p->raw_statistic();
                r<<j.group<<','<<j.key<<','<<c.dataset<<','<<p->name<<','<<trial<<','<<step<<','<<next+1<<','<<e.truth<<','<<number(z.value)<<','<<number(z.clean)<<','<<z.clipped<<','<<number(query)<<'\n';
                if(diagnostics&&p==h.get())for(int k=0;k<h->width();++k){
                    int w=h->width();double probability=j.q==16?exp2(-(k==w-1?w-1:k+1)):(k==0?1-j.q/16.:j.q/16.*exp2(-((k-1)==w-2?w-2:k)));
                    double m=h->row(k),l=log1p(-2*probability/m),slope_prior=-.5*m*l*exp(clamp(prior,1.,(double)c.nmax)*l),slope_truth=-.5*m*l*exp(e.truth*l);
                    diag<<j.key<<','<<trial<<','<<next+1<<','<<step<<','<<e.truth<<','<<number(prior)<<','<<number(z.value)<<','<<k<<','<<h->row(k)<<','<<number(probability)<<','<<number(h->test_weight(k))<<','<<number(h->test_count(k))<<','<<number(slope_prior)<<','<<number(slope_truth)<<','<<number(query)<<','<<number(h->test_response(c.nmax))<<'\n';
                }
            }count=0;sum=sumsq=0;if(++next==slots.size())break;
        }
        for(auto p:ps){auto&v=p->metrics;s<<j.group<<','<<j.key<<','<<c.dataset<<','<<c.pattern<<','<<c.schedule<<','<<c.N<<','<<c.T<<','<<c.K<<','<<c.B<<','<<c.nmax<<','<<c.mem<<','<<c.mem<<','<<p->name<<','<<trial<<','<<sh<<','<<p->noise_seed<<','<<p->width()<<','<<p->bytes()<<','<<number(p->sigma)<<','<<number(sqrt((double)(v.sq/v.all)))<<','<<number(sqrt((double)(v.rel/max<uint64_t>(1,v.pos))))<<','<<number(sqrt((double)(v.clean_sq/v.all)))<<','<<number(sqrt((double)(v.clean_rel/max<uint64_t>(1,v.pos))))<<','<<number(sqrt((double)(v.service_sq/max<uint64_t>(1,v.service_n))))<<','<<p->releases<<','<<p->clipped<<','<<c.T<<",0,"<<j.id<<','<<j.mode<<'\n';}
        s.flush();r.flush();cout<<j.key<<" trial="<<trial<<" done\n"<<flush;
        if(!s || !r || (diagnostics && !diag))throw runtime_error("output write failed");
    }
}catch(const exception&e){cerr<<e.what()<<'\n';return 1;}}
