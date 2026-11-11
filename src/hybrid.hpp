#pragma once
#include "lean_hfx.hpp"
#define LeanHfx PreviousLeanHfx
#include "previous_lean_hfx.hpp"
#undef LeanHfx

template<uint64_t ScanWords,bool IntegerCounts=false> class HybridGmod:public ReversibleReference {
public:
    HybridGmod(string label,const Config& c,uint64_t sh,uint64_t sn):ReversibleReference("gmod_public",c,sh,sn){name=label;}
    void update_hashed(uint64_t x,uint64_t y){ReversibleReference::update_hashed_scan<ScanWords,IntegerCounts>(x,y);}
    void update(uint64_t id,int)override{ReversibleReference::update_id_scan<ScanWords,IntegerCounts>(id);}
    Estimate release(uint64_t j)override{return ReversibleReference::release_scan<ScanWords,IntegerCounts>(j);}
    double raw_statistic()const override{return ReversibleReference::raw_scan_statistic<ScanWords,IntegerCounts>();}
};

template<bool PowerOfTwo> class SingleColumnScanHfx final : public LeanHfx<false,UINT64_MAX> {
public:
    using LeanHfx<false,UINT64_MAX>::LeanHfx;
    void update(uint64_t id,int) override {update_single_scan<PowerOfTwo>(id);}
};

static vector<uint64_t> parse_rows(string s){vector<uint64_t>r;string item;stringstream in(s);while(getline(in,item,','))r.push_back(stoull(item));return r;}
static unique_ptr<Mechanism> make_hybrid_hfx(int threshold,string label,Config c,const vector<uint64_t>&rows,int q,uint64_t sh,uint64_t sn){
    if(threshold==INT_MAX && q==16 && rows.size()==1) {
        if (has_single_bit(rows[0])) return make_unique<SingleColumnScanHfx<true>>(label,c,rows,q,sh,sn);
        return make_unique<SingleColumnScanHfx<false>>(label,c,rows,q,sh,sn);
    }
    if(threshold<0){switch(threshold){
#define HX_INT_CASE(S) case -(S+1):return make_unique<LeanHfx<false,S,true>>(label,c,rows,q,sh,sn);
    HX_INT_CASE(0) HX_INT_CASE(8) HX_INT_CASE(32) HX_INT_CASE(128) HX_INT_CASE(512) HX_INT_CASE(2048) HX_INT_CASE(32768)
#undef HX_INT_CASE
    default:throw runtime_error("unsupported integer HFX policy");}}
    switch(threshold){
    case INT_MAX:return make_unique<LeanHfx<false,UINT64_MAX>>(label,c,rows,q,sh,sn);
#define HX_CASE(S) case S:return make_unique<LeanHfx<false,S>>(label,c,rows,q,sh,sn);
    HX_CASE(0) HX_CASE(8) HX_CASE(32) HX_CASE(128) HX_CASE(512) HX_CASE(2048) HX_CASE(32768)
#undef HX_CASE
    default:throw runtime_error("unsupported HFX scan threshold");}
}
static unique_ptr<Mechanism> make_hybrid_gmod(int threshold,string label,Config c,int w,uint64_t rows,uint64_t sh,uint64_t sn){
    FULL_STATE_GMOD_WIDTH=w;PUBLIC_AUDIT_ROWS_OVERRIDE=rows;unique_ptr<Mechanism> out;
    switch(threshold){
    case INT_MAX:out=make_unique<HybridGmod<UINT64_MAX>>(label,c,sh,sn);break;
#define GM_INT_CASE(S) case -(S+1):out=make_unique<HybridGmod<S,true>>(label,c,sh,sn);break;
    GM_INT_CASE(0) GM_INT_CASE(8) GM_INT_CASE(32) GM_INT_CASE(128) GM_INT_CASE(512) GM_INT_CASE(2048) GM_INT_CASE(32768)
#undef GM_INT_CASE
#define GM_CASE(S) case S:out=make_unique<HybridGmod<S>>(label,c,sh,sn);break;
    GM_CASE(0) GM_CASE(8) GM_CASE(32) GM_CASE(128) GM_CASE(512) GM_CASE(2048) GM_CASE(32768)
#undef GM_CASE
    default:throw runtime_error("unsupported GMod scan threshold");}
    FULL_STATE_GMOD_WIDTH=0;PUBLIC_AUDIT_ROWS_OVERRIDE=0;return out;
}
