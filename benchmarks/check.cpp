// Small deterministic regression checks for the implementation used in the paper.
#include <climits>
#include "../src/hybrid.hpp"

static void require(bool condition, const char* message) {
    if(!condition) throw runtime_error(message);
}

static void near(double a, double b, double tolerance, const char* message) {
    require(isfinite(a) && isfinite(b) && abs(a-b)<=tolerance,message);
}

static Config test_config() {
    Config c;
    c.N=1000;c.T=21000;c.nmax=2000000;c.K=20;c.mem=32;
    return c;
}

static void cancellation_and_labels() {
    auto c=test_config();
    vector<uint64_t> rows{256,65024,65024,131072};
    LeanHfx<false,UINT64_MAX> positive("hfx",c,rows,8,37,41);
    LeanHfx<false,UINT64_MAX> arbitrary("hfx",c,rows,8,37,41);
    for(uint64_t id=1;id<=2000;++id) {
        positive.update(id,1);
        arbitrary.update(id,static_cast<int>(id%3)-1);
    }
    require(positive.test_bitmap_weight()==arbitrary.test_bitmap_weight(),
            "endpoint signs changed the bitmap");
    auto x=positive.release(1), y=arbitrary.release(1);
    near(x.value,y.value,0,"endpoint signs changed a private release");
    for(uint64_t id=1;id<=2000;++id) {
        positive.update(id,1);
        arbitrary.update(id,0);
    }
    require(positive.test_bitmap_weight()==0 && arbitrary.test_bitmap_weight()==0,
            "two occurrences of an identifier did not cancel");
    near(positive.release(2).clean,0,0,"cancelled stream has a nonzero clean count");
}

static void weights_and_inverse() {
    auto c=test_config();
    LeanHfx<false,UINT64_MAX> h("hfx",c,{256,65024,65024,131072},8,37,41);
    for(int j=0;j<h.width();++j)
        near(h.test_weight(j),1,0,"first release does not start at unit weights");
    for(double pilot:{1.,10000.,300000.,1500000.}) {
        h.design(pilot);
        double maximum=0;
        for(int j=0;j<h.width();++j) {
            auto a=h.test_weight(j);
            require(isfinite(a) && a>=0 && a<=1,"weight outside [0,1]");
            maximum=max(maximum,a);
        }
        near(maximum,1,1e-12,"bounded weight normalization lost a unit coordinate");
        near(h.test_decode(-1),0,0,"lower clipping endpoint failed");
        near(h.test_decode(h.test_response(c.nmax)),static_cast<double>(c.nmax),0,
             "upper response did not invert to the public capacity");
        for(double n:{10.,1000.,10000.,100000.})
            near(h.test_decode(h.test_response(n)),n,1e-4,
                 "response/inverse round trip failed");
    }
}

static void scan_count_equivalence() {
    auto c=test_config();
    vector<uint64_t> rows{256,65024,65024,131072};
    LeanHfx<false,0> counted("hfx",c,rows,8,731,921);
    LeanHfx<false,UINT64_MAX> scanned("hfx",c,rows,8,731,921);
    Stream stream(c);
    uint64_t release=0;
    for(uint64_t slot=1;slot<=c.T;++slot) {
        auto event=stream.next();
        counted.update(event.item,event.op);scanned.update(event.item,event.op);
        if(slot>c.N && (slot-c.N)%1000==0) {
            auto a=counted.release(++release), b=scanned.release(release);
            near(a.value,b.value,0,"counted/scanned private estimates differ");
            near(a.clean,b.clean,0,"counted/scanned clean estimates differ");
            require(a.clipped==b.clipped,"counted/scanned clipping differs");
            for(int j=0;j<counted.width();++j) {
                near(counted.test_count(j),scanned.test_count(j),0,
                     "counted/scanned popcounts differ");
                near(counted.test_weight(j),scanned.test_weight(j),0,
                     "counted/scanned adaptive weights differ");
            }
        }
    }
    require(release==c.K,"unexpected release count in scan/count check");
}

static void budget_and_gaussian() {
    auto c=test_config();
    LeanHfx<false,UINT64_MAX> probe("hfx",c,{64,64},8,1,2);
    uint64_t auxiliary=probe.bytes()-probe.payload_bytes(), budget=c.mem*1024ull;
    require(budget>auxiliary,"test budget too small");
    uint64_t bits=(budget-auxiliary)/8*64;
    LeanHfx<false,UINT64_MAX> h("hfx",c,{384,bits-384},8,1,2);
    require(h.bytes()<=budget && h.bytes()+8>budget,
            "complete-state allocation did not respect the memory cap");
    for(uint64_t K:{1ull,20ull,200ull,393ull}) {
        auto sigma=analytic_gaussian_sigma(1,1e-6,sqrt(static_cast<double>(K)));
        near(analytic_gaussian_delta(1,sqrt(static_cast<double>(K))/sigma),
             1e-6,1e-12,"analytic Gaussian calibration mismatch");
    }
}

int main() {
    try {
        cancellation_and_labels();
        weights_and_inverse();
        scan_count_equivalence();
        budget_and_gaussian();
        cout << "PASS endpoint cancellation, directionless updates, bounded weights, "
                "inverse endpoints, scan/count equivalence, complete-state budget, "
                "and analytic Gaussian calibration\n";
    } catch(const exception& error) {
        cerr << error.what() << '\n';
        return 1;
    }
}
