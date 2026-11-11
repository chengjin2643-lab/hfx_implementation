#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;

static constexpr uint64_t GOLDEN64 = 0x9E3779B97F4A7C15ull;
static constexpr uint64_t PUBLIC_PERMUTATION_SEED = 0xA24BAED4963EE407ull;

struct Args {
    string mode = "active-scale";
    string schedule = "fixed-k";
    string output_prefix = "results/run";
    uint64_t N = 100000;
    uint64_t T = 0;
    uint64_t L = 2000000;
    uint64_t n_max = 0;
    uint64_t B = 10000;
    uint64_t K = 200;
    int memory_kb = 32;
    int width = 0;
    int trials = 5;
    double epsilon = 1.0;
    double delta = 1e-6;
    double completion = 0.0;
    uint64_t seed_hash = 63242691;
    uint64_t seed_noise = 981723641;
    uint64_t progress_step = 100000000;
    bool run_exact_dict = false;
};

static uint64_t splitmix64(uint64_t x) {
    x += GOLDEN64;
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull);
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EBull);
    return x ^ (x >> 31);
}

static uint64_t public_permute(uint64_t item) {
    return splitmix64(item ^ PUBLIC_PERMUTATION_SEED);
}

static pair<uint64_t, uint64_t> hash_pair(uint64_t item, uint64_t seed) {
    const uint64_t h1 = splitmix64(item ^ (seed * GOLDEN64));
    const uint64_t h2 = splitmix64(h1 ^ (seed << 32) ^ 0xD1B54A32D192ED03ull);
    return {h1, h2};
}

static int ctz64(uint64_t x) {
    return x == 0 ? 64 : static_cast<int>(countr_zero(x));
}

static string number(double value) {
    if (!isfinite(value)) return "";
    ostringstream output;
    output << setprecision(17) << value;
    return output.str();
}

static double normal_cdf(double x) {
    return 0.5 * erfc(-x / sqrt(2.0));
}

static double analytic_gaussian_delta(double epsilon, double mu) {
    if (!(mu > 0.0)) return 0.0;
    const double first = normal_cdf(0.5 * mu - epsilon / mu);
    const double second = exp(epsilon) * normal_cdf(-0.5 * mu - epsilon / mu);
    return max(0.0, first - second);
}

static double analytic_gaussian_sigma(double epsilon, double delta,
                                      double l2_sensitivity) {
    double low = numeric_limits<double>::min();
    double high = max(1.0, l2_sensitivity);
    while (analytic_gaussian_delta(epsilon, l2_sensitivity / high) > delta) {
        high *= 2.0;
        if (!isfinite(high)) throw runtime_error("analytic Gaussian calibration overflow");
    }
    for (int iteration = 0; iteration < 160; ++iteration) {
        const double mid = 0.5 * (low + high);
        if (analytic_gaussian_delta(epsilon, l2_sensitivity / mid) > delta) low = mid;
        else high = mid;
    }
    return high;
}

static uint64_t packed_rows(int memory_kb, int width) {
    const uint64_t bits = static_cast<uint64_t>(memory_kb) * 1024ull * 8ull;
    if (bits == 0 || width <= 0) throw invalid_argument("invalid packed geometry");
    return max<uint64_t>(1, bits / static_cast<uint64_t>(width));
}

static vector<double> column_probabilities(int width) {
    vector<double> probabilities;
    probabilities.reserve(static_cast<size_t>(width));
    for (int column = 0; column < width; ++column) {
        probabilities.push_back(column == width - 1 ? pow(2.0, -(width - 1.0))
                                                     : pow(2.0, -(column + 1.0)));
    }
    return probabilities;
}

static double endpoint_slope(int memory_kb, int width, uint64_t n_max) {
    const uint64_t rows = packed_rows(memory_kb, width);
    double total = 0.0;
    for (double probability : column_probabilities(width)) {
        const double log_term = log1p(-2.0 * probability / static_cast<double>(rows));
        const double exponent = static_cast<double>(n_max) * log_term;
        const double power = exponent < -745.0 ? 0.0 : exp(exponent);
        total += -log_term * power;
    }
    return 0.5 * static_cast<double>(rows) * total;
}

static int select_public_width(int memory_kb, uint64_t n_max) {
    int best_width = 1;
    double best_slope = -1.0;
    for (int width = 1; width <= 32; ++width) {
        const double slope = endpoint_slope(memory_kb, width, n_max);
        if (slope > best_slope) {
            best_slope = slope;
            best_width = width;
        }
    }
    return best_width;
}

class PackedGeometricHfx {
    uint64_t m_;
    int w_;
    uint64_t n_max_;
    vector<uint64_t> words_;
    double weight_ = 0.0;
    vector<double> log_terms_;
    double max_phi_ = 0.0;

    double phi(double n) const {
        double total = 0.0;
        for (double term : log_terms_) total += -expm1(n * term);
        return 0.5 * static_cast<double>(m_) * total;
    }

public:
    PackedGeometricHfx(int memory_kb, int width, uint64_t n_max, uint64_t rows_override=0)
        : m_(rows_override?rows_override:packed_rows(memory_kb, width)), w_(width), n_max_(n_max) {
        if (w_ <= 0 || w_ > 32 || n_max_ == 0) throw invalid_argument("invalid HFX geometry");
        const uint64_t cells = m_ * static_cast<uint64_t>(w_);
        words_.assign((cells + 63ull) / 64ull, 0ull);
        log_terms_.reserve(static_cast<size_t>(w_));
        for (double probability : column_probabilities(w_))
            log_terms_.push_back(log1p(-2.0 * probability / static_cast<double>(m_)));
        max_phi_ = phi(static_cast<double>(n_max_));
    }

    pair<int,double> update_hashed_record(uint64_t h1, uint64_t h2) {
        const uint64_t row = h1 % m_;
        const int column = min(ctz64(h2), w_ - 1);
        const uint64_t cell = row * static_cast<uint64_t>(w_) + static_cast<uint64_t>(column);
        const uint64_t word = cell >> 6;
        const uint64_t mask = 1ull << (cell & 63ull);
        const bool old_one = (words_[word] & mask) != 0;
        words_[word] ^= mask;
        const double increment=old_one ? -1.0 : 1.0;
        weight_ += increment;
        return {column,increment};
    }

    void update_hashed(uint64_t h1,uint64_t h2){update_hashed_record(h1,h2);}

    double weight() const { return weight_; }
    double max_state() const { return max_phi_; }
    uint64_t rows() const { return m_; }
    uint64_t logical_cells() const { return m_ * static_cast<uint64_t>(w_); }
    uint64_t payload_bytes() const { return words_.size() * sizeof(uint64_t); }
    uint64_t full_state_bytes() const {
        return sizeof(*this) + words_.capacity() * sizeof(uint64_t) +
               log_terms_.capacity() * sizeof(double);
    }

    double invert(double observed) const {
        // In a saturated layout phi may be a floating-point plateau. Preserve
        // the public clipped endpoints rather than inverting its left edge.
        if (observed <= 0.0) return 0.0;
        if (observed >= max_phi_) return static_cast<double>(n_max_);
        const double z = clamp(observed, 0.0, max_phi_);
        double low = 0.0;
        double high = static_cast<double>(n_max_);
        for (int iteration = 0; iteration < 52; ++iteration) {
            const double mid = 0.5 * (low + high);
            if (phi(mid) < z) low = mid;
            else high = mid;
        }
        return 0.5 * (low + high);
    }
};

class FMSplit {
    uint64_t m_;
    vector<uint32_t> add_;
    vector<uint32_t> del_;
    double add_state_ = 0.0;
    double del_state_ = 0.0;

    static double update_one(vector<uint32_t>& rows, uint64_t row, int column) {
        const uint32_t old_value = rows[row];
        const int old_prefix = static_cast<int>(countr_one(old_value));
        rows[row] |= (1u << column);
        return static_cast<double>(static_cast<int>(countr_one(rows[row])) - old_prefix);
    }

public:
    explicit FMSplit(int memory_kb)
        : m_(max<uint64_t>(1, static_cast<uint64_t>(memory_kb) * 1024ull /
                                  (2ull * sizeof(uint32_t)))),
          add_(m_, 0u), del_(m_, 0u) {}

    void update(uint64_t h1, uint64_t h2, int op) {
        const uint64_t row = h1 % m_;
        const int column = min(ctz64(h2), 31);
        if (op > 0) add_state_ += update_one(add_, row, column);
        else del_state_ += update_one(del_, row, column);
    }

    double add_state() const { return add_state_; }
    double del_state() const { return del_state_; }
    double max_state() const { return 32.0 * static_cast<double>(m_); }
    uint64_t logical_cells() const { return 2ull * m_ * 32ull; }
    uint64_t payload_bytes() const { return 2ull * m_ * sizeof(uint32_t); }
    uint64_t full_state_bytes() const {
        return sizeof(*this) + (add_.capacity() + del_.capacity()) * sizeof(uint32_t);
    }
    double invert(double observed, uint64_t branch_cap) const {
        const double z = clamp(observed, 0.0, max_state());
        return clamp(static_cast<double>(m_) * pow(2.0, z / static_cast<double>(m_)) /
                         0.77351,
                     0.0, static_cast<double>(branch_cap));
    }
};

class LLSplit {
    uint64_t m_;
    vector<uint8_t> add_;
    vector<uint8_t> del_;
    double add_state_ = 0.0;
    double del_state_ = 0.0;

    static double update_one(vector<uint8_t>& regs, uint64_t row, uint8_t rank) {
        if (rank <= regs[row]) return 0.0;
        const double jump = static_cast<double>(rank - regs[row]);
        regs[row] = rank;
        return jump;
    }

public:
    explicit LLSplit(int memory_kb)
        : m_(max<uint64_t>(1, static_cast<uint64_t>(memory_kb) * 1024ull / 2ull)),
          add_(m_, 0u), del_(m_, 0u) {}

    void update(uint64_t h1, uint64_t h2, int op) {
        const uint64_t row = h1 % m_;
        const uint8_t rank = static_cast<uint8_t>(min(ctz64(h2) + 1, 64));
        if (op > 0) add_state_ += update_one(add_, row, rank);
        else del_state_ += update_one(del_, row, rank);
    }

    double add_state() const { return add_state_; }
    double del_state() const { return del_state_; }
    double max_state() const { return 64.0 * static_cast<double>(m_); }
    uint64_t logical_cells() const { return 2ull * m_; }
    uint64_t payload_bytes() const { return 2ull * m_ * sizeof(uint8_t); }
    uint64_t full_state_bytes() const {
        return sizeof(*this) + (add_.capacity() + del_.capacity()) * sizeof(uint8_t);
    }
    double invert(double observed, uint64_t branch_cap) const {
        const double z = clamp(observed, 0.0, max_state());
        return clamp(0.39701 * static_cast<double>(m_) *
                         pow(2.0, z / static_cast<double>(m_)),
                     0.0, static_cast<double>(branch_cap));
    }
};

class HLLSplit {
    uint64_t m_;
    vector<uint8_t> add_;
    vector<uint8_t> del_;
    double add_state_ = 0.0;
    double del_state_ = 0.0;
    double alpha_ = 0.0;

    static double update_one(vector<uint8_t>& regs, uint64_t row, uint8_t rank) {
        if (rank <= regs[row]) return 0.0;
        const double jump = exp2(-static_cast<double>(regs[row])) -
                            exp2(-static_cast<double>(rank));
        regs[row] = rank;
        return jump;
    }

public:
    explicit HLLSplit(int memory_kb)
        : m_(max<uint64_t>(1, static_cast<uint64_t>(memory_kb) * 1024ull / 2ull)),
          add_(m_, 0u), del_(m_, 0u) {
        alpha_ = m_ == 16 ? 0.673 : m_ == 32 ? 0.697 : m_ == 64 ? 0.709
                         : 0.7213 / (1.0 + 1.079 / static_cast<double>(m_));
    }

    void update(uint64_t h1, uint64_t h2, int op) {
        const uint64_t row = h1 % m_;
        const uint8_t rank = static_cast<uint8_t>(min(ctz64(h2) + 1, 64));
        if (op > 0) add_state_ += update_one(add_, row, rank);
        else del_state_ += update_one(del_, row, rank);
    }

    double add_state() const { return add_state_; }
    double del_state() const { return del_state_; }
    double max_state() const { return nextafter(static_cast<double>(m_), 0.0); }
    uint64_t logical_cells() const { return 2ull * m_; }
    uint64_t payload_bytes() const { return 2ull * m_ * sizeof(uint8_t); }
    uint64_t full_state_bytes() const {
        return sizeof(*this) + (add_.capacity() + del_.capacity()) * sizeof(uint8_t);
    }
    double invert(double observed, uint64_t branch_cap) const {
        const double state = clamp(observed, 0.0, max_state());
        const double harmonic_sum = static_cast<double>(m_) - state;
        return clamp(alpha_ * static_cast<double>(m_) * static_cast<double>(m_) /
                         harmonic_sum,
                     0.0, static_cast<double>(branch_cap));
    }
};

struct Metrics {
    long double squared_relative = 0.0L;
    long double squared_absolute = 0.0L;
    long double squared_zero = 0.0L;
    uint64_t positive_count = 0;
    uint64_t all_count = 0;
    uint64_t zero_count = 0;

    void add(double estimate, uint64_t truth) {
        const long double error = static_cast<long double>(estimate) -
                                  static_cast<long double>(truth);
        squared_absolute += error * error;
        ++all_count;
        if (truth > 0) {
            const long double relative = error / static_cast<long double>(truth);
            squared_relative += relative * relative;
            ++positive_count;
        } else {
            squared_zero += error * error;
            ++zero_count;
        }
    }

    double rmsd() const {
        return positive_count ? sqrt(static_cast<double>(squared_relative / positive_count)) : 0.0;
    }
    double rmse() const {
        return all_count ? sqrt(static_cast<double>(squared_absolute / all_count)) : 0.0;
    }
    double zero_rmse() const {
        return zero_count ? sqrt(static_cast<double>(squared_zero / zero_count)) : 0.0;
    }
};

struct MethodState {
    string name;
    string input_model;
    string calibration;
    double scalar_sensitivity = 1.0;
    double l2_sensitivity = 0.0;
    double sigma = 0.0;
    uint64_t logical_cells = 0;
    uint64_t payload_bytes = 0;
    uint64_t peak_state_bytes = 0;
    uint64_t clipped_count = 0;
    Metrics private_metrics;
    Metrics clean_metrics;
};

struct ReleaseEstimate {
    double private_estimate = 0.0;
    double clean_estimate = 0.0;
    bool private_state_clipped = false;
    bool clean_state_clipped = false;
    bool output_clipped = false;
};

static double clip_output(double value, uint64_t n_max, bool& clipped) {
    clipped = value < 0.0 || value > static_cast<double>(n_max);
    return clamp(value, 0.0, static_cast<double>(n_max));
}

static double clip_state(double value, double maximum, bool& clipped) {
    clipped = value < 0.0 || value > maximum;
    return clamp(value, 0.0, maximum);
}

static vector<uint64_t> make_release_slots(const Args& args, uint64_t H,
                                           uint64_t T_total) {
    vector<uint64_t> slots;
    if (args.mode == "lifecycle") {
        for (uint64_t slot = args.B; slot <= T_total; slot += args.B) slots.push_back(slot);
        if (slots.empty() || slots.back() != T_total) slots.push_back(T_total);
    } else if (args.schedule == "fixed-k") {
        const uint64_t window = T_total - H;
        if (args.K == 0 || window < args.K) throw invalid_argument("fixed-k schedule is not feasible");
        for (uint64_t index = 1; index <= args.K; ++index) {
            const unsigned __int128 numerator =
                static_cast<unsigned __int128>(index) * window + args.K - 1;
            const uint64_t offset = static_cast<uint64_t>(numerator / args.K);
            slots.push_back(H + offset);
        }
    } else if (args.schedule == "fixed-b") {
        const uint64_t window = T_total - H;
        for (uint64_t offset = args.B; offset <= window; offset += args.B)
            slots.push_back(H + offset);
        if (slots.empty() || slots.back() != T_total) slots.push_back(T_total);
    } else {
        throw invalid_argument("schedule must be fixed-k or fixed-b");
    }
    if (!is_sorted(slots.begin(), slots.end()) ||
        adjacent_find(slots.begin(), slots.end()) != slots.end())
        throw runtime_error("release schedule is not strictly increasing");
    return slots;
}

struct Event {
    uint64_t item = 0;
    int op = 0;
    uint64_t truth = 0;
};

class ReplacementGenerator {
    uint64_t N_;
    uint64_t T_;
    uint64_t t_ = 0;

public:
    ReplacementGenerator(uint64_t N, uint64_t T) : N_(N), T_(T) {
        if (N_ == 0 || T_ < N_) throw invalid_argument("invalid replacement stream");
    }

    Event next() {
        if (t_ >= T_) throw runtime_error("replacement generator exhausted");
        ++t_;
        if (t_ <= N_) return {public_permute(t_), 1, t_};
        const uint64_t observation_index = t_ - N_ - 1;
        const uint64_t pair_index = observation_index / 2;
        if ((observation_index & 1ull) == 0)
            return {public_permute(pair_index + 1ull), -1, N_ - 1ull};
        return {public_permute(N_ + pair_index + 1ull), 1, N_};
    }
};

class LifecycleGenerator {
    uint64_t T_;
    uint64_t insertions_;
    uint64_t deletions_;
    uint64_t emitted_ = 0;
    uint64_t inserted_ = 0;
    uint64_t deleted_ = 0;
    uint64_t truth_ = 0;
    bool deletion_pending_ = false;

public:
    LifecycleGenerator(uint64_t T, double completion)
        : T_(T),
          insertions_(static_cast<uint64_t>(llround(static_cast<long double>(T) /
                                                   (1.0L + completion)))),
          deletions_(T - insertions_) {
        if (T_ == 0 || completion < 0.0 || completion > 1.0 || deletions_ > insertions_)
            throw invalid_argument("invalid lifecycle stream");
    }

    Event next() {
        if (emitted_ >= T_) throw runtime_error("lifecycle generator exhausted");
        Event event;
        if (deletion_pending_) {
            ++deleted_;
            --truth_;
            event = {public_permute(deleted_), -1, truth_};
            deletion_pending_ = false;
        } else {
            ++inserted_;
            ++truth_;
            event = {public_permute(inserted_), 1, truth_};
            const uint64_t target_deletions = static_cast<uint64_t>(
                (static_cast<unsigned __int128>(inserted_) * deletions_) / insertions_);
            deletion_pending_ = deleted_ < target_deletions;
        }
        ++emitted_;
        return event;
    }

    uint64_t insertions() const { return insertions_; }
    uint64_t deletions() const { return deletions_; }
};

class ReleaseCsv {
    ofstream output_;
    string experiment_;
    string run_id_;
    const Args& args_;
    uint64_t H_;
    uint64_t T_;
    uint64_t planned_K_;
    int width_;
    uint64_t rows_;

public:
    ReleaseCsv(const string& path, string experiment, string run_id, const Args& args,
               uint64_t H, uint64_t T, uint64_t planned_K, int width, uint64_t rows)
        : output_(path), experiment_(std::move(experiment)), run_id_(std::move(run_id)),
          args_(args), H_(H), T_(T), planned_K_(planned_K), width_(width), rows_(rows) {
        if (!output_) throw runtime_error("cannot create release CSV: " + path);
        output_ << "experiment,run_id,method,trial,public_slot,release_index,truth,"
                   "clean_estimate,private_estimate,private_state_clipped,clean_state_clipped,"
                   "output_clipped,N,Nmax,payload_bytes,w,m,T_total,H,L,K,B,epsilon,delta\n";
    }

    void add(const MethodState& method, int trial, uint64_t public_slot,
             uint64_t release_index, uint64_t truth, const ReleaseEstimate& estimate) {
        output_ << experiment_ << ',' << run_id_ << ',' << method.name << ',' << trial << ','
                << public_slot << ',' << release_index << ',' << truth << ','
                << number(estimate.clean_estimate) << ',' << number(estimate.private_estimate)
                << ',' << estimate.private_state_clipped << ',' << estimate.clean_state_clipped
                << ',' << estimate.output_clipped << ',' << args_.N << ',' << args_.n_max << ','
                << method.payload_bytes << ',' << width_ << ',' << rows_ << ',' << T_ << ','
                << H_ << ',' << (T_ - H_) << ',' << planned_K_ << ',' << args_.B << ','
                << number(args_.epsilon) << ',' << number(args_.delta) << '\n';
    }
};

class Trial {
public:
    enum MethodIndex { EXACT_QUERY, EXACT_DICT, HFX, FM, LL, HLL, METHOD_COUNT };

private:
    int trial_;
    uint64_t seed_hash_;
    uint64_t seed_noise_;
    uint64_t T_;
    uint64_t n_max_;
    bool include_exact_dict_;
    PackedGeometricHfx hfx_;
    FMSplit fm_;
    LLSplit ll_;
    HLLSplit hll_;
    mt19937_64 rng_;
    normal_distribution<double> standard_normal_{0.0, 1.0};
    vector<MethodState> methods_;

    template <class SplitSketch>
    ReleaseEstimate release_split(MethodIndex index, const SplitSketch& sketch,
                                  double z_add, double z_del) {
        MethodState& method = methods_[index];
        bool add_clipped = false;
        bool del_clipped = false;
        const double noisy_add = clip_state(sketch.add_state() + method.sigma * z_add,
                                            sketch.max_state(), add_clipped);
        const double noisy_del = clip_state(sketch.del_state() + method.sigma * z_del,
                                            sketch.max_state(), del_clipped);
        const double private_raw = sketch.invert(noisy_add, T_) - sketch.invert(noisy_del, T_);
        const double clean_raw = sketch.invert(sketch.add_state(), T_) -
                                 sketch.invert(sketch.del_state(), T_);
        ReleaseEstimate result;
        result.private_state_clipped = add_clipped || del_clipped;
        result.clean_state_clipped = false;
        result.private_estimate = clip_output(private_raw, n_max_, result.output_clipped);
        bool clean_output_clipped = false;
        result.clean_estimate = clip_output(clean_raw, n_max_, clean_output_clipped);
        result.output_clipped = result.output_clipped || clean_output_clipped;
        return result;
    }

    void record(MethodIndex index, uint64_t public_slot, uint64_t release_index,
                uint64_t truth, const ReleaseEstimate& estimate, ReleaseCsv& writer) {
        MethodState& method = methods_[index];
        method.private_metrics.add(estimate.private_estimate, truth);
        method.clean_metrics.add(estimate.clean_estimate, truth);
        method.clipped_count += estimate.private_state_clipped + estimate.clean_state_clipped +
                                estimate.output_clipped;
        writer.add(method, trial_, public_slot, release_index, truth, estimate);
    }

public:
    Trial(int trial, const Args& args, uint64_t T, uint64_t planned_K, int width,
          bool include_exact_dict)
        : trial_(trial), seed_hash_(args.seed_hash + static_cast<uint64_t>(trial) * 100ull),
          seed_noise_(args.seed_noise + static_cast<uint64_t>(trial) * 100ull), T_(T),
          n_max_(args.n_max), include_exact_dict_(include_exact_dict),
          hfx_(args.memory_kb, width, args.n_max), fm_(args.memory_kb),
          ll_(args.memory_kb), hll_(args.memory_kb), rng_(seed_noise_),
          methods_(METHOD_COUNT) {
        auto initialize = [&](MethodIndex index, string name, string input_model,
                              string calibration, double scalar_sensitivity,
                              uint64_t logical_cells, uint64_t payload_bytes,
                              uint64_t peak_state_bytes) {
            MethodState& method = methods_[index];
            method.name = std::move(name);
            method.input_model = std::move(input_model);
            method.calibration = std::move(calibration);
            method.scalar_sensitivity = scalar_sensitivity;
            method.l2_sensitivity = scalar_sensitivity * sqrt(static_cast<double>(planned_K));
            method.sigma = analytic_gaussian_sigma(args.epsilon, args.delta,
                                                    method.l2_sensitivity);
            method.logical_cells = logical_cells;
            method.payload_bytes = payload_bytes;
            method.peak_state_bytes = peak_state_bytes;
        };
        initialize(EXACT_QUERY, "exact_query_dp", "offline_truth_query",
                   "isotropic_gaussian_release_vector", 1.0, 0, 0, 0);
        initialize(EXACT_DICT, "exact_dict_dp", "endpoint_identifier_only",
                   "isotropic_gaussian_release_vector", 1.0, 0, 0, 0);
        initialize(HFX, "hfx_packed", "endpoint_identifier_only",
                   "isotropic_gaussian_release_vector", 1.0, hfx_.logical_cells(),
                   hfx_.payload_bytes(), hfx_.full_state_bytes());
        initialize(FM, "fm_diff", "reliable_direction_label",
                   "joint_two_branch_isotropic_gaussian", 32.0, fm_.logical_cells(),
                   fm_.payload_bytes(), fm_.full_state_bytes());
        initialize(LL, "ll_diff", "reliable_direction_label",
                   "joint_two_branch_isotropic_gaussian", 64.0, ll_.logical_cells(),
                   ll_.payload_bytes(), ll_.full_state_bytes());
        initialize(HLL, "hll_diff", "reliable_direction_label",
                   "joint_two_branch_isotropic_gaussian", 1.0, hll_.logical_cells(),
                   hll_.payload_bytes(), hll_.full_state_bytes());
    }

    void update(uint64_t item, int op) {
        const auto [h1, h2] = hash_pair(item, seed_hash_);
        hfx_.update_hashed(h1, h2);
        fm_.update(h1, h2, op);
        ll_.update(h1, h2, op);
        hll_.update(h1, h2, op);
    }

    void release(uint64_t public_slot, uint64_t release_index, uint64_t truth,
                 ReleaseCsv& writer) {
        const double z_direct = standard_normal_(rng_);
        const double z_add = standard_normal_(rng_);
        const double z_del = standard_normal_(rng_);

        ReleaseEstimate exact;
        exact.clean_estimate = static_cast<double>(truth);
        exact.private_estimate = clip_output(static_cast<double>(truth) +
                                                 methods_[EXACT_QUERY].sigma * z_direct,
                                             n_max_, exact.output_clipped);
        record(EXACT_QUERY, public_slot, release_index, truth, exact, writer);
        if (include_exact_dict_) record(EXACT_DICT, public_slot, release_index, truth, exact, writer);

        ReleaseEstimate hfx;
        hfx.clean_state_clipped = hfx_.weight() < 0.0 || hfx_.weight() > hfx_.max_state();
        hfx.clean_estimate = hfx_.invert(hfx_.weight());
        const double noisy_hfx = clip_state(hfx_.weight() + methods_[HFX].sigma * z_direct,
                                            hfx_.max_state(), hfx.private_state_clipped);
        hfx.private_estimate = hfx_.invert(noisy_hfx);
        record(HFX, public_slot, release_index, truth, hfx, writer);

        const ReleaseEstimate fm = release_split(FM, fm_, z_add, z_del);
        const ReleaseEstimate ll = release_split(LL, ll_, z_add, z_del);
        const ReleaseEstimate hll = release_split(HLL, hll_, z_add, z_del);
        record(FM, public_slot, release_index, truth, fm, writer);
        record(LL, public_slot, release_index, truth, ll, writer);
        record(HLL, public_slot, release_index, truth, hll, writer);
    }

    int trial() const { return trial_; }
    uint64_t seed_hash() const { return seed_hash_; }
    uint64_t seed_noise() const { return seed_noise_; }
    vector<MethodState>& methods() { return methods_; }
    const vector<MethodState>& methods() const { return methods_; }
};

struct AllocationTracker {
    uint64_t current = 0;
    uint64_t peak = 0;
    void add(uint64_t bytes) {
        current += bytes;
        peak = max(peak, current);
    }
    void subtract(uint64_t bytes) {
        if (bytes > current) throw runtime_error("allocator accounting underflow");
        current -= bytes;
    }
};

template <class T>
class CountingAllocator {
public:
    using value_type = T;
    shared_ptr<AllocationTracker> tracker;

    CountingAllocator() : tracker(make_shared<AllocationTracker>()) {}
    explicit CountingAllocator(shared_ptr<AllocationTracker> value) : tracker(std::move(value)) {}
    template <class U>
    CountingAllocator(const CountingAllocator<U>& other) noexcept : tracker(other.tracker) {}

    T* allocate(size_t count) {
        const uint64_t bytes = static_cast<uint64_t>(count) * sizeof(T);
        tracker->add(bytes);
        try {
            return allocator<T>{}.allocate(count);
        } catch (...) {
            tracker->subtract(bytes);
            throw;
        }
    }
    void deallocate(T* pointer, size_t count) noexcept {
        tracker->subtract(static_cast<uint64_t>(count) * sizeof(T));
        allocator<T>{}.deallocate(pointer, count);
    }
    template <class U>
    bool operator==(const CountingAllocator<U>& other) const noexcept {
        return tracker == other.tracker;
    }
    template <class U>
    bool operator!=(const CountingAllocator<U>& other) const noexcept {
        return !(*this == other);
    }
};

struct ExactDictResult {
    string status = "not_run";
    uint64_t peak_state_bytes = 0;
    uint64_t peak_size = 0;
    uint64_t peak_bucket_count = 0;
    uint64_t truth_mismatches = 0;
    double preload_seconds = 0.0;
    double observation_seconds = 0.0;
    double total_seconds = 0.0;
};

template <class Generator>
static ExactDictResult run_exact_dictionary(Generator generator, uint64_t H, uint64_t T,
                                            const vector<uint64_t>& release_slots) {
    using Set = unordered_set<uint64_t, hash<uint64_t>, equal_to<uint64_t>,
                              CountingAllocator<uint64_t>>;
    ExactDictResult result;
    result.status = "completed";
    auto tracker = make_shared<AllocationTracker>();
    CountingAllocator<uint64_t> allocator(tracker);
    Set active(0, hash<uint64_t>{}, equal_to<uint64_t>{}, allocator);
    active.max_load_factor(1.0f);

    const auto started = chrono::steady_clock::now();
    chrono::steady_clock::time_point preload_finished = started;
    size_t release_cursor = 0;
    for (uint64_t t = 1; t <= T; ++t) {
        const Event event = generator.next();
        const auto [iterator, inserted] = active.insert(event.item);
        if (!inserted) active.erase(iterator);
        result.peak_size = max<uint64_t>(result.peak_size, active.size());
        result.peak_bucket_count = max<uint64_t>(result.peak_bucket_count, active.bucket_count());
        if (release_cursor < release_slots.size() && t == release_slots[release_cursor]) {
            if (active.size() != event.truth) ++result.truth_mismatches;
            ++release_cursor;
        }
        if (t == H) preload_finished = chrono::steady_clock::now();
    }
    const auto finished = chrono::steady_clock::now();
    if (H == 0) preload_finished = started;
    result.preload_seconds = chrono::duration<double>(preload_finished - started).count();
    result.observation_seconds = chrono::duration<double>(finished - preload_finished).count();
    result.total_seconds = chrono::duration<double>(finished - started).count();
    result.peak_state_bytes = sizeof(Set) + tracker->peak;
    return result;
}

struct TimingResult {
    double preload_seconds = 0.0;
    double observation_seconds = 0.0;
    double release_seconds = 0.0;
    double total_seconds = 0.0;
};

template <class Generator>
static TimingResult run_sketches(const Args& args, Generator generator, uint64_t H,
                                 uint64_t T, const vector<uint64_t>& release_slots,
                                 vector<Trial>& trials, ReleaseCsv& release_writer) {
    TimingResult timing;
    const auto started = chrono::steady_clock::now();
    chrono::steady_clock::time_point preload_finished = started;
    size_t release_cursor = 0;
    for (uint64_t t = 1; t <= T; ++t) {
        const Event event = generator.next();
        for (Trial& trial : trials) trial.update(event.item, event.op);

        if (release_cursor < release_slots.size() && t == release_slots[release_cursor]) {
            const auto release_started = chrono::steady_clock::now();
            for (Trial& trial : trials)
                trial.release(t, release_cursor + 1ull, event.truth, release_writer);
            timing.release_seconds +=
                chrono::duration<double>(chrono::steady_clock::now() - release_started).count();
            ++release_cursor;
        }
        if (t == H) preload_finished = chrono::steady_clock::now();
        if (args.progress_step > 0 && t % args.progress_step == 0) {
            const double elapsed = chrono::duration<double>(chrono::steady_clock::now() - started).count();
            cerr << "progress=" << t << '/' << T << ", elapsed=" << fixed << setprecision(2)
                 << elapsed << " s\n";
        }
    }
    const auto finished = chrono::steady_clock::now();
    if (release_cursor != release_slots.size()) throw runtime_error("not all releases were emitted");
    if (H == 0) preload_finished = started;
    timing.preload_seconds = chrono::duration<double>(preload_finished - started).count();
    timing.observation_seconds = chrono::duration<double>(finished - preload_finished).count();
    timing.total_seconds = chrono::duration<double>(finished - started).count();
    return timing;
}

static string base_name(const string& path) {
    return filesystem::path(path).filename().string();
}

static void write_config(const Args& args, uint64_t H, uint64_t T,
                         const vector<uint64_t>& release_slots, int width,
                         uint64_t rows, uint64_t lifecycle_insertions,
                         uint64_t lifecycle_deletions) {
    ofstream output(args.output_prefix + "_config.txt");
    if (!output) throw runtime_error("cannot create config file");
    output << "mode=" << args.mode << '\n'
           << "schedule=" << args.schedule << '\n'
           << "N=" << args.N << '\n'
           << "Nmax=" << args.n_max << '\n'
           << "memory_kb=" << args.memory_kb << '\n'
           << "w=" << width << '\n'
           << "m=" << rows << '\n'
           << "T_total=" << T << '\n'
           << "H=" << H << '\n'
           << "L=" << (T - H) << '\n'
           << "K=" << release_slots.size() << '\n'
           << "B=" << args.B << '\n'
           << "epsilon=" << setprecision(17) << args.epsilon << '\n'
           << "delta=" << args.delta << '\n'
           << "trials=" << args.trials << '\n'
           << "completion_requested=" << args.completion << '\n'
           << "lifecycle_insertions=" << lifecycle_insertions << '\n'
           << "lifecycle_deletions=" << lifecycle_deletions << '\n'
           << "completion_actual="
           << (lifecycle_insertions ? static_cast<double>(lifecycle_deletions) /
                                          lifecycle_insertions
                                    : 0.0)
           << '\n';
}

static void write_summary(const Args& args, uint64_t H, uint64_t T,
                          const vector<uint64_t>& release_slots, int width,
                          uint64_t rows, const vector<Trial>& trials,
                          const TimingResult& timing, const ExactDictResult& exact_dict) {
    ofstream output(args.output_prefix + "_summary.csv");
    if (!output) throw runtime_error("cannot create summary CSV");
    output << "experiment,run_id,method,input_model,calibration,trial,seed_hash,seed_noise,"
              "N,Nmax,payload_bytes,peak_state_bytes,w,m,T_total,H,L,K,B,epsilon,delta,"
              "scalar_sensitivity,l2_sensitivity,sigma,RMSD,RMSE,clean_RMSD,clean_RMSE,"
              "positive_release_count,release_count,zero_release_count,zero_RMSE,clipped_count,"
              "preload_seconds,observation_seconds,release_seconds,total_seconds,"
              "exact_dict_seconds,exact_dict_peak_size,exact_dict_peak_bucket_count,status\n";
    const string run_id = base_name(args.output_prefix);
    for (const Trial& trial : trials) {
        for (size_t index = 0; index < trial.methods().size(); ++index) {
            MethodState method = trial.methods()[index];
            if (index == Trial::EXACT_DICT) {
                if (!args.run_exact_dict) continue;
                method.peak_state_bytes = exact_dict.peak_state_bytes;
            }
            const string status = index == Trial::EXACT_DICT ? exact_dict.status : "completed";
            output << args.mode << ',' << run_id << ',' << method.name << ','
                   << method.input_model << ',' << method.calibration << ',' << trial.trial()
                   << ',' << trial.seed_hash() << ',' << trial.seed_noise() << ',' << args.N << ','
                   << args.n_max << ',' << method.payload_bytes << ',' << method.peak_state_bytes
                   << ',' << width << ',' << rows << ',' << T << ',' << H << ',' << (T - H)
                   << ',' << release_slots.size() << ',' << args.B << ',' << number(args.epsilon)
                   << ',' << number(args.delta) << ',' << number(method.scalar_sensitivity) << ','
                   << number(method.l2_sensitivity) << ',' << number(method.sigma) << ','
                   << number(method.private_metrics.rmsd()) << ','
                   << number(method.private_metrics.rmse()) << ','
                   << number(method.clean_metrics.rmsd()) << ','
                   << number(method.clean_metrics.rmse()) << ','
                   << method.private_metrics.positive_count << ','
                   << method.private_metrics.all_count << ',' << method.private_metrics.zero_count
                   << ',' << number(method.private_metrics.zero_rmse()) << ','
                   << method.clipped_count << ',' << number(timing.preload_seconds) << ','
                   << number(timing.observation_seconds) << ',' << number(timing.release_seconds)
                   << ',' << number(timing.total_seconds) << ','
                   << number(exact_dict.total_seconds) << ',' << exact_dict.peak_size << ','
                   << exact_dict.peak_bucket_count << ',' << status << '\n';
        }
    }
}

static void write_status(const Args& args, uint64_t T, uint64_t K,
                         const TimingResult& timing, const ExactDictResult& exact_dict) {
    ofstream output(args.output_prefix + "_status.csv");
    if (!output) throw runtime_error("cannot create status CSV");
    output << "experiment,run_id,component,status,T_total,K,total_seconds,peak_state_bytes,"
              "truth_mismatches\n";
    const string run_id = base_name(args.output_prefix);
    output << args.mode << ',' << run_id << ",sketch_suite,completed," << T << ',' << K << ','
           << number(timing.total_seconds) << ",,0\n";
    output << args.mode << ',' << run_id << ",exact_dict," << exact_dict.status << ',' << T
           << ',' << K << ',' << number(exact_dict.total_seconds) << ','
           << exact_dict.peak_state_bytes << ',' << exact_dict.truth_mismatches << '\n';
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int index = 1; index < argc; ++index) {
        const string key = argv[index];
        auto value = [&]() -> string {
            if (index + 1 >= argc) throw invalid_argument("missing value for " + key);
            return argv[++index];
        };
        if (key == "--mode") args.mode = value();
        else if (key == "--schedule") args.schedule = value();
        else if (key == "--output-prefix") args.output_prefix = value();
        else if (key == "--N") args.N = stoull(value());
        else if (key == "--T") args.T = stoull(value());
        else if (key == "--L") args.L = stoull(value());
        else if (key == "--Nmax") args.n_max = stoull(value());
        else if (key == "--B") args.B = stoull(value());
        else if (key == "--K") args.K = stoull(value());
        else if (key == "--memory-kb") args.memory_kb = stoi(value());
        else if (key == "--width") args.width = stoi(value());
        else if (key == "--trials") args.trials = stoi(value());
        else if (key == "--epsilon") args.epsilon = stod(value());
        else if (key == "--delta") args.delta = stod(value());
        else if (key == "--completion") args.completion = stod(value());
        else if (key == "--seed-hash") args.seed_hash = stoull(value());
        else if (key == "--seed-noise") args.seed_noise = stoull(value());
        else if (key == "--progress-step") args.progress_step = stoull(value());
        else if (key == "--exact-dict") args.run_exact_dict = stoi(value()) != 0;
        else throw invalid_argument("unknown argument: " + key);
    }
    if (args.memory_kb <= 0 || args.trials <= 0 || args.B == 0 || args.K == 0 ||
        !(args.epsilon > 0.0) || !(args.delta > 0.0 && args.delta < 1.0))
        throw invalid_argument("invalid numeric arguments");
    return args;
}

static int run_selftest() {
    if (select_public_width(32, 1000000) != 7) throw runtime_error("width audit failed at 1e6");
    if (select_public_width(32, 2000000) != 9) throw runtime_error("width audit failed at 2e6");

    PackedGeometricHfx collision_test(1, 2, 100);
    collision_test.update_hashed(5, 8);
    collision_test.update_hashed(5, 8);
    if (collision_test.weight() != 0.0) throw runtime_error("collision parity test failed");

    const vector<vector<uint64_t>> streams = {
        {1, 1},
        {1, 2, 1, 2},
        {7, 7, 7, 7},
        {1, 2, 3, 1, 4, 2, 5, 3, 4, 5},
    };
    for (const auto& stream : streams) {
        PackedGeometricHfx hfx(1, 3, 100);
        unordered_set<uint64_t> active;
        vector<uint8_t> reference(hfx.logical_cells(), 0u);
        for (uint64_t raw : stream) {
            const uint64_t item = public_permute(raw);
            const auto [h1, h2] = hash_pair(item, 11);
            const uint64_t row = h1 % hfx.rows();
            const int column = min(ctz64(h2), 2);
            const uint64_t cell = row * 3ull + static_cast<uint64_t>(column);
            reference[cell] ^= 1u;
            hfx.update_hashed(h1, h2);
            const auto [iterator, inserted] = active.insert(item);
            if (!inserted) active.erase(iterator);
            const double expected_weight = static_cast<double>(
                count(reference.begin(), reference.end(), static_cast<uint8_t>(1)));
            if (hfx.weight() != expected_weight) throw runtime_error("clean-state parity test failed");
        }
        if (!active.empty()) throw runtime_error("small-stream truth test failed");
    }

    Args schedule_args;
    schedule_args.mode = "active-scale";
    schedule_args.schedule = "fixed-k";
    schedule_args.K = 200;
    const auto fixed_k = make_release_slots(schedule_args, 1000, 2001000);
    if (fixed_k.size() != 200 || fixed_k.front() != 11000 || fixed_k.back() != 2001000)
        throw runtime_error("fixed-k schedule test failed");
    schedule_args.schedule = "fixed-b";
    schedule_args.B = 10000;
    const auto fixed_b = make_release_slots(schedule_args, 100000, 1000000);
    if (fixed_b.size() != 90 || fixed_b.back() != 1000000)
        throw runtime_error("fixed-b schedule test failed");

    const double sigma = analytic_gaussian_sigma(1.0, 1e-6, sqrt(200.0));
    if (!(sigma > 0.0) || analytic_gaussian_delta(1.0, sqrt(200.0) / sigma) > 1.000001e-6)
        throw runtime_error("Gaussian calibration test failed");

    cout << "SELFTEST PASS: parity, collision, schedules, width rule, and Gaussian calibration\n";
    return 0;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    try {
        Args args = parse_args(argc, argv);
        if (args.mode == "selftest") return run_selftest();

        uint64_t H = args.mode == "lifecycle" ? 0ull : args.N;
        uint64_t T_total = 0;
        if (args.mode == "active-scale" || args.mode == "parameter")
            T_total = args.N + args.L;
        else if (args.mode == "long-stream")
            T_total = args.T;
        else if (args.mode == "lifecycle")
            T_total = args.T;
        else
            throw invalid_argument("mode must be selftest, active-scale, parameter, long-stream, or lifecycle");
        if (T_total <= H && args.mode != "lifecycle") throw invalid_argument("T must exceed H");
        if (args.n_max == 0) args.n_max = args.mode == "lifecycle" ? T_total : 2ull * args.N;
        const int width = args.width == 0 ? select_public_width(args.memory_kb, args.n_max)
                                          : args.width;
        const uint64_t rows = packed_rows(args.memory_kb, width);
        const vector<uint64_t> release_slots = make_release_slots(args, H, T_total);

        filesystem::path prefix(args.output_prefix);
        if (prefix.has_parent_path()) filesystem::create_directories(prefix.parent_path());
        const string run_id = base_name(args.output_prefix);
        ReleaseCsv release_writer(args.output_prefix + "_releases.csv", args.mode, run_id, args,
                                  H, T_total, release_slots.size(), width, rows);

        vector<Trial> trials;
        trials.reserve(static_cast<size_t>(args.trials));
        for (int trial = 0; trial < args.trials; ++trial)
            trials.emplace_back(trial, args, T_total, release_slots.size(), width,
                                args.run_exact_dict);

        TimingResult timing;
        ExactDictResult exact_dict;
        uint64_t lifecycle_insertions = 0;
        uint64_t lifecycle_deletions = 0;
        if (args.mode == "lifecycle") {
            LifecycleGenerator generator(T_total, args.completion);
            lifecycle_insertions = generator.insertions();
            lifecycle_deletions = generator.deletions();
            timing = run_sketches(args, generator, H, T_total, release_slots, trials,
                                  release_writer);
            if (args.run_exact_dict) {
                LifecycleGenerator exact_generator(T_total, args.completion);
                exact_dict = run_exact_dictionary(exact_generator, H, T_total, release_slots);
            }
        } else {
            ReplacementGenerator generator(args.N, T_total);
            timing = run_sketches(args, generator, H, T_total, release_slots, trials,
                                  release_writer);
            if (args.run_exact_dict) {
                ReplacementGenerator exact_generator(args.N, T_total);
                exact_dict = run_exact_dictionary(exact_generator, H, T_total, release_slots);
            }
        }
        if (!args.run_exact_dict) exact_dict.status = "not_run_resource_boundary";
        if (exact_dict.truth_mismatches != 0) throw runtime_error("ExactDict truth mismatch");

        write_config(args, H, T_total, release_slots, width, rows, lifecycle_insertions,
                     lifecycle_deletions);
        write_summary(args, H, T_total, release_slots, width, rows, trials, timing, exact_dict);
        write_status(args, T_total, release_slots.size(), timing, exact_dict);

        cout << "run_id=" << run_id << ", mode=" << args.mode << ", T=" << T_total
             << ", K=" << release_slots.size() << ", memory_kb=" << args.memory_kb
             << ", Nmax=" << args.n_max << ", w=" << width << ", m=" << rows
             << ", trials=" << args.trials << ", elapsed=" << setprecision(8)
             << timing.total_seconds << " s, exact_dict=" << exact_dict.status << '\n';
        return 0;
    } catch (const exception& error) {
        cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
