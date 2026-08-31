#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

static constexpr uint64_t GOLDEN64 = 0x9E3779B97F4A7C15ull;

struct Args {
    vector<uint64_t> T{20000};
    vector<int> memory_kb{8};
    vector<double> epsilon{1.0};
    int trials = 1;
    string methods = "hfx";
    string output = "results/experiment1_active_set_cpp.csv";
    string input_binary = "";
    string dataset = "synthetic_active_set";
    uint64_t n_max = 0;
    double p_ins = 0.55;
    double delta = 1e-6;
    int q = 8;
    int w = 32;
    double column_cap_scale = 2.0;
    uint64_t eval_every = 0;
    uint64_t ignore = 1000;
    uint64_t seed_stream = 4801289;
    uint64_t seed_hash = 63242691;
    uint64_t seed_noise = 981723641;
    string hfx_noise_mode = "load_aware";
    bool resume = false;
    bool batch_stream = false;
    bool delete_heavy = false;
    vector<double> deletion_ratio{0.5};
};

struct Result {
    string method;
    double rmsd = 0.0;
    double relative_error = 0.0;
    uint64_t accepted = 0;
    uint64_t filtered = 0;
    uint64_t saturated = 0;
    uint64_t num_cells = 0;
    double load_p50 = numeric_limits<double>::quiet_NaN();
    double load_p95 = numeric_limits<double>::quiet_NaN();
    double load_p99 = numeric_limits<double>::quiet_NaN();
    double load_max = numeric_limits<double>::quiet_NaN();
    double throughput_mops = 0.0;
    string column_q_schedule;
    double sigma_node = numeric_limits<double>::quiet_NaN();
    uint64_t clipped_evals = 0;
    uint64_t eval_samples = 0;
};

static uint64_t splitmix64(uint64_t x) {
    x = (x + GOLDEN64);
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull);
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EBull);
    return x ^ (x >> 31);
}

static pair<uint64_t, uint64_t> hash_pair(uint64_t item, uint64_t seed) {
    uint64_t h1 = splitmix64(item ^ (seed * GOLDEN64));
    uint64_t h2 = splitmix64(h1 ^ (seed << 32) ^ 0xD1B54A32D192ED03ull);
    return {h1, h2};
}

static int ctz64(uint64_t x) {
    if (x == 0) return 64;
    return __builtin_ctzll(x);
}

static int trailing_ones32(uint32_t x) {
    uint32_t first_zero = ~x;
    if (first_zero == 0) return 32;
    return ctz64(first_zero);
}

static vector<string> split(const string& s, char delim) {
    vector<string> out;
    string item;
    stringstream ss(s);
    while (getline(ss, item, delim)) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

template <class T>
static vector<T> parse_list(const string& s) {
    vector<T> out;
    for (const auto& item : split(s, ',')) {
        stringstream ss(item);
        T value{};
        ss >> value;
        out.push_back(value);
    }
    return out;
}

static double epsilon_delta_to_rho(double epsilon, double delta) {
    double log_delta = log(1.0 / delta);
    return pow(sqrt(log_delta + epsilon) - sqrt(log_delta), 2.0);
}

static int tree_height(uint64_t T) {
    return max(1, (int)ceil(log2((double)T + 1.0)));
}

static uint64_t default_nmax(uint64_t T) {
    if (T < 10000000ull) return max<uint64_t>(1000, T / 10);
    if (T == 10000000ull) return 1000000ull;
    if (T == 100000000ull) return 10000000ull;
    if (T == 1000000000ull) return 100000000ull;
    return max<uint64_t>(1000, T / 10);
}

static int choose_hfx_q(uint64_t T, int memory_kb, int w) {
    double memory_bits = (double)memory_kb * 1024.0 * 8.0;
    double tree_bits = 2.0 * (double)(tree_height(T) + 1) * 64.0;
    for (int q = 4; q <= 30; ++q) {
        double available = max(0.0, memory_bits - tree_bits);
        double num_cells = floor(floor(available / (1.0 + q)) / (double)w) * (double)w;
        if (num_cells < w) num_cells = w;
        double p99_load_estimate = 18.0 * (double)T / num_cells;
        if (pow(2.0, q) - 1.0 >= p99_load_estimate) return q;
    }
    return 30;
}

static uint64_t coprime_multiplier(uint64_t n, uint64_t seed) {
    uint64_t a = splitmix64(seed) % max<uint64_t>(1, n);
    if (a == 0) a = 1;
    while (std::gcd(a, n) != 1) {
        a++;
        if (a >= n) a = 1;
    }
    return a;
}

static bool is_hfx_method(const string& method) {
    return method == "hfx" || method == "hfx_checkpoint" || method == "hfx_checkpoint_ideal" || method == "hfx_column" || method == "hfx_column_unfiltered" || method == "hfx_column_ideal" || method == "hfx_column_filtered" || method == "hfx_full" || method == "hfx_filtered" || method == "hfx_ideal";
}

static double hfx_sigma(uint64_t T, int q, double eps, double delta, const string& noise, bool enable_noise) {
    if (!enable_noise) return 0.0;
    int H = tree_height(T);
    double scale = 0.0;
    if (noise == "load_aware" || noise == "theorem") {
        // R = 2^q - 1, so R + 1 = 2^q.  This is the Gaussian
        // calibration proved for the Gaussian mechanism in the paper.
        scale = 2.0 * (double)(1ull << q) * H;
    } else if (noise == "unit_diagnostic" || noise == "unit") {
        // Historical sensitivity diagnostic.  This mode is intentionally not
        // advertised as satisfying the load-aware HFX theorem.
        scale = 2.0 * H;
    } else {
        throw invalid_argument("unknown --hfx-noise-mode: " + noise);
    }
    return sqrt(scale / epsilon_delta_to_rho(eps, delta));
}

static double percentile(vector<uint32_t> values, double q) {
    if (values.empty()) return 0.0;
    sort(values.begin(), values.end());
    double pos = (q / 100.0) * (values.size() - 1);
    size_t lo = (size_t)floor(pos);
    size_t hi = (size_t)ceil(pos);
    if (lo == hi) return values[lo];
    return values[lo] * (hi - pos) + values[hi] * (pos - lo);
}

class GaussianBinaryTree {
public:
    int H;
    vector<double> U, P;
    normal_distribution<double> normal;
    mt19937_64 rng;
    double active_noisy_sum = 0.0;

    GaussianBinaryTree(uint64_t T, double sigma, uint64_t seed)
        : H(tree_height(T)), U(H + 1, 0.0), P(H + 1, 0.0), normal(0.0, sigma), rng(seed) {}

    double update(uint64_t t, double impulse) {
        int idx = ctz64(t);
        double partial = impulse;
        for (int i = 0; i < idx; ++i) {
            partial += U[i];
            active_noisy_sum -= P[i];
            U[i] = 0.0;
            P[i] = 0.0;
        }
        U[idx] = partial;
        P[idx] = partial + normal(rng);
        active_noisy_sum += P[idx];
        return active_noisy_sum;
    }
};

class SharedGaussianBinaryTree {
public:
    int H;
    vector<double> U, P_signal, P_noise;
    normal_distribution<double> normal;
    mt19937_64 rng;
    double last_signal = 0.0, last_noise = 0.0;
    double active_signal_sum = 0.0, active_noise_sum = 0.0;

    SharedGaussianBinaryTree(uint64_t T, uint64_t seed)
        : H(tree_height(T)), U(H + 1, 0.0), P_signal(H + 1, 0.0), P_noise(H + 1, 0.0), normal(0.0, 1.0), rng(seed) {}

    void update(uint64_t t, double impulse) {
        int idx = ctz64(t);
        double partial = impulse;
        for (int i = 0; i < idx; ++i) {
            partial += U[i];
            active_signal_sum -= P_signal[i];
            active_noise_sum -= P_noise[i];
            U[i] = 0.0;
            P_signal[i] = 0.0;
            P_noise[i] = 0.0;
        }
        U[idx] = partial;
        P_signal[idx] = partial;
        P_noise[idx] = normal(rng);
        active_signal_sum += P_signal[idx];
        active_noise_sum += P_noise[idx];
        last_signal = active_signal_sum;
        last_noise = active_noise_sum;
    }

    double value(double sigma) const {
        return last_signal + sigma * last_noise;
    }
};

class PackedCounters {
public:
    int bits = 1;
    uint64_t mask = 1;
    size_t count = 0;
    vector<uint64_t> words;

    PackedCounters() = default;

    void reset(size_t count_, int bits_) {
        if (bits_ < 1 || bits_ > 30) throw invalid_argument("counter width must be in [1,30]");
        bits = bits_;
        mask = (1ull << bits) - 1ull;
        count = count_;
        words.assign((count * (size_t)bits + 63) / 64, 0ull);
    }

    uint32_t get(size_t index) const {
        const size_t bit = index * (size_t)bits;
        const size_t word = bit >> 6;
        const int offset = (int)(bit & 63ull);
        if (offset + bits <= 64) return (uint32_t)((words[word] >> offset) & mask);
        const int low_bits = 64 - offset;
        const uint64_t low = words[word] >> offset;
        const uint64_t high = words[word + 1] & ((1ull << (bits - low_bits)) - 1ull);
        return (uint32_t)((high << low_bits) | low);
    }

    void set(size_t index, uint32_t value) {
        const uint64_t clipped = (uint64_t)value & mask;
        const size_t bit = index * (size_t)bits;
        const size_t word = bit >> 6;
        const int offset = (int)(bit & 63ull);
        if (offset + bits <= 64) {
            const uint64_t shifted_mask = mask << offset;
            words[word] = (words[word] & ~shifted_mask) | (clipped << offset);
            return;
        }
        const int low_bits = 64 - offset;
        const int high_bits = bits - low_bits;
        const uint64_t low_mask = (1ull << low_bits) - 1ull;
        const uint64_t high_mask = (1ull << high_bits) - 1ull;
        words[word] = (words[word] & ~(low_mask << offset)) | ((clipped & low_mask) << offset);
        words[word + 1] = (words[word + 1] & ~high_mask) | ((clipped >> low_bits) & high_mask);
    }
};

class HfxEstimator {
public:
    uint64_t m, w, n_max;
    vector<double> log_terms;
    double max_phi;

    HfxEstimator(uint64_t m_, uint64_t w_, uint64_t n_max_) : m(m_), w(w_), n_max(n_max_) {
        log_terms.reserve(w);
        for (uint64_t j = 1; j <= w; ++j) {
            double p = (j == w) ? pow(2.0, -(double)(w - 1)) : pow(2.0, -(double)j);
            log_terms.push_back(log1p(-2.0 * p / (double)m));
        }
        max_phi = phi((double)n_max);
    }

    double phi(double n) const {
        double sum = 0.0;
        for (double lt : log_terms) sum += -expm1(n * lt);
        return 0.5 * (double)m * sum;
    }

    double invert(double z_observed) const {
        double z = min(max(z_observed, 0.0), max_phi);
        double lo = 0.0, hi = (double)n_max;
        for (int i = 0; i < 48; ++i) {
            double mid = 0.5 * (lo + hi);
            if (phi(mid) < z) lo = mid;
            else hi = mid;
        }
        return 0.5 * (lo + hi);
    }
};

class WeightedHfxEstimator {
public:
    uint64_t m, w, n_max;
    vector<double> log_terms, weights;
    double max_phi;

    WeightedHfxEstimator(uint64_t m_, uint64_t w_, uint64_t n_max_, vector<double> weights_)
        : m(m_), w(w_), n_max(n_max_), weights(std::move(weights_)) {
        log_terms.reserve(w);
        for (uint64_t j = 1; j <= w; ++j) {
            double p = (j == w) ? pow(2.0, -(double)(w - 1)) : pow(2.0, -(double)j);
            log_terms.push_back(log1p(-2.0 * p / (double)m));
        }
        max_phi = phi((double)n_max);
    }

    double phi(double n) const {
        double sum = 0.0;
        for (size_t j = 0; j < log_terms.size(); ++j) {
            sum += weights[j] * -expm1(n * log_terms[j]);
        }
        return 0.5 * (double)m * sum;
    }

    double invert(double z_observed) const {
        double z = min(max(z_observed, 0.0), max_phi);
        double lo = 0.0, hi = (double)n_max;
        for (int i = 0; i < 48; ++i) {
            double mid = 0.5 * (lo + hi);
            if (phi(mid) < z) lo = mid;
            else hi = mid;
        }
        return 0.5 * (lo + hi);
    }
};

class Method {
public:
    uint64_t T = 0, n_max = 0;
    double last_estimate = 0.0;
    uint64_t clipped_evals = 0, eval_samples = 0;
    virtual ~Method() = default;
    virtual void update(uint64_t t, uint64_t item, int op) = 0;
    virtual double estimate() = 0;
    virtual Result finish(double rmsd, uint64_t final_true, double elapsed) = 0;
};

class HFXGaussian : public Method {
public:
    int q, w, H;
    uint64_t R;
    uint64_t memory_kb, num_cells, m, seed_hash, accepted = 0, filtered = 0;
    bool enable_filter = true;
    string method_name = "HFX";
    vector<uint32_t> B, raw_load;
    PackedCounters V;
    GaussianBinaryTree tree;
    HfxEstimator estimator;
    double last_z = 0.0;

    HFXGaussian(uint64_t T_, uint64_t nmax, int mem, double eps, double delta, int q_, int w_, uint64_t sh, uint64_t sn, const string& noise, bool filter_on = true, bool noise_on = true, string name = "HFX")
        : q(q_), w(w_), H(tree_height(T_)), R((1ull << q_) - 1ull), memory_kb(mem), seed_hash(sh),
          enable_filter(filter_on), method_name(std::move(name)),
          tree(T_, hfx_sigma(T_, q_, eps, delta, noise, noise_on), sn),
          estimator(1, w_, nmax) {
        T = T_;
        n_max = nmax;
        uint64_t memory_bits = (uint64_t)mem * 1024ull * 8ull;
        uint64_t tree_bits = 2ull * (H + 1ull) * 64ull;
        uint64_t available = memory_bits > tree_bits ? memory_bits - tree_bits : 0;
        num_cells = max<uint64_t>(w, (available / (1 + q) / w) * w);
        m = max<uint64_t>(1, num_cells / w);
        num_cells = m * w;
        B.assign(m, 0);
        V.reset(num_cells, q);
        raw_load.assign(num_cells, 0);
        estimator = HfxEstimator(m, w, nmax);
    }

    void update(uint64_t t, uint64_t item, int) override {
        auto [h1, h2] = hash_pair(item, seed_hash);
        uint64_t row = h1 % m;
        int col = min(ctz64(h2), w - 1);
        uint64_t coord = row * w + col;
        raw_load[coord]++;
        double impulse = 0.0;
        uint32_t load = V.get(coord);
        if (enable_filter && load >= R) {
            filtered++;
        } else {
            if (load < R) V.set(coord, load + 1);
            uint32_t bit = 1u << col;
            bool old_one = (B[row] & bit) != 0;
            B[row] ^= bit;
            impulse = old_one ? -1.0 : 1.0;
            accepted++;
        }
        last_z = tree.update(t, impulse);
    }

    double estimate() override {
        eval_samples++;
        if (last_z < 0.0 || last_z > estimator.max_phi) clipped_evals++;
        last_estimate = estimator.invert(last_z);
        return last_estimate;
    }

    Result finish(double rmsd, uint64_t final_true, double elapsed) override {
        Result r;
        r.method = method_name;
        r.rmsd = rmsd;
        r.relative_error = final_true ? (last_estimate - final_true) / (double)final_true : 0.0;
        r.accepted = accepted;
        r.filtered = filtered;
        r.num_cells = num_cells;
        vector<uint32_t> nonzero;
        nonzero.reserve(raw_load.size());
        for (size_t i = 0; i < V.count; ++i) {
            if (V.get(i) == R) r.saturated++;
            if (raw_load[i] > 0) nonzero.push_back(raw_load[i]);
        }
        r.load_p50 = percentile(nonzero, 50);
        r.load_p95 = percentile(nonzero, 95);
        r.load_p99 = percentile(nonzero, 99);
        r.load_max = nonzero.empty() ? 0.0 : *max_element(nonzero.begin(), nonzero.end());
        r.throughput_mops = elapsed > 0 ? T / elapsed / 1000000.0 : 0.0;
        return r;
    }
};

class HFXColumnGaussian : public Method {
public:
    int q_max, w, H;
    uint64_t memory_kb, num_cells, m, seed_hash, accepted = 0, filtered = 0;
    bool enable_filter;
    string method_name;
    vector<uint32_t> B, V, raw_load;
    vector<int> column_q;
    vector<uint32_t> column_R;
    vector<double> column_weight;
    GaussianBinaryTree tree;
    WeightedHfxEstimator estimator;
    double last_z = 0.0;

    static double column_probability(int col, int width) {
        return (col == width - 1) ? pow(2.0, -(double)(width - 1)) : pow(2.0, -(double)(col + 1));
    }

    static vector<int> make_schedule(uint64_t T, uint64_t rows, int width, int q_cap, double cap_scale) {
        vector<int> schedule(width, 1);
        double safe_rows = (double)max<uint64_t>(1, rows);
        for (int col = 0; col < width; ++col) {
            double expected_load = (double)T * column_probability(col, width) / safe_rows;
            int q = (int)ceil(log2(1.0 + cap_scale * expected_load));
            schedule[col] = max(1, min(q_cap, q));
        }
        return schedule;
    }

    static uint64_t choose_rows(uint64_t T, int mem, int width, int q_cap, double cap_scale) {
        uint64_t memory_bits = (uint64_t)mem * 1024ull * 8ull;
        uint64_t tree_bits = 2ull * (tree_height(T) + 1ull) * 64ull;
        uint64_t available = memory_bits > tree_bits ? memory_bits - tree_bits : 0;
        uint64_t rows = max<uint64_t>(1, available / max(1, width * (1 + q_cap)));
        for (int iteration = 0; iteration < 64; ++iteration) {
            vector<int> schedule = make_schedule(T, rows, width, q_cap, cap_scale);
            uint64_t bits_per_row = (uint64_t)width + accumulate(schedule.begin(), schedule.end(), 0ull);
            uint64_t next = max<uint64_t>(1, available / max<uint64_t>(1, bits_per_row));
            if (next == rows) break;
            rows = next;
        }
        auto fits = [&](uint64_t candidate) {
            vector<int> schedule = make_schedule(T, candidate, width, q_cap, cap_scale);
            uint64_t bits_per_row = (uint64_t)width + accumulate(schedule.begin(), schedule.end(), 0ull);
            return candidate * bits_per_row <= available;
        };
        while (rows > 1 && !fits(rows)) --rows;
        while (fits(rows + 1)) ++rows;
        return rows;
    }

    HFXColumnGaussian(uint64_t T_, uint64_t nmax, int mem, double eps, double delta, int q_cap, int w_, double cap_scale, uint64_t sh, uint64_t sn, bool filter_on = true, bool noise_on = true, string name = "HFX-Column")
        : q_max(max(1, min(30, q_cap))), w(w_), H(tree_height(T_)), memory_kb(mem),
          m(choose_rows(T_, mem, w_, max(1, min(30, q_cap)), cap_scale)), seed_hash(sh), enable_filter(filter_on), method_name(std::move(name)),
          column_q(make_schedule(T_, m, w_, max(1, min(30, q_cap)), cap_scale)),
          tree(T_, noise_on ? sqrt(2.0 * H / epsilon_delta_to_rho(eps, delta)) : 0.0, sn),
          estimator(1, w_, nmax, vector<double>(w_, 1.0)) {
        if (w <= 0 || w > 32) throw runtime_error("hfx_column requires 1 <= w <= 32");
        T = T_;
        n_max = nmax;
        num_cells = m * (uint64_t)w;
        B.assign(m, 0);
        V.assign(num_cells, 0);
        raw_load.assign(num_cells, 0);
        column_R.reserve(w);
        column_weight.reserve(w);
        for (int q : column_q) {
            uint32_t cap = (q >= 32) ? numeric_limits<uint32_t>::max() : ((1u << q) - 1u);
            column_R.push_back(cap);
            column_weight.push_back(1.0 / sqrt((double)cap + 1.0));
        }
        estimator = WeightedHfxEstimator(m, w, nmax, column_weight);
    }

    void update(uint64_t t, uint64_t item, int) override {
        auto [h1, h2] = hash_pair(item, seed_hash);
        uint64_t row = h1 % m;
        int col = min(ctz64(h2), w - 1);
        uint64_t coord = row * (uint64_t)w + (uint64_t)col;
        raw_load[coord]++;
        double impulse = 0.0;
        if (enable_filter && V[coord] >= column_R[col]) {
            filtered++;
        } else {
            if (V[coord] < column_R[col]) V[coord]++;
            uint32_t bit = 1u << col;
            bool old_one = (B[row] & bit) != 0;
            B[row] ^= bit;
            impulse = old_one ? -column_weight[col] : column_weight[col];
            accepted++;
        }
        last_z = tree.update(t, impulse);
    }

    double estimate() override {
        eval_samples++;
        if (last_z < 0.0 || last_z > estimator.max_phi) clipped_evals++;
        last_estimate = estimator.invert(last_z);
        return last_estimate;
    }

    Result finish(double rmsd, uint64_t final_true, double elapsed) override {
        Result r;
        r.method = method_name;
        r.rmsd = rmsd;
        r.relative_error = final_true ? (last_estimate - final_true) / (double)final_true : 0.0;
        r.accepted = accepted;
        r.filtered = filtered;
        r.num_cells = num_cells;
        vector<uint32_t> nonzero;
        nonzero.reserve(raw_load.size());
        for (uint64_t row = 0; row < m; ++row) {
            for (int col = 0; col < w; ++col) {
                uint64_t coord = row * (uint64_t)w + (uint64_t)col;
                if (V[coord] == column_R[col]) r.saturated++;
                if (raw_load[coord] > 0) nonzero.push_back(raw_load[coord]);
            }
        }
        r.load_p50 = percentile(nonzero, 50);
        r.load_p95 = percentile(nonzero, 95);
        r.load_p99 = percentile(nonzero, 99);
        r.load_max = nonzero.empty() ? 0.0 : *max_element(nonzero.begin(), nonzero.end());
        r.throughput_mops = elapsed > 0 ? T / elapsed / 1000000.0 : 0.0;
        ostringstream schedule;
        for (size_t j = 0; j < column_q.size(); ++j) {
            if (j) schedule << ';';
            schedule << column_q[j];
        }
        r.column_q_schedule = schedule.str();
        return r;
    }
};

class HFXCheckpointGaussian : public Method {
public:
    int w;
    uint64_t memory_kb, num_cells, m, seed_hash, reports, accepted = 0;
    vector<uint32_t> B;
    HfxEstimator estimator;
    normal_distribution<double> normal;
    mt19937_64 rng;
    string method_name;
    double exact_z = 0.0;

    HFXCheckpointGaussian(uint64_t T_, uint64_t nmax, int mem, double eps, double delta, int w_, uint64_t eval_every, uint64_t sh, uint64_t sn, bool noise_on = true, string name = "HFX-Checkpoint")
        : w(w_), memory_kb(mem), seed_hash(sh),
          reports((T_ + max<uint64_t>(1, eval_every) - 1) / max<uint64_t>(1, eval_every) + 1),
          estimator(1, w_, nmax),
          normal(0.0, noise_on ? sqrt((double)reports / (2.0 * epsilon_delta_to_rho(eps, delta))) : 0.0), rng(sn), method_name(std::move(name)) {
        if (w <= 0 || w > 32) throw runtime_error("hfx_checkpoint requires 1 <= w <= 32");
        T = T_;
        n_max = nmax;
        uint64_t memory_bits = (uint64_t)mem * 1024ull * 8ull;
        m = max<uint64_t>(1, memory_bits / (uint64_t)w);
        num_cells = m * (uint64_t)w;
        B.assign(m, 0);
        estimator = HfxEstimator(m, w, nmax);
    }

    void update(uint64_t, uint64_t item, int) override {
        auto [h1, h2] = hash_pair(item, seed_hash);
        uint64_t row = h1 % m;
        int col = min(ctz64(h2), w - 1);
        uint32_t bit = 1u << col;
        bool old_one = (B[row] & bit) != 0;
        B[row] ^= bit;
        exact_z += old_one ? -1.0 : 1.0;
        accepted++;
    }

    double estimate() override {
        double observed_z = exact_z + normal(rng);
        eval_samples++;
        if (observed_z < 0.0 || observed_z > estimator.max_phi) clipped_evals++;
        last_estimate = estimator.invert(observed_z);
        return last_estimate;
    }

    Result finish(double rmsd, uint64_t final_true, double elapsed) override {
        Result r;
        r.method = method_name;
        r.rmsd = rmsd;
        r.relative_error = final_true ? (last_estimate - final_true) / (double)final_true : 0.0;
        r.accepted = accepted;
        r.num_cells = num_cells;
        r.throughput_mops = elapsed > 0 ? T / elapsed / 1000000.0 : 0.0;
        return r;
    }
};

class SplitSketch : public Method {
public:
    uint64_t memory_bits, sketch_bits, seed_hash, num_cells = 0, num_inserts = 0, num_deletes = 0, current_t = 0;
    GaussianBinaryTree tree_add, tree_del;
    double last_add_z = 0.0, last_del_z = 0.0;

    SplitSketch(uint64_t T_, uint64_t nmax, int mem, double eps, double delta, double sensitivity, uint64_t sh, uint64_t sn)
        : memory_bits((uint64_t)mem * 1024ull * 8ull), sketch_bits(0), seed_hash(sh),
          tree_add(T_, sensitivity * sqrt(tree_height(T_) / epsilon_delta_to_rho(eps, delta)), sn),
          tree_del(T_, sensitivity * sqrt(tree_height(T_) / epsilon_delta_to_rho(eps, delta)), sn + 1000003ull) {
        T = T_;
        n_max = nmax;
        uint64_t tree_bits = 4ull * (tree_height(T_) + 1ull) * 64ull;
        sketch_bits = memory_bits > tree_bits ? memory_bits - tree_bits : 0;
    }
    virtual double insert_into_add(uint64_t item) = 0;
    virtual double insert_into_del(uint64_t item) = 0;
    virtual double estimate_one(double z, uint64_t seen) = 0;
    void update(uint64_t t, uint64_t item, int op) override {
        current_t = t;
        double da = 0.0, dd = 0.0;
        if (op > 0) {
            num_inserts++;
            da = insert_into_add(item);
        } else {
            num_deletes++;
            dd = insert_into_del(item);
        }
        last_add_z = tree_add.update(t, da);
        last_del_z = tree_del.update(t, dd);
    }
    double estimate() override {
        last_estimate = max(0.0, min<double>({(double)n_max, (double)current_t, estimate_one(last_add_z, num_inserts) - estimate_one(last_del_z, num_deletes)}));
        return last_estimate;
    }
    Result finish_base(const string& name, double rmsd, uint64_t final_true, double elapsed) {
        Result r;
        r.method = name;
        r.rmsd = rmsd;
        r.relative_error = final_true ? (last_estimate - final_true) / (double)final_true : 0.0;
        r.accepted = T;
        r.num_cells = num_cells;
        r.throughput_mops = elapsed > 0 ? T / elapsed / 1000000.0 : 0.0;
        return r;
    }
};

class FMTurnstile : public SplitSketch {
public:
    uint64_t m;
    vector<uint32_t> add_bitmap, del_bitmap;
    FMTurnstile(uint64_t T, uint64_t n, int mem, double eps, double delta, uint64_t sh, uint64_t sn)
        : SplitSketch(T, n, mem, eps, delta, 64.0, sh, sn) {
        m = max<uint64_t>(1, sketch_bits / 64ull);
        add_bitmap.assign(m, 0);
        del_bitmap.assign(m, 0);
        num_cells = 2 * m * 32;
    }
    double insert_into(vector<uint32_t>& bitmap, uint64_t item) {
        auto [h1, h2] = hash_pair(item, seed_hash);
        uint64_t row = h1 % m;
        int col = min(ctz64(h2), 31);
        uint32_t old = bitmap[row];
        int old_prefix = trailing_ones32(old);
        bitmap[row] = old | (1u << col);
        return trailing_ones32(bitmap[row]) - old_prefix;
    }
    double insert_into_add(uint64_t item) override { return insert_into(add_bitmap, item); }
    double insert_into_del(uint64_t item) override { return insert_into(del_bitmap, item); }
    double estimate_one(double z, uint64_t seen) override {
        if (seen == 0 || z <= 0.0) return 0.0;
        return max(0.0, min<double>(max(T, n_max), m * pow(2.0, z / m) / 0.77351));
    }
    Result finish(double rmsd, uint64_t final_true, double elapsed) override { return finish_base("FM-Turnstile", rmsd, final_true, elapsed); }
};

class LLTurnstile : public SplitSketch {
public:
    uint64_t m;
    vector<uint8_t> add_regs, del_regs;
    string name;
    LLTurnstile(uint64_t T, uint64_t n, int mem, double eps, double delta, uint64_t sh, uint64_t sn, string method_name = "LL-Turnstile", double sensitivity = 64.0)
        : SplitSketch(T, n, mem, eps, delta, sensitivity, sh, sn), name(std::move(method_name)) {
        // Two uint8_t register arrays consume 16 actual bits per row.
        m = max<uint64_t>(1, sketch_bits / 16ull);
        add_regs.assign(m, 0);
        del_regs.assign(m, 0);
        num_cells = 2 * m;
    }
    double insert_into(vector<uint8_t>& regs, uint64_t item) {
        auto [h1, h2] = hash_pair(item, seed_hash);
        uint64_t row = h1 % m;
        uint8_t rank = (uint8_t)min(ctz64(h2) + 1, 64);
        if (rank > regs[row]) {
            double d = rank - regs[row];
            regs[row] = rank;
            return d;
        }
        return 0.0;
    }
    double insert_into_add(uint64_t item) override { return insert_into(add_regs, item); }
    double insert_into_del(uint64_t item) override { return insert_into(del_regs, item); }
    double estimate_one(double z, uint64_t seen) override {
        if (seen == 0 || z <= 0.0) return 0.0;
        return max(0.0, min<double>(max(T, n_max), 0.39701 * m * pow(2.0, z / m)));
    }
    Result finish(double rmsd, uint64_t final_true, double elapsed) override { return finish_base(name, rmsd, final_true, elapsed); }
};

class HLLTurnstile : public LLTurnstile {
public:
    double alpha;
    HLLTurnstile(uint64_t T, uint64_t n, int mem, double eps, double delta, uint64_t sh, uint64_t sn)
        : LLTurnstile(T, n, mem, eps, delta, sh, sn, "HLL-Turnstile", 1.0) {
        alpha = (m == 16) ? 0.673 : (m == 32) ? 0.697 : (m == 64) ? 0.709 : 0.7213 / (1.0 + 1.079 / m);
    }
    double insert_into_hll(vector<uint8_t>& regs, uint64_t item) {
        auto [h1, h2] = hash_pair(item, seed_hash);
        uint64_t row = h1 % m;
        uint8_t rank = (uint8_t)min(ctz64(h2) + 1, 64);
        if (rank > regs[row]) {
            double d = pow(2.0, -regs[row]) - pow(2.0, -rank);
            regs[row] = rank;
            return d;
        }
        return 0.0;
    }
    double insert_into_add(uint64_t item) override { return insert_into_hll(add_regs, item); }
    double insert_into_del(uint64_t item) override { return insert_into_hll(del_regs, item); }
    double estimate_one(double z, uint64_t seen) override {
        if (seen == 0) return 0.0;
        double cap = (double)max(T, n_max);
        double inv = m - z;
        if (inv <= 0.0) return cap;
        return max(0.0, min(cap, alpha * m * m / inv));
    }
};

class FlipDPProxy : public Method {
public:
    uint64_t slots, seed_hash, current_t = 0;
    double sample_rate, sample_threshold, last_z = 0.0;
    vector<uint64_t> keys;
    vector<uint8_t> active, flips;
    GaussianBinaryTree tree;
    FlipDPProxy(uint64_t T_, uint64_t n, int mem, double eps, double delta, uint64_t sh, uint64_t sn)
        : seed_hash(sh),
          tree(T_, 1.0, sn) {
        T = T_;
        n_max = n;
        int H = tree_height(T);
        uint64_t memory_bits = (uint64_t)mem * 1024ull * 8ull;
        uint64_t tree_bits = 2ull * (H + 1ull) * 64ull;
        uint64_t available = memory_bits > tree_bits ? memory_bits - tree_bits : 0;
        // key (64 bits), active byte, and flip byte per slot.
        slots = max<uint64_t>(16, available / 80ull);
        sample_rate = min(1.0, max(1.0 / max<uint64_t>(1, n), 0.8 * slots / (double)max<uint64_t>(1, n)));
        double flip_sens = sqrt(16.0 * 2.0 * (H + 1.0) * (H + 1.0) / H) / sample_rate;
        tree = GaussianBinaryTree(T, flip_sens * sqrt(H / (2.0 * epsilon_delta_to_rho(eps, delta))), sn);
        sample_threshold = sample_rate * (double)numeric_limits<uint64_t>::max();
        keys.assign(slots, 0);
        active.assign(slots, 0);
        flips.assign(slots, 0);
    }
    void update(uint64_t t, uint64_t item, int op) override {
        current_t = t;
        auto [h1, h2] = hash_pair(item, seed_hash);
        if ((double)h2 >= sample_threshold) {
            last_z = tree.update(t, 0.0);
            return;
        }
        uint64_t start = h1 % slots;
        uint64_t key = h2 ? h2 : 1;
        int64_t chosen = -1, empty = -1;
        for (int off = 0; off < 16; ++off) {
            uint64_t slot = (start + off) % slots;
            if (keys[slot] == key) {
                chosen = (int64_t)slot;
                break;
            }
            if (keys[slot] == 0 && empty < 0) empty = (int64_t)slot;
        }
        if (chosen < 0 && op > 0 && empty >= 0) {
            chosen = empty;
            keys[chosen] = key;
        }
        double delta = 0.0;
        if (chosen >= 0) {
            if (op > 0 && !active[chosen] && flips[chosen] < 255) {
                active[chosen] = 1;
                flips[chosen]++;
                delta = 1.0 / sample_rate;
            } else if (op < 0 && active[chosen] && flips[chosen] < 255) {
                active[chosen] = 0;
                flips[chosen]++;
                keys[chosen] = 0;
                delta = -1.0 / sample_rate;
            }
        }
        last_z = tree.update(t, delta);
    }
    double estimate() override {
        last_estimate = max(0.0, min<double>({(double)n_max, (double)current_t, last_z}));
        return last_estimate;
    }
    Result finish(double rmsd, uint64_t final_true, double elapsed) override {
        Result r;
        r.method = "Flip-DP";
        r.rmsd = rmsd;
        r.relative_error = final_true ? (last_estimate - final_true) / (double)final_true : 0.0;
        r.accepted = T;
        r.num_cells = slots;
        r.throughput_mops = elapsed > 0 ? T / elapsed / 1000000.0 : 0.0;
        return r;
    }
};

struct StreamStats { uint64_t max_active = 0; double avg_active = 0.0; };

static pair<Result, StreamStats> run_one(const string& method, uint64_t T, uint64_t n_max, double p_ins, int mem, double eps, double delta, int q, int w, double column_cap_scale, uint64_t seed_stream, uint64_t seed_hash, uint64_t seed_noise, uint64_t eval_every, uint64_t ignore, const string& noise_mode) {
    unique_ptr<Method> m;
    if (method == "hfx") m = make_unique<HFXGaussian>(T, n_max, mem, eps, delta, q, w, seed_hash, seed_noise, noise_mode);
    else if (method == "hfx_checkpoint") m = make_unique<HFXCheckpointGaussian>(T, n_max, mem, eps, delta, w, eval_every, seed_hash, seed_noise);
    else if (method == "hfx_checkpoint_ideal") m = make_unique<HFXCheckpointGaussian>(T, n_max, mem, eps, delta, w, eval_every, seed_hash, seed_noise, false, "Checkpoint Ideal");
    else if (method == "hfx_column") m = make_unique<HFXColumnGaussian>(T, n_max, mem, eps, delta, q, w, column_cap_scale, seed_hash, seed_noise);
    else if (method == "hfx_column_unfiltered") m = make_unique<HFXColumnGaussian>(T, n_max, mem, eps, delta, q, w, column_cap_scale, seed_hash, seed_noise, false, true, "HFX-Column-Unfiltered");
    else if (method == "hfx_column_ideal") m = make_unique<HFXColumnGaussian>(T, n_max, mem, eps, delta, q, w, column_cap_scale, seed_hash, seed_noise, false, false, "Column Ideal");
    else if (method == "hfx_column_filtered") m = make_unique<HFXColumnGaussian>(T, n_max, mem, eps, delta, q, w, column_cap_scale, seed_hash, seed_noise, true, false, "Column Filtered");
    else if (method == "hfx_full") m = make_unique<HFXGaussian>(T, n_max, mem, eps, delta, q, w, seed_hash, seed_noise, noise_mode, true, true, "Full HFX");
    else if (method == "hfx_filtered") m = make_unique<HFXGaussian>(T, n_max, mem, eps, delta, q, w, seed_hash, seed_noise, noise_mode, true, false, "Filtered XOR");
    else if (method == "hfx_ideal") m = make_unique<HFXGaussian>(T, n_max, mem, eps, delta, q, w, seed_hash, seed_noise, noise_mode, false, false, "Ideal XOR");
    else if (method == "fm") m = make_unique<FMTurnstile>(T, n_max, mem, eps, delta, seed_hash, seed_noise);
    else if (method == "ll") m = make_unique<LLTurnstile>(T, n_max, mem, eps, delta, seed_hash, seed_noise);
    else if (method == "hll") m = make_unique<HLLTurnstile>(T, n_max, mem, eps, delta, seed_hash, seed_noise);
    else if (method == "flipdp") m = make_unique<FlipDPProxy>(T, n_max, mem, eps, delta, seed_hash, seed_noise);
    else throw runtime_error("unknown method");

    mt19937_64 rng(seed_stream);
    uniform_real_distribution<double> uni(0.0, 1.0);
    vector<uint64_t> active;
    active.reserve((size_t)min<uint64_t>(n_max, 10000000ull));
    uint64_t next_id = 1, active_sum = 0, max_active = 0, final_true = 0, samples = 0;
    double err2 = 0.0;
    auto start = chrono::steady_clock::now();
    for (uint64_t t = 1; t <= T; ++t) {
        bool do_insert = active.empty() || (active.size() < n_max && uni(rng) < p_ins);
        uint64_t item;
        int op;
        if (do_insert) {
            item = next_id++;
            active.push_back(item);
            op = 1;
        } else {
            uniform_int_distribution<uint64_t> pick(0, active.size() - 1);
            uint64_t idx = pick(rng);
            item = active[idx];
            uint64_t last = active.back();
            active[idx] = last;
            active.pop_back();
            op = -1;
        }
        uint64_t true_n = active.size();
        active_sum += true_n;
        max_active = max(max_active, true_n);
        m->update(t, item, op);
        final_true = true_n;
        if (t >= ignore && t % eval_every == 0 && true_n > 0) {
            double est = m->estimate();
            double rel = (est - true_n) / (double)true_n;
            err2 += rel * rel;
            samples++;
        }
    }
    double elapsed = chrono::duration<double>(chrono::steady_clock::now() - start).count();
    double rmsd = samples ? sqrt(err2 / samples) : 0.0;
    m->estimate();
    Result result = m->finish(rmsd, final_true, elapsed);
    result.clipped_evals = m->clipped_evals;
    result.eval_samples = m->eval_samples;
    return {result, {max_active, T ? active_sum / (double)T : 0.0}};
}

static string method_label(const string& key) {
    if (key == "hfx") return "HFX";
    if (key == "hfx_checkpoint") return "HFX-Checkpoint";
    if (key == "hfx_checkpoint_ideal") return "Checkpoint Ideal";
    if (key == "hfx_column") return "HFX-Column";
    if (key == "hfx_column_unfiltered") return "HFX-Column-Unfiltered";
    if (key == "hfx_column_ideal") return "Column Ideal";
    if (key == "hfx_column_filtered") return "Column Filtered";
    if (key == "hfx_full") return "Full HFX";
    if (key == "hfx_filtered") return "Filtered XOR";
    if (key == "hfx_ideal") return "Ideal XOR";
    if (key == "fm") return "FM-Turnstile";
    if (key == "ll") return "LL-Turnstile";
    if (key == "hll") return "HLL-Turnstile";
    if (key == "flipdp") return "Flip-DP";
    return key;
}

static string csv_num(double x) {
    if (isnan(x)) return "";
    ostringstream ss;
    ss << setprecision(17) << x;
    return ss.str();
}

static string mechanism_label(const string& method_key, const string& hfx_noise_mode) {
    if (method_key == "hfx_ideal") return "ablation-no_filter-no_noise";
    if (method_key == "hfx_filtered") return "ablation-filter-no_noise";
    if (method_key == "hfx_column_ideal") return "ablation-column-no_filter-no_noise";
    if (method_key == "hfx_column_filtered") return "ablation-column-filter-no_noise";
    if (method_key == "hfx_column_unfiltered") return "gaussian-column-random-load";
    if (method_key == "hfx_checkpoint") return "gaussian-checkpoint-vector";
    if (method_key == "hfx_checkpoint_ideal") return "ablation-checkpoint-no-noise";
    if (method_key == "hfx_full" || method_key == "hfx") return string("gaussian-") + hfx_noise_mode;
    if (method_key == "hfx_column") return "gaussian-column-weighted";
    return "gaussian-baseline";
}

static double method_sigma_node(const string& method_key, uint64_t T, int q, double eps, double delta, const string& hfx_noise_mode) {
    if (method_key == "hfx" || method_key == "hfx_full") {
        return hfx_sigma(T, q, eps, delta, hfx_noise_mode, true);
    }
    if (method_key == "hfx_filtered" || method_key == "hfx_ideal" ||
        method_key == "hfx_column_filtered" || method_key == "hfx_column_ideal" ||
        method_key == "hfx_checkpoint_ideal") {
        return 0.0;
    }
    if (method_key == "hfx_column" || method_key == "hfx_column_unfiltered") {
        return sqrt(2.0 * tree_height(T) / epsilon_delta_to_rho(eps, delta));
    }
    // Checkpoint noise is drawn directly per released vector, not per tree node;
    // baseline mechanisms use different sensitivities.  Leave sigma_node blank.
    return numeric_limits<double>::quiet_NaN();
}

static string make_key(uint64_t T, int trial, int q, int mem, double eps, const string& method) {
    ostringstream ss;
    ss << T << "|" << trial << "|" << q << "|" << mem << "|" << setprecision(17) << eps << "|" << method;
    return ss.str();
}

static unordered_set<string> load_completed(const string& path) {
    unordered_set<string> keys;
    ifstream f(path);
    if (!f) return keys;
    string header, line;
    getline(f, header);
    while (getline(f, line)) {
        auto cols = split(line, ',');
        if (cols.size() < 37) continue;
        int q = cols[9].empty() ? -1 : stoi(cols[9]);
        keys.insert(make_key(stoull(cols[3]), stoi(cols[21]), q, stoi(cols[8]), stod(cols[14]), cols[18]));
    }
    return keys;
}

static void write_header(ofstream& f) {
    f << "experiment_name,dataset,stream_type,T,num_raw_events,num_set_updates,max_active,avg_active,memory_kb,q,R,m,w,num_cells,epsilon,delta,rho,mechanism,method,seed_hash,seed_noise,trial,rmsd,relative_error,accepted_updates,filtered_updates,filtering_rate,saturated_cells,saturated_cell_ratio,load_p50,load_p95,load_p99,load_max,throughput_mops,eval_every,p_ins,N_max,column_q_schedule,sigma_node,clipped_evals,eval_samples,clipping_rate\n";
}

static void append_row(const string& path, const vector<string>& row) {
    bool header = !filesystem::exists(path) || filesystem::file_size(path) == 0;
    const filesystem::path parent = filesystem::path(path).parent_path();
    if (!parent.empty()) filesystem::create_directories(parent);
    ofstream f(path, ios::app);
    if (header) write_header(f);
    for (size_t i = 0; i < row.size(); ++i) {
        if (i) f << ',';
        f << row[i];
    }
    f << '\n';
}

struct BatchState {
    string method_key;
    string label;
    int mem = 0;
    double eps = 0.0;
    uint64_t seed_hash = 0;
    uint64_t seed_noise = 0;
    unique_ptr<Method> method;
    double err2 = 0.0;
    uint64_t samples = 0;
};

class SharedSplitSketch {
public:
    string method_key, method_name;
    uint64_t T = 0, n_max = 0, memory_bits = 0, sketch_bits = 0, seed_hash = 0, seed_noise = 0;
    uint64_t num_cells = 0, num_inserts = 0, num_deletes = 0, current_t = 0;
    vector<double> epsilons, sigmas, err2, last_estimates;
    vector<uint64_t> samples;
    SharedGaussianBinaryTree tree_add, tree_del;

    SharedSplitSketch(string key, string name, uint64_t T_, uint64_t nmax, int mem, const vector<double>& eps, double delta, double sensitivity, uint64_t sh, uint64_t sn)
        : method_key(std::move(key)), method_name(std::move(name)), T(T_), n_max(nmax), memory_bits((uint64_t)mem * 1024ull * 8ull),
          seed_hash(sh), seed_noise(sn), epsilons(eps), tree_add(T_, sn), tree_del(T_, sn + 1000003ull) {
        int H = tree_height(T);
        uint64_t tree_bits = 6ull * (H + 1ull) * 64ull;
        sketch_bits = memory_bits > tree_bits ? memory_bits - tree_bits : 0;
        for (double epsilon : epsilons) {
            sigmas.push_back(sensitivity * sqrt(H / epsilon_delta_to_rho(epsilon, delta)));
        }
        err2.assign(epsilons.size(), 0.0);
        last_estimates.assign(epsilons.size(), 0.0);
        samples.assign(epsilons.size(), 0);
    }

    virtual ~SharedSplitSketch() = default;
    virtual double insert_into_add(uint64_t item) = 0;
    virtual double insert_into_del(uint64_t item) = 0;
    virtual double estimate_one(double z, uint64_t seen) const = 0;

    void update(uint64_t t, uint64_t item, int op) {
        current_t = t;
        double da = 0.0, dd = 0.0;
        if (op > 0) {
            num_inserts++;
            da = insert_into_add(item);
        } else {
            num_deletes++;
            dd = insert_into_del(item);
        }
        tree_add.update(t, da);
        tree_del.update(t, dd);
    }

    double estimate_index(size_t index) {
        double add = estimate_one(tree_add.value(sigmas[index]), num_inserts);
        double del = estimate_one(tree_del.value(sigmas[index]), num_deletes);
        last_estimates[index] = max(0.0, min<double>({(double)n_max, (double)current_t, add - del}));
        return last_estimates[index];
    }

    void record(uint64_t true_n) {
        for (size_t i = 0; i < epsilons.size(); ++i) {
            double est = estimate_index(i);
            double rel = (est - true_n) / (double)true_n;
            err2[i] += rel * rel;
            samples[i]++;
        }
    }

    Result result(size_t index, uint64_t final_true, double elapsed) {
        double rmsd = samples[index] ? sqrt(err2[index] / samples[index]) : 0.0;
        estimate_index(index);
        Result r;
        r.method = method_name;
        r.rmsd = rmsd;
        r.relative_error = final_true ? (last_estimates[index] - final_true) / (double)final_true : 0.0;
        r.accepted = T;
        r.num_cells = num_cells;
        r.throughput_mops = elapsed > 0 ? T / elapsed / 1000000.0 : 0.0;
        return r;
    }
};

class SharedFMSketch : public SharedSplitSketch {
public:
    uint64_t m;
    vector<uint32_t> add_bitmap, del_bitmap;

    SharedFMSketch(uint64_t T, uint64_t n, int mem, const vector<double>& eps, double delta, uint64_t sh, uint64_t sn)
        : SharedSplitSketch("fm", "FM-Turnstile", T, n, mem, eps, delta, 64.0, sh, sn) {
        m = max<uint64_t>(1, sketch_bits / 64ull);
        add_bitmap.assign(m, 0);
        del_bitmap.assign(m, 0);
        num_cells = 2 * m * 32;
    }

    double insert_into(vector<uint32_t>& bitmap, uint64_t item) {
        auto [h1, h2] = hash_pair(item, seed_hash);
        uint64_t row = h1 % m;
        int col = min(ctz64(h2), 31);
        uint32_t old = bitmap[row];
        int old_prefix = trailing_ones32(old);
        bitmap[row] = old | (1u << col);
        return trailing_ones32(bitmap[row]) - old_prefix;
    }

    double insert_into_add(uint64_t item) override { return insert_into(add_bitmap, item); }
    double insert_into_del(uint64_t item) override { return insert_into(del_bitmap, item); }

    double estimate_one(double z, uint64_t seen) const override {
        if (seen == 0 || z <= 0.0) return 0.0;
        return max(0.0, min<double>(max(T, n_max), m * pow(2.0, z / m) / 0.77351));
    }
};

class SharedHLLSketch : public SharedSplitSketch {
public:
    uint64_t m;
    double alpha;
    vector<uint8_t> add_regs, del_regs;

    SharedHLLSketch(uint64_t T, uint64_t n, int mem, const vector<double>& eps, double delta, uint64_t sh, uint64_t sn)
        : SharedSplitSketch("hll", "HLL-Turnstile", T, n, mem, eps, delta, 1.0, sh, sn) {
        m = max<uint64_t>(1, sketch_bits / 16ull);
        add_regs.assign(m, 0);
        del_regs.assign(m, 0);
        alpha = (m == 16) ? 0.673 : (m == 32) ? 0.697 : (m == 64) ? 0.709 : 0.7213 / (1.0 + 1.079 / m);
        num_cells = 2 * m;
    }

    double insert_into(vector<uint8_t>& regs, uint64_t item) {
        auto [h1, h2] = hash_pair(item, seed_hash);
        uint64_t row = h1 % m;
        uint8_t rank = (uint8_t)min(ctz64(h2) + 1, 64);
        if (rank > regs[row]) {
            double d = pow(2.0, -regs[row]) - pow(2.0, -rank);
            regs[row] = rank;
            return d;
        }
        return 0.0;
    }

    double insert_into_add(uint64_t item) override { return insert_into(add_regs, item); }
    double insert_into_del(uint64_t item) override { return insert_into(del_regs, item); }

    double estimate_one(double z, uint64_t seen) const override {
        if (seen == 0) return 0.0;
        double inv = m - z;
        if (inv <= 0.0) return (double)max(T, n_max);
        return max(0.0, min<double>(max(T, n_max), alpha * m * m / inv));
    }
};

class SharedCheckpointSketch {
public:
    uint64_t T, n_max, memory_kb, num_cells, m, seed_hash, seed_noise, reports;
    int w;
    vector<uint32_t> B;
    vector<double> epsilons, sigmas, err2, last_estimates;
    vector<uint64_t> samples;
    HfxEstimator estimator;
    normal_distribution<double> normal;
    mt19937_64 rng;
    double exact_z = 0.0;

    SharedCheckpointSketch(uint64_t T_, uint64_t nmax, int mem, const vector<double>& eps, double delta,
                           int width, uint64_t eval_every, uint64_t sh, uint64_t sn)
        : T(T_), n_max(nmax), memory_kb(mem), seed_hash(sh), seed_noise(sn),
          reports((T_ + max<uint64_t>(1, eval_every) - 1) / max<uint64_t>(1, eval_every) + 1),
          w(width), epsilons(eps), estimator(1, width, nmax), normal(0.0, 1.0), rng(sn) {
        uint64_t memory_bits = (uint64_t)mem * 1024ull * 8ull;
        m = max<uint64_t>(1, memory_bits / (uint64_t)w);
        num_cells = m * (uint64_t)w;
        B.assign(m, 0);
        estimator = HfxEstimator(m, w, nmax);
        for (double epsilon : epsilons) {
            sigmas.push_back(sqrt((double)reports / (2.0 * epsilon_delta_to_rho(epsilon, delta))));
        }
        err2.assign(epsilons.size(), 0.0);
        last_estimates.assign(epsilons.size(), 0.0);
        samples.assign(epsilons.size(), 0);
    }

    void update(uint64_t item) {
        auto [h1, h2] = hash_pair(item, seed_hash);
        uint64_t row = h1 % m;
        int col = min(ctz64(h2), w - 1);
        uint32_t bit = 1u << col;
        bool old_one = (B[row] & bit) != 0;
        B[row] ^= bit;
        exact_z += old_one ? -1.0 : 1.0;
    }

    void record(uint64_t true_n) {
        double g = normal(rng);
        for (size_t i = 0; i < epsilons.size(); ++i) {
            double estimate = estimator.invert(exact_z + sigmas[i] * g);
            last_estimates[i] = estimate;
            double rel = (estimate - true_n) / (double)true_n;
            err2[i] += rel * rel;
            samples[i]++;
        }
    }

    void finalize() {
        double g = normal(rng);
        for (size_t i = 0; i < epsilons.size(); ++i) {
            last_estimates[i] = estimator.invert(exact_z + sigmas[i] * g);
        }
    }

    Result result(size_t index, uint64_t final_true, double elapsed) const {
        Result r;
        r.method = "HFX-Checkpoint";
        r.rmsd = samples[index] ? sqrt(err2[index] / samples[index]) : 0.0;
        r.relative_error = final_true ? (last_estimates[index] - final_true) / (double)final_true : 0.0;
        r.accepted = T;
        r.num_cells = num_cells;
        r.throughput_mops = elapsed > 0 ? T / elapsed / 1000000.0 : 0.0;
        return r;
    }
};

static unique_ptr<Method> make_method(const string& method, uint64_t T, uint64_t n_max, int mem, double eps, double delta, int q, int w, double column_cap_scale, uint64_t eval_every, uint64_t seed_hash, uint64_t seed_noise, const string& noise_mode) {
    if (method == "hfx") return make_unique<HFXGaussian>(T, n_max, mem, eps, delta, q, w, seed_hash, seed_noise, noise_mode);
    if (method == "hfx_checkpoint") return make_unique<HFXCheckpointGaussian>(T, n_max, mem, eps, delta, w, eval_every, seed_hash, seed_noise);
    if (method == "hfx_checkpoint_ideal") return make_unique<HFXCheckpointGaussian>(T, n_max, mem, eps, delta, w, eval_every, seed_hash, seed_noise, false, "Checkpoint Ideal");
    if (method == "hfx_column") return make_unique<HFXColumnGaussian>(T, n_max, mem, eps, delta, q, w, column_cap_scale, seed_hash, seed_noise);
    if (method == "hfx_column_unfiltered") return make_unique<HFXColumnGaussian>(T, n_max, mem, eps, delta, q, w, column_cap_scale, seed_hash, seed_noise, false, true, "HFX-Column-Unfiltered");
    if (method == "hfx_column_ideal") return make_unique<HFXColumnGaussian>(T, n_max, mem, eps, delta, q, w, column_cap_scale, seed_hash, seed_noise, false, false, "Column Ideal");
    if (method == "hfx_column_filtered") return make_unique<HFXColumnGaussian>(T, n_max, mem, eps, delta, q, w, column_cap_scale, seed_hash, seed_noise, true, false, "Column Filtered");
    if (method == "hfx_full") return make_unique<HFXGaussian>(T, n_max, mem, eps, delta, q, w, seed_hash, seed_noise, noise_mode, true, true, "Full HFX");
    if (method == "hfx_filtered") return make_unique<HFXGaussian>(T, n_max, mem, eps, delta, q, w, seed_hash, seed_noise, noise_mode, true, false, "Filtered XOR");
    if (method == "hfx_ideal") return make_unique<HFXGaussian>(T, n_max, mem, eps, delta, q, w, seed_hash, seed_noise, noise_mode, false, false, "Ideal XOR");
    if (method == "fm") return make_unique<FMTurnstile>(T, n_max, mem, eps, delta, seed_hash, seed_noise);
    if (method == "ll") return make_unique<LLTurnstile>(T, n_max, mem, eps, delta, seed_hash, seed_noise);
    if (method == "hll") return make_unique<HLLTurnstile>(T, n_max, mem, eps, delta, seed_hash, seed_noise);
    if (method == "flipdp") return make_unique<FlipDPProxy>(T, n_max, mem, eps, delta, seed_hash, seed_noise);
    throw runtime_error("unknown method");
}

static bool has_only_fixed_baseline_methods(const vector<string>& methods) {
    if (methods.empty()) return false;
    for (const string& method : methods) {
        if (method != "fm" && method != "hll") return false;
    }
    return true;
}

static bool has_only_checkpoint_method(const vector<string>& methods) {
    return methods.size() == 1 && methods[0] == "hfx_checkpoint";
}

static void append_result_row(const string& output, const string& dataset, uint64_t T, uint64_t n_max, double p_ins, double delta, int q, int w, uint64_t eval_every, int trial, int mem, double eps, const string& method_key, uint64_t seed_hash, uint64_t seed_noise, const Result& r, const StreamStats& s, const string& hfx_noise_mode, const string& experiment_name = "experiment1_accuracy_memory");

static uint64_t read_binary_stream_length(const string& path) {
    ifstream in(path, ios::binary);
    if (!in) throw runtime_error("cannot open binary stream: " + path);
    char magic[8];
    in.read(magic, 8);
    if (in.gcount() != 8 || string(magic, 8) != "HFXSTRM1") throw runtime_error("bad binary stream header: " + path);
    uint64_t T = 0;
    in.read(reinterpret_cast<char*>(&T), sizeof(T));
    if (!in) throw runtime_error("bad binary stream length: " + path);
    return T;
}

static void run_binary_stream(const Args& args) {
    auto methods = split(args.methods, ',');
    uint64_t T = read_binary_stream_length(args.input_binary);
    uint64_t n_max = args.n_max ? args.n_max : default_nmax(T);
    uint64_t eval_every = args.eval_every ? args.eval_every : (T >= 1000000000ull ? 10000ull : 1000ull);

    for (int trial = 0; trial < args.trials; ++trial) {
        uint64_t seed_hash = args.seed_hash + trial * 100ull;
        vector<BatchState> states;
        for (int mem : args.memory_kb) {
            int hfx_q = args.q > 0 ? args.q : choose_hfx_q(T, mem, args.w);
            for (double eps : args.epsilon) {
                for (const string& method : methods) {
                    uint64_t offset = method == "hfx" ? 101 : method == "hfx_column" ? 102 : method == "hfx_full" ? 103 : method == "hfx_filtered" ? 107 : method == "hfx_ideal" ? 109 : method == "fm" ? 211 : method == "ll" ? 307 : method == "hll" ? 401 : 503;
                    BatchState state;
                    state.method_key = method;
                    state.label = method_label(method);
                    state.mem = mem;
                    state.eps = eps;
                    state.seed_hash = seed_hash;
                    state.seed_noise = args.seed_noise + trial * 100ull + offset;
                    state.method = make_method(method, T, n_max, mem, eps, args.delta, hfx_q, args.w, args.column_cap_scale, eval_every, seed_hash, state.seed_noise, args.hfx_noise_mode);
                    states.push_back(std::move(state));
                }
            }
        }

        cout << "Binary stream running " << states.size() << " states: dataset=" << args.dataset
             << ", T=" << T << ", N_max=" << n_max << ", trial=" << (trial + 1) << "/" << args.trials << "\n" << flush;

        ifstream in(args.input_binary, ios::binary);
        char magic[8];
        uint64_t header_T = 0;
        in.read(magic, 8);
        in.read(reinterpret_cast<char*>(&header_T), sizeof(header_T));
        if (!in || header_T != T) throw runtime_error("failed to read binary stream header");

        int64_t true_active = 0;
        uint64_t max_active = 0, final_true = 0;
        long double active_sum = 0.0L;
        auto start = chrono::steady_clock::now();
        for (uint64_t t = 1; t <= T; ++t) {
            uint64_t item = 0;
            int8_t op8 = 0;
            in.read(reinterpret_cast<char*>(&item), sizeof(item));
            in.read(reinterpret_cast<char*>(&op8), sizeof(op8));
            if (!in) throw runtime_error("unexpected EOF in binary stream");
            int op = (int)op8;
            if (op != 1 && op != -1) throw runtime_error("invalid op in binary stream");

            true_active += op;
            if (true_active < 0) throw runtime_error("true active count went negative");
            uint64_t true_n = (uint64_t)true_active;
            active_sum += (long double)true_n;
            max_active = max(max_active, true_n);
            final_true = true_n;

            for (auto& state : states) state.method->update(t, item, op);
            if (t >= args.ignore && t % eval_every == 0 && true_n > 0) {
                for (auto& state : states) {
                    double est = state.method->estimate();
                    double rel = (est - true_n) / (double)true_n;
                    state.err2 += rel * rel;
                    state.samples++;
                }
            }
        }
        double elapsed = chrono::duration<double>(chrono::steady_clock::now() - start).count();
        StreamStats stats{max_active, T ? (double)(active_sum / (long double)T) : 0.0};
        for (auto& state : states) {
            double rmsd = state.samples ? sqrt(state.err2 / state.samples) : 0.0;
            state.method->estimate();
            Result r = state.method->finish(rmsd, final_true, elapsed);
            r.clipped_evals = state.method->clipped_evals;
            r.eval_samples = state.method->eval_samples;
            int hfx_q = args.q > 0 ? args.q : choose_hfx_q(T, state.mem, args.w);
            append_result_row(args.output, args.dataset, T, n_max, args.p_ins, args.delta, hfx_q, args.w, eval_every, trial, state.mem, state.eps, state.method_key, state.seed_hash, state.seed_noise, r, stats, args.hfx_noise_mode);
        }
    }
}

static void append_result_row(const string& output, const string& dataset, uint64_t T, uint64_t n_max, double p_ins, double delta, int q, int w, uint64_t eval_every, int trial, int mem, double eps, const string& method_key, uint64_t seed_hash, uint64_t seed_noise, const Result& r, const StreamStats& s, const string& hfx_noise_mode, const string& experiment_name) {
    double rho = epsilon_delta_to_rho(eps, delta);
    double sigma_node = method_sigma_node(method_key, T, q, eps, delta, hfx_noise_mode);
    bool hfx_like = is_hfx_method(method_key);
    bool has_q = hfx_like && method_key != "hfx_checkpoint" && method_key != "hfx_checkpoint_ideal";
    vector<string> row = {
        experiment_name, dataset, "active_set",
        to_string(T), to_string(T), to_string(T), to_string(s.max_active), csv_num(s.avg_active),
        to_string(mem), has_q ? to_string(q) : "", has_q ? to_string((1ull << q) - 1ull) : "",
        hfx_like ? to_string(r.num_cells / w) : to_string(r.num_cells), hfx_like ? to_string(w) : "",
        to_string(r.num_cells), csv_num(eps), csv_num(delta), csv_num(rho),
        mechanism_label(method_key, hfx_noise_mode), r.method,
        to_string(seed_hash), to_string(seed_noise), to_string(trial), csv_num(r.rmsd), csv_num(r.relative_error),
        to_string(r.accepted), to_string(r.filtered), csv_num(T ? r.filtered / (double)T : 0.0),
        to_string(r.saturated), csv_num(r.num_cells ? r.saturated / (double)r.num_cells : 0.0),
        csv_num(r.load_p50), csv_num(r.load_p95), csv_num(r.load_p99), csv_num(r.load_max),
        csv_num(r.throughput_mops), to_string(eval_every), csv_num(p_ins), to_string(n_max), r.column_q_schedule,
        csv_num(sigma_node), to_string(r.clipped_evals), to_string(r.eval_samples),
        csv_num(r.eval_samples ? r.clipped_evals / (double)r.eval_samples : 0.0)
    };
    append_row(output, row);
}

static void run_delete_heavy_stream(const Args& args) {
    auto methods = split(args.methods, ',');
    for (uint64_t N : args.T) {
        for (double deletion_ratio : args.deletion_ratio) {
            double clipped_ratio = min(max(deletion_ratio, 0.0), 0.999999);
            uint64_t deletes = (uint64_t)llround((double)N * clipped_ratio);
            if (deletes >= N) deletes = N - 1;
            uint64_t stream_T = N + deletes;
            uint64_t final_true = N - deletes;
            long double insert_sum = ((long double)N * (long double)(N + 1)) / 2.0L;
            long double delete_sum = (long double)deletes * (long double)N - ((long double)deletes * (long double)(deletes + 1)) / 2.0L;
            StreamStats stats{N, stream_T ? (double)((insert_sum + delete_sum) / (long double)stream_T) : 0.0};
            uint64_t eval_every = args.eval_every ? args.eval_every : stream_T;
            string dataset = "delete_heavy_N" + to_string(N);

            for (int trial = 0; trial < args.trials; ++trial) {
                uint64_t seed_hash = args.seed_hash + trial * 100ull;
                uint64_t perm_a = coprime_multiplier(N, args.seed_stream + trial * 10000ull + 79ull + deletes);
                uint64_t perm_b = splitmix64(args.seed_stream + trial * 10000ull + 131ull + deletes) % N;
                for (int mem : args.memory_kb) {
                    int hfx_q = args.q > 0 ? args.q : choose_hfx_q(stream_T, mem, args.w);
                    for (double eps : args.epsilon) {
                        for (const string& method : methods) {
                            uint64_t offset = method == "hfx" ? 101 : method == "hfx_column" ? 102 : method == "hfx_full" ? 103 : method == "hfx_filtered" ? 107 : method == "hfx_ideal" ? 109 : method == "fm" ? 211 : method == "ll" ? 307 : method == "hll" ? 401 : 503;
                            uint64_t seed_noise = args.seed_noise + trial * 100ull + offset;
                            cout << "Delete-heavy " << method_label(method)
                                 << ": N=" << N << ", deletion=" << clipped_ratio
                                 << ", mem=" << mem << "KB, eps=" << eps
                                 << ", q=" << hfx_q << ", trial=" << (trial + 1) << "/" << args.trials << "\n" << flush;

                            auto estimator = make_method(method, stream_T, N, mem, eps, args.delta, hfx_q, args.w, args.column_cap_scale, eval_every, seed_hash, seed_noise, args.hfx_noise_mode);
                            auto start = chrono::steady_clock::now();
                            uint64_t t = 0;
                            for (uint64_t item = 1; item <= N; ++item) {
                                estimator->update(++t, item, 1);
                            }
                            uint64_t samples = 0;
                            double err2 = 0.0;
                            for (uint64_t j = 0; j < deletes; ++j) {
                                uint64_t item = ((perm_a * j + perm_b) % N) + 1;
                                estimator->update(++t, item, -1);
                                uint64_t true_n = N - j - 1;
                                if (((j + 1) % eval_every == 0 || j + 1 == deletes) && true_n > 0) {
                                    double est = estimator->estimate();
                                    double rel = (est - true_n) / (double)true_n;
                                    err2 += rel * rel;
                                    samples++;
                                }
                            }
                            double elapsed = chrono::duration<double>(chrono::steady_clock::now() - start).count();
                            estimator->estimate();
                            double rmsd = samples ? sqrt(err2 / samples) : 0.0;
                            Result r = estimator->finish(rmsd, final_true, elapsed);
                            r.clipped_evals = estimator->clipped_evals;
                            r.eval_samples = estimator->eval_samples;
                            append_result_row(args.output, dataset, stream_T, N, clipped_ratio, args.delta, hfx_q, args.w, eval_every, trial,
                                              mem, eps, method, seed_hash, seed_noise, r, stats, args.hfx_noise_mode,
                                              "experiment7_delete_heavy");
                        }
                    }
                }
            }
        }
    }
}

static void run_batch_stream(const Args& args) {
    auto methods = split(args.methods, ',');
    if (has_only_checkpoint_method(methods)) {
        for (auto T : args.T) {
            uint64_t n_max = args.n_max ? args.n_max : default_nmax(T);
            uint64_t eval_every = args.eval_every ? args.eval_every : (T >= 1000000000ull ? 10000ull : 1000ull);
            for (int trial = 0; trial < args.trials; ++trial) {
                uint64_t seed_stream = args.seed_stream + trial * 10000ull;
                uint64_t seed_hash = args.seed_hash + trial * 100ull;
                uint64_t seed_noise = args.seed_noise + trial * 100ull + 155ull;
                vector<unique_ptr<SharedCheckpointSketch>> sketches;
                for (int mem : args.memory_kb) {
                    sketches.push_back(make_unique<SharedCheckpointSketch>(T, n_max, mem, args.epsilon, args.delta,
                                                                           args.w, eval_every, seed_hash, seed_noise));
                }

                cout << "Fast checkpoint batch running " << sketches.size() << " sketches x " << args.epsilon.size()
                     << " eps: T=" << T << ", N_max=" << n_max << ", trial=" << (trial + 1) << "/" << args.trials << "\n" << flush;
                mt19937_64 rng(seed_stream);
                uniform_real_distribution<double> uni(0.0, 1.0);
                vector<uint64_t> active;
                active.reserve((size_t)min<uint64_t>(n_max, 10000000ull));
                uint64_t next_id = 1, active_sum = 0, max_active = 0, final_true = 0;
                auto start = chrono::steady_clock::now();
                for (uint64_t t = 1; t <= T; ++t) {
                    bool do_insert = active.empty() || (active.size() < n_max && uni(rng) < args.p_ins);
                    uint64_t item;
                    if (do_insert) {
                        item = next_id++;
                        active.push_back(item);
                    } else {
                        uniform_int_distribution<uint64_t> pick(0, active.size() - 1);
                        uint64_t idx = pick(rng);
                        item = active[idx];
                        active[idx] = active.back();
                        active.pop_back();
                    }
                    uint64_t true_n = active.size();
                    active_sum += true_n;
                    max_active = max(max_active, true_n);
                    final_true = true_n;
                    for (auto& sketch : sketches) sketch->update(item);
                    if (t >= args.ignore && t % eval_every == 0 && true_n > 0) {
                        for (auto& sketch : sketches) sketch->record(true_n);
                    }
                }
                double elapsed = chrono::duration<double>(chrono::steady_clock::now() - start).count();
                StreamStats stats{max_active, T ? active_sum / (double)T : 0.0};
                for (auto& sketch : sketches) {
                    sketch->finalize();
                    for (size_t i = 0; i < sketch->epsilons.size(); ++i) {
                        Result r = sketch->result(i, final_true, elapsed);
                        append_result_row(args.output, args.dataset, T, n_max, args.p_ins, args.delta, -1, args.w,
                                          eval_every, trial, (int)sketch->memory_kb, sketch->epsilons[i],
                                          "hfx_checkpoint", sketch->seed_hash, sketch->seed_noise, r, stats, args.hfx_noise_mode);
                    }
                }
            }
        }
        return;
    }
    if (has_only_fixed_baseline_methods(methods)) {
        for (auto T : args.T) {
            uint64_t n_max = args.n_max ? args.n_max : default_nmax(T);
            uint64_t eval_every = args.eval_every ? args.eval_every : (T >= 1000000000ull ? 10000ull : 1000ull);
            for (int trial = 0; trial < args.trials; ++trial) {
                uint64_t seed_stream = args.seed_stream + trial * 10000ull;
                uint64_t seed_hash = args.seed_hash + trial * 100ull;
                vector<unique_ptr<SharedSplitSketch>> sketches;
                for (int mem : args.memory_kb) {
                    for (const string& method : methods) {
                        uint64_t offset = method == "fm" ? 211 : 401;
                        uint64_t seed_noise = args.seed_noise + trial * 100ull + offset;
                        if (method == "fm") {
                            sketches.push_back(make_unique<SharedFMSketch>(T, n_max, mem, args.epsilon, args.delta, seed_hash, seed_noise));
                        } else {
                            sketches.push_back(make_unique<SharedHLLSketch>(T, n_max, mem, args.epsilon, args.delta, seed_hash, seed_noise));
                        }
                    }
                }

                cout << "Fast batch running " << sketches.size() << " sketches x " << args.epsilon.size()
                     << " eps: T=" << T << ", N_max=" << n_max << ", trial=" << (trial + 1) << "/" << args.trials << "\n" << flush;
                mt19937_64 rng(seed_stream);
                uniform_real_distribution<double> uni(0.0, 1.0);
                vector<uint64_t> active;
                active.reserve((size_t)min<uint64_t>(n_max, 10000000ull));
                uint64_t next_id = 1, active_sum = 0, max_active = 0, final_true = 0;
                auto start = chrono::steady_clock::now();
                for (uint64_t t = 1; t <= T; ++t) {
                    bool do_insert = active.empty() || (active.size() < n_max && uni(rng) < args.p_ins);
                    uint64_t item;
                    int op;
                    if (do_insert) {
                        item = next_id++;
                        active.push_back(item);
                        op = 1;
                    } else {
                        uniform_int_distribution<uint64_t> pick(0, active.size() - 1);
                        uint64_t idx = pick(rng);
                        item = active[idx];
                        uint64_t last = active.back();
                        active[idx] = last;
                        active.pop_back();
                        op = -1;
                    }
                    uint64_t true_n = active.size();
                    active_sum += true_n;
                    max_active = max(max_active, true_n);
                    final_true = true_n;
                    for (auto& sketch : sketches) sketch->update(t, item, op);
                    if (t >= args.ignore && t % eval_every == 0 && true_n > 0) {
                        for (auto& sketch : sketches) sketch->record(true_n);
                    }
                }
                double elapsed = chrono::duration<double>(chrono::steady_clock::now() - start).count();
                StreamStats stats{max_active, T ? active_sum / (double)T : 0.0};
                for (auto& sketch : sketches) {
                    for (size_t i = 0; i < sketch->epsilons.size(); ++i) {
                        Result r = sketch->result(i, final_true, elapsed);
                        append_result_row(args.output, args.dataset, T, n_max, args.p_ins, args.delta, -1, args.w, eval_every, trial,
                                          (int)(sketch->memory_bits / 1024ull / 8ull), sketch->epsilons[i], sketch->method_key,
                                          sketch->seed_hash, sketch->seed_noise, r, stats, args.hfx_noise_mode);
                    }
                }
            }
        }
        return;
    }
    for (auto T : args.T) {
        uint64_t n_max = args.n_max ? args.n_max : default_nmax(T);
        uint64_t eval_every = args.eval_every ? args.eval_every : (T >= 1000000000ull ? 10000ull : 1000ull);
        for (int trial = 0; trial < args.trials; ++trial) {
            uint64_t seed_stream = args.seed_stream + trial * 10000ull;
            uint64_t seed_hash = args.seed_hash + trial * 100ull;

            vector<BatchState> states;
            for (int mem : args.memory_kb) {
                int hfx_q = args.q > 0 ? args.q : choose_hfx_q(T, mem, args.w);
                for (double eps : args.epsilon) {
                    for (const string& method : methods) {
                        uint64_t offset = method == "hfx" ? 101 : method == "hfx_column" ? 102 : method == "hfx_full" ? 103 : method == "hfx_filtered" ? 107 : method == "hfx_ideal" ? 109 : method == "fm" ? 211 : method == "ll" ? 307 : method == "hll" ? 401 : 503;
                        BatchState state;
                        state.method_key = method;
                        state.label = method_label(method);
                        state.mem = mem;
                        state.eps = eps;
                        state.seed_hash = seed_hash;
                        state.seed_noise = args.seed_noise + trial * 100ull + offset;
                        state.method = make_method(method, T, n_max, mem, eps, args.delta, hfx_q, args.w, args.column_cap_scale, eval_every, seed_hash, state.seed_noise, args.hfx_noise_mode);
                        states.push_back(std::move(state));
                    }
                }
            }

            cout << "Batch running " << states.size() << " states: T=" << T << ", N_max=" << n_max
                 << ", trial=" << (trial + 1) << "/" << args.trials << "\n" << flush;
            mt19937_64 rng(seed_stream);
            uniform_real_distribution<double> uni(0.0, 1.0);
            vector<uint64_t> active;
            active.reserve((size_t)min<uint64_t>(n_max, 10000000ull));
            uint64_t next_id = 1, active_sum = 0, max_active = 0, final_true = 0;
            auto start = chrono::steady_clock::now();
            for (uint64_t t = 1; t <= T; ++t) {
                bool do_insert = active.empty() || (active.size() < n_max && uni(rng) < args.p_ins);
                uint64_t item;
                int op;
                if (do_insert) {
                    item = next_id++;
                    active.push_back(item);
                    op = 1;
                } else {
                    uniform_int_distribution<uint64_t> pick(0, active.size() - 1);
                    uint64_t idx = pick(rng);
                    item = active[idx];
                    uint64_t last = active.back();
                    active[idx] = last;
                    active.pop_back();
                    op = -1;
                }
                uint64_t true_n = active.size();
                active_sum += true_n;
                max_active = max(max_active, true_n);
                final_true = true_n;
                for (auto& state : states) state.method->update(t, item, op);
                if (t >= args.ignore && t % eval_every == 0 && true_n > 0) {
                    for (auto& state : states) {
                        double est = state.method->estimate();
                        double rel = (est - true_n) / (double)true_n;
                        state.err2 += rel * rel;
                        state.samples++;
                    }
                }
            }
            double elapsed = chrono::duration<double>(chrono::steady_clock::now() - start).count();
            StreamStats stats{max_active, T ? active_sum / (double)T : 0.0};
            for (auto& state : states) {
                double rmsd = state.samples ? sqrt(state.err2 / state.samples) : 0.0;
                state.method->estimate();
                Result r = state.method->finish(rmsd, final_true, elapsed);
                r.clipped_evals = state.method->clipped_evals;
                r.eval_samples = state.method->eval_samples;
                int hfx_q = args.q > 0 ? args.q : choose_hfx_q(T, state.mem, args.w);
                append_result_row(args.output, args.dataset, T, n_max, args.p_ins, args.delta, hfx_q, args.w, eval_every, trial, state.mem, state.eps, state.method_key, state.seed_hash, state.seed_noise, r, stats, args.hfx_noise_mode);
            }
        }
    }
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        string k = argv[i];
        auto val = [&]() -> string { if (i + 1 >= argc) throw runtime_error("missing value for " + k); return argv[++i]; };
        if (k == "--preset") {
            string p = val();
            if (p == "full") {
                a.T = {10000000ull, 100000000ull, 1000000000ull};
                a.memory_kb = {8, 16, 32, 64};
                a.epsilon = {0.5, 1.0, 2.0, 3.0, 4.0};
                a.trials = 10;
                a.methods = "hfx,fm,ll,hll,flipdp";
                a.q = 0;
                a.hfx_noise_mode = "load_aware";
            } else if (p == "smoke") {
                a.T = {20000};
            }
        } else if (k == "--T") a.T = parse_list<uint64_t>(val());
        else if (k == "--memory-kb") a.memory_kb = parse_list<int>(val());
        else if (k == "--epsilon") a.epsilon = parse_list<double>(val());
        else if (k == "--trials") a.trials = stoi(val());
        else if (k == "--methods") a.methods = val();
        else if (k == "--output") a.output = val();
        else if (k == "--input-binary") a.input_binary = val();
        else if (k == "--dataset") a.dataset = val();
        else if (k == "--n-max") a.n_max = stoull(val());
        else if (k == "--p-ins") a.p_ins = stod(val());
        else if (k == "--delta") a.delta = stod(val());
        else if (k == "--q") a.q = stoi(val());
        else if (k == "--w") a.w = stoi(val());
        else if (k == "--column-cap-scale") a.column_cap_scale = stod(val());
        else if (k == "--eval-every") a.eval_every = stoull(val());
        else if (k == "--ignore") a.ignore = stoull(val());
        else if (k == "--seed") a.seed_stream = stoull(val());
        else if (k == "--seed-stream") a.seed_stream = stoull(val());
        else if (k == "--seed-hash") a.seed_hash = stoull(val());
        else if (k == "--seed-noise") a.seed_noise = stoull(val());
        else if (k == "--hfx-noise-mode") a.hfx_noise_mode = val();
        else if (k == "--resume") a.resume = true;
        else if (k == "--batch-stream") a.batch_stream = true;
        else if (k == "--delete-heavy") a.delete_heavy = true;
        else if (k == "--deletion-ratio") a.deletion_ratio = parse_list<double>(val());
    }
    return a;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    Args args = parse_args(argc, argv);
    if (args.q > 30) throw invalid_argument("--q must be at most 30");
    if (args.hfx_noise_mode != "load_aware" && args.hfx_noise_mode != "theorem" &&
        args.hfx_noise_mode != "unit_diagnostic" && args.hfx_noise_mode != "unit") {
        throw invalid_argument("--hfx-noise-mode must be load_aware, theorem, or unit_diagnostic");
    }
    if (!args.input_binary.empty()) {
        run_binary_stream(args);
        cout << "Finished; results are in " << filesystem::absolute(args.output).string() << "\n";
        return 0;
    }
    if (args.delete_heavy) {
        run_delete_heavy_stream(args);
        cout << "Finished; results are in " << filesystem::absolute(args.output).string() << "\n";
        return 0;
    }
    if (args.batch_stream) {
        run_batch_stream(args);
        cout << "Finished; results are in " << filesystem::absolute(args.output).string() << "\n";
        return 0;
    }
    auto completed = args.resume ? load_completed(args.output) : unordered_set<string>{};
    auto methods = split(args.methods, ',');
    unordered_set<string> done_now = completed;
    for (auto T : args.T) {
        uint64_t n_max = args.n_max ? args.n_max : default_nmax(T);
        uint64_t eval_every = args.eval_every ? args.eval_every : (T >= 1000000000ull ? 10000ull : 1000ull);
        for (int trial = 0; trial < args.trials; ++trial) {
            uint64_t seed_stream = args.seed_stream + trial * 10000ull;
            for (int mem : args.memory_kb) {
                int hfx_q = args.q > 0 ? args.q : choose_hfx_q(T, mem, args.w);
                for (double eps : args.epsilon) {
                    for (const string& method : methods) {
                        string label = method_label(method);
                        int q_key = is_hfx_method(method) ? hfx_q : -1;
                        string key = make_key(T, trial, q_key, mem, eps, label);
                        if (done_now.count(key)) {
                            cout << "Skipping completed " << label << ": T=" << T << ", mem=" << mem << "KB, eps=" << eps << ", q=" << hfx_q << ", trial=" << (trial + 1) << "/" << args.trials << "\n";
                            continue;
                        }
                        uint64_t seed_hash = args.seed_hash + trial * 100ull;
                        uint64_t offset = method == "hfx" ? 101 : method == "hfx_column" ? 102 : method == "hfx_full" ? 103 : method == "hfx_filtered" ? 107 : method == "hfx_ideal" ? 109 : method == "fm" ? 211 : method == "ll" ? 307 : method == "hll" ? 401 : 503;
                        uint64_t seed_noise = args.seed_noise + trial * 100ull + offset;
                        int audit_H = tree_height(T);
                        double audit_sigma = is_hfx_method(method) ? hfx_sigma(T, hfx_q, eps, args.delta, args.hfx_noise_mode, true) : 0.0;
                        uint64_t audit_tree_bits = 2ull * (audit_H + 1ull) * 64ull;
                        uint64_t audit_memory_bits = (uint64_t)mem * 1024ull * 8ull;
                        uint64_t audit_sketch_bits = audit_memory_bits > audit_tree_bits ? audit_memory_bits - audit_tree_bits : 0;
                        uint64_t audit_cells = max<uint64_t>(args.w, (audit_sketch_bits / (1ull + hfx_q) / args.w) * args.w);
                        uint64_t audit_m = max<uint64_t>(1, audit_cells / args.w);
                        double audit_phi_nmax = HfxEstimator(audit_m, args.w, n_max).max_phi;
                        cout << "Running " << label << ": T=" << T << ", H=" << audit_H << ", N_max=" << n_max
                             << ", mem=" << mem << " KB, eps=" << eps << ", delta=" << args.delta
                             << ", q=" << hfx_q << ", R=" << ((1ull << hfx_q) - 1ull)
                             << ", rho=" << epsilon_delta_to_rho(eps, args.delta)
                             << ", sigma_node=" << audit_sigma
                             << ", s_t_max=" << audit_H
                             << ", prefix_noise_sd_max=" << sqrt((double)audit_H) * audit_sigma
                             << ", m=" << audit_m << ", w=" << args.w
                             << ", Phi_N_max=" << audit_phi_nmax
                             << ", tree_bits=" << audit_tree_bits
                             << ", hfx_sketch_bits=" << audit_sketch_bits
                             << ", noise_mode=" << args.hfx_noise_mode
                             << ", eval_every=" << eval_every
                             << ", seed_stream=" << seed_stream
                             << ", seed_hash=" << seed_hash
                             << ", seed_noise=" << seed_noise
                             << ", trial=" << (trial + 1) << "/" << args.trials << "\n" << flush;
                        auto [r, s] = run_one(method, T, n_max, args.p_ins, mem, eps, args.delta, hfx_q, args.w, args.column_cap_scale, seed_stream, seed_hash, seed_noise, eval_every, args.ignore, args.hfx_noise_mode);
                        double rho = epsilon_delta_to_rho(eps, args.delta);
                        double sigma_node = method_sigma_node(method, T, hfx_q, eps, args.delta, args.hfx_noise_mode);
                        vector<string> row = {
                            "experiment1_accuracy_memory", "synthetic_active_set", "active_set",
                            to_string(T), to_string(T), to_string(T), to_string(s.max_active), csv_num(s.avg_active),
                            to_string(mem), (is_hfx_method(method) && method != "hfx_checkpoint" && method != "hfx_checkpoint_ideal") ? to_string(hfx_q) : "", (is_hfx_method(method) && method != "hfx_checkpoint" && method != "hfx_checkpoint_ideal") ? to_string((1ull << hfx_q) - 1ull) : "",
                            is_hfx_method(method) ? to_string(r.num_cells / args.w) : to_string(r.num_cells), is_hfx_method(method) ? to_string(args.w) : "",
                            to_string(r.num_cells), csv_num(eps), csv_num(args.delta), csv_num(rho),
                            mechanism_label(method, args.hfx_noise_mode), r.method,
                            to_string(seed_hash), to_string(seed_noise), to_string(trial), csv_num(r.rmsd), csv_num(r.relative_error),
                            to_string(r.accepted), to_string(r.filtered), csv_num(T ? r.filtered / (double)T : 0.0),
                            to_string(r.saturated), csv_num(r.num_cells ? r.saturated / (double)r.num_cells : 0.0),
                            csv_num(r.load_p50), csv_num(r.load_p95), csv_num(r.load_p99), csv_num(r.load_max),
                            csv_num(r.throughput_mops), to_string(eval_every), csv_num(args.p_ins), to_string(n_max), r.column_q_schedule,
                            csv_num(sigma_node), to_string(r.clipped_evals), to_string(r.eval_samples),
                            csv_num(r.eval_samples ? r.clipped_evals / (double)r.eval_samples : 0.0)
                        };
                        append_row(args.output, row);
                        done_now.insert(key);
                    }
                }
            }
        }
    }
    cout << "Finished; results are in " << filesystem::absolute(args.output).string() << "\n";
    return 0;
}
