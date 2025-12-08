#ifndef _LL_DELETABLE_H
#define _LL_DELETABLE_H

#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "MurmurHash3.h"
#include "parameters.h"

class LL_Deletable {
public:
    LL_Deletable(unsigned total_bits) {
        m = total_bits / 6;
        if (m == 0) m = 1;
        sketch_add.assign(m, 0);
        sketch_del.assign(m, 0);
    }

    void insert(const std::string& item) {
        _update_sketch(sketch_add, item);
    }

    void remove(const std::string& item) {
        _update_sketch(sketch_del, item);
    }

    double estimate() {
        double est_add = _get_estimate(sketch_add);
        double est_del = _get_estimate(sketch_del);
        return std::max(0.0, est_add - est_del);
    }

private:
    unsigned m;
    const double alpha_m = 0.39701;
    std::vector<unsigned> sketch_add;
    std::vector<unsigned> sketch_del;
    const unsigned hash_seed = 456;

    void _update_sketch(std::vector<unsigned>& sketch, const std::string& item) {
        uint64_t hash_out[2];
        MurmurHash3_x64_128(item.c_str(), item.length(), hash_seed, &hash_out);
        unsigned m_index = hash_out[0] % m;
        unsigned w_index = std::countr_zero(hash_out[1]);
        sketch[m_index] = std::max(sketch[m_index], w_index + 1);
    }

    double _get_estimate(const std::vector<unsigned>& sketch) {
        // CORRECTED LOGIC: Use a trimmed mean to stabilize the estimate.
        std::vector<unsigned> sorted_sketch = sketch;
        std::sort(sorted_sketch.begin(), sorted_sketch.end());
        
        // Discard the 30% of registers with the smallest values.
        unsigned trim_idx = static_cast<unsigned>(m * 0.3);
        if (trim_idx >= m) return 0.0; // Should not happen with reasonable m

        double sum_of_ranks = 0;
        for (unsigned i = trim_idx; i < m; ++i) {
            sum_of_ranks += sorted_sketch[i];
        }

        unsigned count = m - trim_idx;
        double avg_rank = sum_of_ranks / static_cast<double>(count);
        
        return alpha_m * m * std::pow(2.0, avg_rank);
    }
};

#endif // _LL_DELETABLE_H