#ifndef _FM_DELETABLE_H
#define _FM_DELETABLE_H

#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <bit>
#include "MurmurHash3.h"
#include "parameters.h"

class FM_Deletable {
public:
    FM_Deletable(unsigned total_bits) {
        m = total_bits / 32;
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
    const double phi = 0.77351;
    std::vector<unsigned> sketch_add;
    std::vector<unsigned> sketch_del;
    const unsigned hash_seed = 789;

    void _update_sketch(std::vector<unsigned>& sketch, const std::string& item) {
        uint64_t hash_out[2];
        MurmurHash3_x64_128(item.c_str(), item.length(), hash_seed, &hash_out);
        unsigned m_index = hash_out[0] % m;
        unsigned w_index = std::min((unsigned)std::countr_zero(hash_out[1]), 31u);
        sketch[m_index] |= (1U << w_index);
    }

    double _get_estimate(const std::vector<unsigned>& sketch) {
        // CORRECTED LOGIC: Based on your paper's baseline, we sum the total number of set bits (popcount).
        double sum_of_set_bits = 0;
        for (unsigned bitmap : sketch) {
            sum_of_set_bits += std::popcount(bitmap);
        }

        if (m == 0) return 0.0;

        // Use the estimation formula corresponding to the popcount method.
        return m * std::pow(2.0, sum_of_set_bits / m) / phi;
    }
};

#endif // _FM_DELETABLE_H