#ifndef _HLL_DELETABLE_H
#define _HLL_DELETABLE_H

#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "MurmurHash3.h"
#include "parameters.h"

class HLL_Deletable {
public:
    HLL_Deletable(unsigned total_bits) {
        m = total_bits / 6;
        if (m == 0) m = 1;
        sketch_add.assign(m, 0);
        sketch_del.assign(m, 0);
        if (m == 16) { alpha_m = 0.673; }
        else if (m == 32) { alpha_m = 0.697; }
        else if (m == 64) { alpha_m = 0.709; }
        else { alpha_m = 0.7213 / (1.0 + 1.079 / m); }
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
    double alpha_m;
    std::vector<unsigned> sketch_add;
    std::vector<unsigned> sketch_del;
    const unsigned hash_seed = 123;

    void _update_sketch(std::vector<unsigned>& sketch, const std::string& item) {
        uint64_t hash_out[2];
        MurmurHash3_x64_128(item.c_str(), item.length(), hash_seed, &hash_out);
        unsigned m_index = hash_out[0] % m;
        unsigned w_index = std::countr_zero(hash_out[1]);
        sketch[m_index] = std::max(sketch[m_index], w_index + 1);
    }

    double _get_estimate(const std::vector<unsigned>& sketch) {
        double z = 0;
        unsigned zeros = 0;
        for (unsigned reg_val : sketch) {
            if (reg_val == 0) {
                zeros++;
            }
            z += std::pow(2.0, -static_cast<double>(reg_val));
        }

        double estimate = alpha_m * m * m / z;

        // FIXED: Re-introducing the small-range correction to ensure correct trend.
        // This uses LinearCounting when the HLL estimate is small.
        if (m > 0 && estimate <= 2.5 * m && zeros > 0) {
            estimate = m * std::log(static_cast<double>(m) / zeros);
        }
        
        return estimate;
    }
};

#endif // _HLL_DELETABLE_H