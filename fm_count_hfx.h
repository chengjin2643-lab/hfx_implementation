// fm_count_hfx.h

#ifndef _FM_COUNT_HFX_H
#define _FM_COUNT_HFX_H

#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <bit>
#include "MurmurHash3.h"
#include "parameters.h"

class FM_Deletable_HFX {
public:
    FM_Deletable_HFX(unsigned total_bits) {
        // HFX/FMS sketch typically uses w=32
        w = 32;
        m = total_bits / w;
        if (m == 0) m = 1;
        sketch.assign(m, 0);
    }

    // Insert和Remove现在是同一个操作
    void update(const std::string& item) {
        uint64_t hash_out[2];
        // 使用不同的seed以避免和其它sketch冲突
        MurmurHash3_x64_128(item.c_str(), item.length(), 999, &hash_out);
        unsigned m_index = hash_out[0] % m;
        unsigned w_index = std::min((unsigned)std::countr_zero(hash_out[1]), w - 1);
        
        // 使用XOR操作来实现可逆的更新
        sketch[m_index] ^= (1U << w_index);
    }

    void insert(const std::string& item) {
        update(item);
    }

    void remove(const std::string& item) {
        update(item);
    }

    double estimate() {
        // 使用适用于XOR-based sketch的估算模型
        double set_bits = 0;
        for (unsigned val : sketch) {
            set_bits += std::popcount(val);
        }
        
        // 这个估算模型来自Half-Xor论文，用于动态数据流
        return hfx_cardinality_estimate_dynamic(set_bits);
    }

private:
    unsigned m;
    unsigned w;
    std::vector<unsigned> sketch;

    // --- 动态HFX估算模型 (来自你的hfx.h) ---

    // 计算在基数为n时，sketch中1的比例的期望值
    double hfx_calculate_Yn_dynamic(double n_val) {
        if (n_val < 0) n_val = 0;
        if (m == 0 || w == 0) return 0.0;
        
        double expected_set_bits_in_one_row = 0.0;
        const double lambda = 1.0 / (double)m;

        for (unsigned j_col = 0; j_col < w; ++j_col) {
            double p_j = (j_col < w - 1) ? pow(2.0, -(double)(j_col + 1)) : pow(2.0, -(double)(w - 1));
            // Half-Xor Theorem 2: P(B[i][j]=1) = 0.5 * (1 - exp(-n*lambda*p_j))
            expected_set_bits_in_one_row += 0.5 * (1.0 - exp(-n_val * lambda * p_j));
        }
        
        return expected_set_bits_in_one_row / (double)w;
    }

    // 通过二分查找，根据观察到的1的比例V_target，反解出基数n
    double hfx_binary_search_solve_n_dynamic(double V_target) {
        if (V_target < 0.0) V_target = 0.0;
        if (V_target > 0.5) V_target = 0.5;

        double low_n = 0.0;
        // 设置一个足够大的上界
        double high_n = (double)m * w * 4.0; 

        for(int iter = 0; iter < 100; ++iter) { 
            double mid_n = low_n + (high_n - low_n) / 2.0;
            if (mid_n <= 0) { low_n = 0; high_n = 1; continue; }
            double y_mid = hfx_calculate_Yn_dynamic(mid_n);
            if (y_mid < V_target) {
                low_n = mid_n;
            } else {
                high_n = mid_n;
            }
        }
        return low_n + (high_n - low_n) / 2.0;
    }

    double hfx_cardinality_estimate_dynamic(double total_set_bits) {
        if (m == 0 || w == 0) return 0.0; 
        double V = total_set_bits / (double)(m * w);
        return hfx_binary_search_solve_n_dynamic(V);
    }
};

#endif // _FM_COUNT_HFX_H