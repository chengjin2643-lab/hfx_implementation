// hll_count_hfx.h

#ifndef _HLL_COUNT_HFX_H
#define _HLL_COUNT_HFX_H

#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <map> // 使用map来存储稀疏的计数器
#include "MurmurHash3.h"
#include "parameters.h"

class HLL_Deletable_HFX {
public:
    HLL_Deletable_HFX(unsigned total_bits) {
        // 每个register需要6 bits，这是HLL的典型实现
        m = total_bits / 6;
        if (m == 0) m = 1;

        // 初始化HLL的寄存器
        registers.assign(m, 0);
        // 初始化计数器，每个桶一个map
        counts.resize(m);

        // 初始化HLL的alpha参数
        if (m == 16) { alpha_m = 0.673; }
        else if (m == 32) { alpha_m = 0.697; }
        else if (m == 64) { alpha_m = 0.709; }
        else { alpha_m = 0.7213 / (1.0 + 1.079 / m); }
    }

    void insert(const std::string& item) {
        uint64_t hash_out[2];
        MurmurHash3_x64_128(item.c_str(), item.length(), 123, &hash_out);
        unsigned m_index = hash_out[0] % m;
        unsigned w_index = std::countr_zero(hash_out[1]) + 1;

        // 对应值的计数器加一
        counts[m_index][w_index]++;
        
        // 更新当前桶的最大值
        if (w_index > registers[m_index]) {
            registers[m_index] = w_index;
        }
    }

    void remove(const std::string& item) {
        uint64_t hash_out[2];
        MurmurHash3_x64_128(item.c_str(), item.length(), 123, &hash_out);
        unsigned m_index = hash_out[0] % m;
        unsigned w_index = std::countr_zero(hash_out[1]) + 1;

        // 对应值的计数器减一
        if (counts[m_index].count(w_index) && counts[m_index][w_index] > 0) {
            counts[m_index][w_index]--;

            // 如果被删除的值是当前桶的最大值，且它的计数器归零了
            // 我们需要重新计算这个桶的最大值
            if (counts[m_index][w_index] == 0 && w_index == registers[m_index]) {
                // 从map中移除这个键
                counts[m_index].erase(w_index);
                
                // 寻找新的最大值
                unsigned new_max_w = 0;
                if (!counts[m_index].empty()) {
                    // map的最后一个元素就是键最大的元素
                    new_max_w = counts[m_index].rbegin()->first;
                }
                registers[m_index] = new_max_w;
            }
        }
    }

    double estimate() {
        double z = 0;
        unsigned zeros = 0;
        for (unsigned reg_val : registers) {
            if (reg_val == 0) {
                zeros++;
            }
            z += std::pow(2.0, -static_cast<double>(reg_val));
        }

        double estimate = alpha_m * m * m / z;

        // 保留小范围修正，这对于HLL的准确性至关重要
        if (m > 0 && estimate <= 2.5 * m && zeros > 0) {
            estimate = m * std::log(static_cast<double>(m) / zeros);
        }
        
        return estimate;
    }

private:
    unsigned m;
    double alpha_m;
    // HLL的核心，存储每个桶的MAX值
    std::vector<unsigned> registers;
    // 用于支持删除的计数器
    std::vector<std::map<unsigned, int>> counts;
};

#endif // _HLL_COUNT_HFX_H