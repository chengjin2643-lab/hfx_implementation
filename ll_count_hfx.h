// ll_count_hfx.h

#ifndef _LL_COUNT_HFX_H
#define _LL_COUNT_HFX_H

#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <map> // 使用map来存储稀疏的计数器
#include "MurmurHash3.h"
#include "parameters.h"

class LL_Deletable_HFX {
public:
    LL_Deletable_HFX(unsigned total_bits) {
        // LogLog通常也使用6 bits每个register
        m = total_bits / 6;
        if (m == 0) m = 1;

        // 初始化LogLog的寄存器
        registers.assign(m, 0);
        // 初始化计数器，每个桶一个map
        counts.resize(m);

        // LogLog算法使用的alpha值
        alpha_m = 0.39701; 
    }

    void insert(const std::string& item) {
        uint64_t hash_out[2];
        // 使用一个与HLL和FM不同的seed
        MurmurHash3_x64_128(item.c_str(), item.length(), 456, &hash_out);
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
        MurmurHash3_x64_128(item.c_str(), item.length(), 456, &hash_out);
        unsigned m_index = hash_out[0] % m;
        unsigned w_index = std::countr_zero(hash_out[1]) + 1;

        // 对应值的计数器减一
        if (counts[m_index].count(w_index) && counts[m_index][w_index] > 0) {
            counts[m_index][w_index]--;

            // 如果被删除的值是当前桶的最大值，且它的计数器归零了
            // 我们需要重新计算这个桶的最大值
            if (counts[m_index][w_index] == 0 && w_index == registers[m_index]) {
                counts[m_index].erase(w_index);
                
                // 寻找新的最大值
                unsigned new_max_w = 0;
                if (!counts[m_index].empty()) {
                    new_max_w = counts[m_index].rbegin()->first;
                }
                registers[m_index] = new_max_w;
            }
        }
    }

    double estimate() {
        // LogLog的估算方式是计算所有寄存器值的算术平均数
        double sum_of_ranks = 0;
        for (unsigned reg_val : registers) {
            sum_of_ranks += reg_val;
        }

        if (m == 0) return 0.0;

        double avg_rank = sum_of_ranks / static_cast<double>(m);
        
        // 应用LogLog的估算公式
        double estimate = alpha_m * m * std::pow(2.0, avg_rank);
        
        // LogLog不像HLL那样有标准化的修正方法，但为了避免极端情况
        // 我们可以加一个简单的上限
        return std::max(0.0, estimate);
    }

private:
    unsigned m;
    double alpha_m;
    // LogLog的核心，存储每个桶的MAX值
    std::vector<unsigned> registers;
    // 用于支持删除的计数器
    std::vector<std::map<unsigned, int>> counts;
};

#endif // _LL_COUNT_HFX_H