#ifndef _FM_COUNT_H
#define _FM_COUNT_H

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <cstring>
#include <bit>

#include "MurmurHash3.h"
#include "parameters.h"

using namespace std;

// Global arrays for binary tree mechanism
double fm_tree_alpha[40];
double fm_tree_beta[40];
double phi = 0.77351;

// Estimation function
double fm_estimate(double z, unsigned fm_m, unsigned long long t)
{
    if (z <= 0) {return 1.0;}
    double double_m = fm_m;
    return min(double_m * pow(2.0, z / double_m) / phi, (double) t);
}

// Main experiment function with optimization
double fm_count_estimates(int stream_type, unsigned hash_seed)
{
    const unsigned fm_m = number_of_bits / 32;
    unsigned* fm_sketch = new unsigned[fm_m];
    
    memset(fm_sketch, 0, fm_m * sizeof(unsigned));
    memset(fm_tree_alpha, 0, 40 * 8);
    memset(fm_tree_beta, 0, 40 * 8);
    
    unsigned long long i;
    unsigned data_size = stream_lengths[stream_type];
    unsigned tree_height = ceil(log2(data_size + 1.0));
    double noisy_count = 0, estimated, temp_error;

    double error_sum_sq = 0.0;
    long long samples_count = 0;
    unsigned true_cardi = 0;
    const unsigned SAMPLING_RATE = 10000;

    for (i = 0; i < data_size; ++i)
    {
        unsigned hash_out[4] = {0};
        MurmurHash3_x64_128(data_array + i, 4, hash_seed, hash_out);
        unsigned m_index = hash_out[0] & (fm_m - 1);
        unsigned w_index = min((unsigned)countr_zero(hash_out[1]), (unsigned) 31);
        unsigned temp_row_z = countr_one(fm_sketch[m_index]);
        fm_sketch[m_index] |= (1U << w_index);
        temp_row_z = countr_one(fm_sketch[m_index]) - temp_row_z;
        
        unsigned j = countr_zero(i + 1);
        fm_tree_alpha[j] = temp_row_z;

        unsigned long long temp_t = (i << 8);
        MurmurHash3_x64_128(&temp_t, 8, hash_seed, hash_out);
        double temp_node = gene_laplace_32(hash_out[0], 2.0 * 32 * tree_height / epsilon);
        for (unsigned k = 0; k < j; ++ k)
        {
            fm_tree_alpha[j] += fm_tree_alpha[k];
            noisy_count -= (fm_tree_beta[k] + fm_tree_alpha[k]);
            fm_tree_alpha[k] = 0;
            fm_tree_beta[k] += temp_node;
            temp_t += 1;
            MurmurHash3_x64_128(&temp_t, 8, hash_seed, hash_out);
            double temp_weight = pow(2.0, k + 1) / (pow(2.0, k + 2) - 1.0);
            temp_node = temp_weight * fm_tree_beta[k] + (1.0 - temp_weight) * gene_laplace_32(hash_out[0], 2.0 * 32 * tree_height / epsilon);
            fm_tree_beta[k] = 0;
        }
        fm_tree_beta[j] = temp_node;
        noisy_count += (fm_tree_beta[j] + fm_tree_alpha[j]);

        true_cardi += (bool) (cardi_array[i / 32] & (1U << (i % 32)));

        // --- Performance Optimization: Sampled Error Calculation ---
        unsigned long long ignores = 1000;
        if (i >= ignores && (i % SAMPLING_RATE == 0))
        {
            estimated = fm_estimate(noisy_count, fm_m, i + 1);
            if(true_cardi > 0) {
                temp_error = (estimated - true_cardi) / (double) true_cardi;
                error_sum_sq += pow(temp_error, 2.0);
                samples_count++;
            }
        }
    }
    
    delete[] fm_sketch;
    if (samples_count > 0) {
        return pow(error_sum_sq / samples_count, 0.5);
    }
    return 0.0;
}

#endif