#ifndef _LL_COUNT_H
#define _LL_COUNT_H

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

// Global arrays and constants
double ll_tree_alpha[40];
double ll_tree_beta[40];
double ll_alpha_m = 0.39701;

// Estimation function
double ll_estimate(double z, unsigned ll_m, unsigned long long t)
{
    double ll_double_m = (double) ll_m;
    if (z <= 0) {return 1.0;}
    return min(ll_alpha_m * ll_double_m * pow(2.0, z / ll_double_m), (double) t);
}

// Main experiment function with optimization
double ll_count_estimates(int stream_type, unsigned hash_seed)
{
    const unsigned ll_m = number_of_bits / 6;
    unsigned* ll_sketch = new unsigned[ll_m];
    
    memset(ll_sketch, 0, ll_m * sizeof(unsigned));
    memset(ll_tree_alpha, 0, 40 * 8);
    memset(ll_tree_beta, 0, 40 * 8);
    
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
        unsigned m_index = hash_out[0] % ll_m;
        unsigned w_index = (unsigned)countr_zero(hash_out[1]);
        unsigned temp_row_z = ll_sketch[m_index];
        ll_sketch[m_index] = max(ll_sketch[m_index], w_index + 1);
        temp_row_z = ll_sketch[m_index] - temp_row_z;

        unsigned j = countr_zero(i + 1);
        ll_tree_alpha[j] = temp_row_z;

        unsigned long long temp_t = (i << 8);
        MurmurHash3_x64_128(&temp_t, 8, hash_seed, hash_out);
        double temp_node = gene_laplace_32(hash_out[0], 2.0 * 32 * tree_height / epsilon);
        for (unsigned k = 0; k < j; ++ k)
        {
            ll_tree_alpha[j] += ll_tree_alpha[k];
            noisy_count -= (ll_tree_beta[k] + ll_tree_alpha[k]);
            ll_tree_alpha[k] = 0;
            ll_tree_beta[k] += temp_node;
            temp_t += 1;
            MurmurHash3_x64_128(&temp_t, 8, hash_seed, hash_out);
            double temp_weight = pow(2.0, k + 1) / (pow(2.0, k + 2) - 1.0);
            temp_node = temp_weight * ll_tree_beta[k] + (1.0 - temp_weight) * gene_laplace_32(hash_out[0], 2.0 * 32 * tree_height / epsilon);
            ll_tree_beta[k] = 0;
        }
        ll_tree_beta[j] = temp_node;
        noisy_count += (ll_tree_beta[j] + ll_tree_alpha[j]);
        
        true_cardi += (bool) (cardi_array[i / 32] & (1U << (i % 32)));
        
        // --- Performance Optimization: Sampled Error Calculation ---
        unsigned long long ignores = 1000;
        if (i >= ignores && (i % SAMPLING_RATE == 0))
        {
            estimated = ll_estimate(noisy_count, ll_m, i + 1);
            if(true_cardi > 0) {
                temp_error = (estimated - true_cardi) / (double) true_cardi;
                error_sum_sq += pow(temp_error, 2.0);
                samples_count++;
            }
        }
    }
    
    delete[] ll_sketch;
    if (samples_count > 0) {
        return pow(error_sum_sq / samples_count, 0.5);
    }
    return 0.0;
}

#endif