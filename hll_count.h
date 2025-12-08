#ifndef _HLL_COUNT_H
#define _HLL_COUNT_H

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

// Global arrays
double hll_tree_alpha[40];
double hll_tree_beta[40];

// Helper functions for HLL
double hll_alpha_m_compute(unsigned hll_m)
{
    if (hll_m == 16) {return 0.673;}
    else if (hll_m == 32) {return 0.697;}
    else if (hll_m == 64) {return 0.709;}
    else {return 0.7213 * (double) hll_m / ((double) hll_m + 1.079);}
}

double hll_estimate(double z, unsigned hll_m, double hll_alpha_m, unsigned long long t)
{
    double hll_double_m = (double) hll_m;
    double ret;
    if (z >= hll_double_m) {ret = 1.0;}
    else if (z > 0) {ret = min(hll_alpha_m * hll_double_m * hll_double_m / z, (double) t);}
    else {ret = min(hll_alpha_m * hll_double_m / pow(2.0, - 32), (double) t);}
    return ret;
}

// Main experiment function with optimization
double hll_count_estimates(int stream_type, unsigned hash_seed)
{
    const unsigned hll_m = number_of_bits / 6;
    unsigned* hll_sketch = new unsigned[hll_m];
    double hll_alpha_m = hll_alpha_m_compute(hll_m);
    
    memset(hll_sketch, 0, hll_m * sizeof(unsigned));
    memset(hll_tree_alpha, 0, 40 * 8);
    memset(hll_tree_beta, 0, 40 * 8);
    
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
        unsigned m_index = hash_out[0] % hll_m;
        unsigned w_index = (unsigned)countr_zero(hash_out[1]);
        double temp_row_z = pow(2.0, - (double) hll_sketch[m_index]);
        hll_sketch[m_index] = max(hll_sketch[m_index], w_index + 1);
        temp_row_z -= pow(2.0, - (double) hll_sketch[m_index]);

        unsigned j = countr_zero(i + 1);
        hll_tree_alpha[j] = temp_row_z;

        unsigned long long temp_t = (i << 8);
        MurmurHash3_x64_128(&temp_t, 8, hash_seed, hash_out);
        double temp_node = gene_laplace_32(hash_out[0], 2.0 * (double) tree_height / epsilon);
        for (unsigned k = 0; k < j; ++ k)
        {
            hll_tree_alpha[j] += hll_tree_alpha[k];
            noisy_count -= (hll_tree_beta[k] + hll_tree_alpha[k]);
            hll_tree_alpha[k] = 0;
            hll_tree_beta[k] += temp_node;
            temp_t += 1;
            MurmurHash3_x64_128(&temp_t, 8, hash_seed, hash_out);
            double temp_weight = pow(2.0, k + 1) / (pow(2.0, k + 2) - 1.0);
            temp_node = temp_weight * hll_tree_beta[k] + (1.0 - temp_weight) * gene_laplace_32(hash_out[0], 2.0 * (double) tree_height / epsilon);
            hll_tree_beta[k] = 0;
        }
        hll_tree_beta[j] = temp_node;
        noisy_count += (hll_tree_beta[j] + hll_tree_alpha[j]);
        
        true_cardi += (bool) (cardi_array[i / 32] & (1U << (i % 32)));
        
        // --- Performance Optimization: Sampled Error Calculation ---
        unsigned long long ignores = 1000;
        if (i >= ignores && (i % SAMPLING_RATE == 0))
        {
            double hll_double_m = (double) hll_m;
            estimated = hll_estimate(hll_double_m - noisy_count, hll_m, hll_alpha_m, i + 1);
            if(true_cardi > 0) {
                temp_error = (estimated - true_cardi) / (double) true_cardi;
                error_sum_sq += pow(temp_error, 2.0);
                samples_count++;
            }
        }
    }
    
    delete[] hll_sketch;
    if (samples_count > 0) {
        return pow(error_sum_sq / samples_count, 0.5);
    }
    return 0.0;
}

#endif