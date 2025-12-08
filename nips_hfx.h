#ifndef _NIPS_HFX_H
#define _NIPS_HFX_H

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

// NIPS HFX Implementation based on the paper
// "Dynamic Differentially Private Continual Release NDV Estimation"

// Function to calculate γ(n) as defined in the NIPS paper
double nips_calculate_gamma(double n, unsigned m, unsigned w) {
    if (n < 0) n = 0;
    if (m == 0 || w == 0) return 0.0;
    
    double sum = 0.0;
    double lambda = 1.0 / (double)m;
    
    for (unsigned j = 0; j < w; ++j) {
        double p_j;
        if (j < w - 1) {
            p_j = pow(2.0, -(double)(j + 1));
        } else {
            p_j = pow(2.0, -(double)(w - 1));
        }
        
        sum += (1.0 - exp(-n * lambda * p_j));
    }
    
    return 0.5 * sum;
}

// Binary search to solve γ(n) = V
double nips_binary_search_solve(double V_target, unsigned m, unsigned w, unsigned long long max_n) {
    if (V_target < 0.0) V_target = 0.0;
    if (V_target > (double)w * 0.5) V_target = (double)w * 0.5;
    
    double left_n = 0.0;
    double right_n = (double)max_n * 2.0 + 1000.0;
    
    for (int iter = 0; iter < 100; ++iter) {
        double mid_n = left_n + (right_n - left_n) / 2.0;
        double gamma_mid = nips_calculate_gamma(mid_n, m, w);
        
        if (gamma_mid < V_target) {
            left_n = mid_n;
        } else {
            right_n = mid_n;
        }
    }
    
    return (left_n + right_n) / 2.0;
}

// NIPS cardinality estimation function
double nips_cardinality_estimate(double Z_star_t, unsigned m, unsigned w, unsigned long long t) {
    if (m == 0 || w == 0) return 0.0;
    double V = Z_star_t / (double)(m * w);
    return nips_binary_search_solve(V, m, w, t);
}

// Main NIPS HFX estimation function for insertion-only streams
double nips_hfx_estimates(int stream_type, unsigned hash_seed) {
    const unsigned nips_w = 32;
    const unsigned nips_m = number_of_bits / nips_w;
    unsigned mask_m = nips_m - 1;
    
    auto nips_B = new unsigned[nips_m];
    memset(nips_B, 0, nips_m * sizeof(unsigned));
    
    const unsigned H_tree = ceil(log2(stream_lengths[stream_type] + 1.0));
    double* nips_U = new double[H_tree];
    double* nips_P = new double[H_tree];
    memset(nips_U, 0, H_tree * sizeof(double));
    memset(nips_P, 0, H_tree * sizeof(double));
    
    unsigned long long i;
    unsigned true_cardi = 0;
    unsigned hash_out[4] = {0};
    unsigned data_size = stream_lengths[stream_type];
    double Z_star_t = 0.0;
    
    double error_sum_sq = 0.0;
    long long samples_count = 0;
    const unsigned SAMPLING_RATE = 10000;

    for (i = 0; i < data_size; ++i) {
        MurmurHash3_x64_128(data_array + i, 4, hash_seed, hash_out);
        unsigned i_idx = hash_out[0] & mask_m;
        unsigned j_idx = min((unsigned)std::countr_zero(hash_out[1]), nips_w - 1);
        
        unsigned bit_mask = (1U << j_idx);
        bool B_old = (nips_B[i_idx] & bit_mask) != 0;
        nips_B[i_idx] ^= bit_mask;
        double b_t = !B_old ? 1.0 : -1.0;
        
        unsigned idx = std::countr_zero(i + 1);
        double temp_sum = b_t;
        for (unsigned k = 0; k < idx; ++k) { temp_sum += nips_U[k]; }
        nips_U[idx] = temp_sum;
        
        unsigned long long temp_t_noise = (i << 8);
        MurmurHash3_x64_128(&temp_t_noise, 8, hash_seed, hash_out);
        double xi_t = gene_laplace_32(hash_out[0], 2.0 * (double)H_tree / epsilon);
        nips_P[idx] = nips_U[idx] + xi_t;
        
        for (unsigned k = 0; k < idx; ++k) { nips_U[k] = 0.0; nips_P[k] = 0.0; }
        
        Z_star_t = 0.0;
        unsigned t_binary = i + 1;
        for (unsigned k = 0; k < H_tree; ++k) {
            if (t_binary & (1U << k)) { Z_star_t += nips_P[k]; }
        }
        
        true_cardi += (bool)(cardi_array[i / 32] & (1U << (i % 32)));

        // --- Performance Optimization: Sampled Error Calculation ---
        unsigned long long ignores = 1000;
        if (i >= ignores && (i % SAMPLING_RATE == 0)) {
            double estimated = nips_cardinality_estimate(Z_star_t, nips_m, nips_w, i + 1);
            if (true_cardi > 0) {
                double temp_error = (estimated - (double)true_cardi) / (double)true_cardi;
                error_sum_sq += pow(temp_error, 2.0);
                samples_count++;
            }
        }
    }
    
    delete[] nips_B;
    delete[] nips_U;
    delete[] nips_P;
    
    if (samples_count > 0) {
        return sqrt(error_sum_sq / samples_count);
    }
    return 0.0;
}

// Dynamic version for deletion support (based on the paper's XOR approach)
double nips_hfx_estimates_dynamic(const std::vector<unsigned>& op_stream, unsigned hash_seed_param) {
    const unsigned nips_w = 32;
    const unsigned nips_m = number_of_bits / nips_w;
    unsigned mask_m = nips_m - 1;
    
    auto nips_B = new unsigned[nips_m];
    memset(nips_B, 0, nips_m * sizeof(unsigned));
    
    unsigned data_size = op_stream.size();
    const unsigned H_tree = (data_size > 0) ? ceil(log2(data_size + 1.0)) : 1;
    double* nips_U = new double[H_tree];
    double* nips_P = new double[H_tree];
    memset(nips_U, 0, H_tree * sizeof(double));
    memset(nips_P, 0, H_tree * sizeof(double));
    
    double Z_star_t = 0.0;
    
    for (unsigned long long i = 0; i < data_size; ++i) {
        unsigned element_data = op_stream[i];
        
        unsigned hash_out[4] = {0};
        MurmurHash3_x64_128(&element_data, sizeof(unsigned), hash_seed_param, hash_out);
        unsigned i_idx = hash_out[0] & mask_m;
        unsigned j_idx = min((unsigned)std::countr_zero(hash_out[1]), nips_w - 1);
        
        unsigned bit_mask = (1U << j_idx);
        bool B_old = (nips_B[i_idx] & bit_mask) != 0;
        nips_B[i_idx] ^= bit_mask;
        
        double b_t = B_old ? -1.0 : 1.0;
        
        unsigned idx = std::countr_zero(i + 1);
        double temp_sum = b_t;
        for (unsigned k = 0; k < idx; ++k) {
            temp_sum += nips_U[k];
        }
        nips_U[idx] = temp_sum;
        
        unsigned long long temp_t_noise = (i << 8);
        MurmurHash3_x64_128(&temp_t_noise, sizeof(unsigned long long), hash_seed_param, hash_out);
        double xi_t = gene_laplace_32(hash_out[0], 2.0 * (double)H_tree / epsilon);
        nips_P[idx] = nips_U[idx] + xi_t;
        
        for (unsigned k = 0; k < idx; ++k) {
            nips_U[k] = 0.0;
            nips_P[k] = 0.0;
        }
        
        Z_star_t = 0.0;
        unsigned t_binary = i + 1;
        for (unsigned k = 0; k < H_tree; ++k) {
            if (t_binary & (1U << k)) {
                Z_star_t += nips_P[k];
            }
        }
    }
    
    double result = nips_cardinality_estimate(Z_star_t, nips_m, nips_w, data_size);
    
    delete[] nips_B;
    delete[] nips_U;
    delete[] nips_P;
    
    return result;
}

#endif // _NIPS_HFX_H