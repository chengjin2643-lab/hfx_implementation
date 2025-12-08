#ifndef _HFX_H
#define _HFX_H

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

// --- Function Declarations ---

// Estimator model for DYNAMIC (XOR) sketches based on Half-Xor paper
double hfx_calculate_Yn_dynamic(double n_val, unsigned m_val, unsigned w_val);
// Binary search solver for the dynamic model
double hfx_binary_search_solve_n_dynamic(double V_target, unsigned m_val, unsigned w_val, unsigned long long current_t_elements_processed);
// Top-level estimator for the DYNAMIC version
double hfx_cardinality_estimate_dynamic(double Z_star_t, unsigned m_val, unsigned w_val, unsigned long long t_elements_processed);

// Estimator model for INSERTION-ONLY (OR) sketches (to compare with FC)
double hfx_calculate_Yn_insertion_only(double n_val, unsigned m_val, unsigned w_val);
// Binary search solver for the insertion-only model
double hfx_binary_search_solve_n_insertion_only(double V_target, unsigned m_val, unsigned w_val, unsigned long long current_t_elements_processed);
// Top-level estimator for the INSERTION-ONLY version
double hfx_cardinality_estimate_insertion_only(double Z_star_t, unsigned m_val, unsigned w_val, unsigned long long t_elements_processed);


// Main experiment functions
double hfx_estimates(int stream_type, unsigned hash_seed); // Insertion-only test
double hfx_estimates_dynamic(const std::vector<unsigned>& op_stream, unsigned hash_seed_param); // Deletion test


// ===================================================================================
//                  IMPLEMENTATIONS FOR DYNAMIC (XOR) ESTIMATION
// ===================================================================================

double hfx_calculate_Yn_dynamic(double n_val, unsigned m_val, unsigned w_val) {
    if (n_val < 0) n_val = 0;
    if (m_val == 0 || w_val == 0) return 0.0;
    
    double expected_set_bits_in_one_row = 0.0;
    const double lambda = 1.0 / (double)m_val; // As recommended by Half-Xor paper

    for (unsigned j_col = 0; j_col < w_val; ++j_col) {
        double p_j;
        if (j_col < w_val - 1) {
            p_j = pow(2.0, -(double)(j_col + 1));
        } else {
            p_j = pow(2.0, -(double)(w_val - 1));
        }
        
        // Correct model from Half-Xor Theorem 2: P(B[i][j]=1) = 0.5 * (1 - exp(-n*lambda*p_j))
        expected_set_bits_in_one_row += 0.5 * (1.0 - exp(-n_val * lambda * p_j));
    }
    
    double total_expected_Z = (double)m_val * expected_set_bits_in_one_row;
    return total_expected_Z / (double)(m_val * w_val);
}

double hfx_binary_search_solve_n_dynamic(double V_target, unsigned m_val, unsigned w_val, unsigned long long current_t_elements_processed) {
    if (V_target < 0.0) V_target = 0.0;
    if (V_target > 0.5) V_target = 0.5;

    double low_n = 0.0;
    double high_n = (double)current_t_elements_processed * 2.0 + 1000.0;
    if (high_n <= low_n) high_n = low_n + 1000.0;

    for(int iter = 0; iter < 100; ++iter) { 
        double mid_n = low_n + (high_n - low_n) / 2.0;
        double y_mid = hfx_calculate_Yn_dynamic(mid_n, m_val, w_val);
        if (y_mid < V_target) {
            low_n = mid_n;
        } else {
            high_n = mid_n;
        }
    }
    double estimated_n = low_n + (high_n - low_n) / 2.0;
    return std::min(std::max(0.0, estimated_n), (double)current_t_elements_processed * 2.0); // Cap at a reasonable upper bound
}

double hfx_cardinality_estimate_dynamic(double Z_star_t, unsigned m_val, unsigned w_val, unsigned long long t_elements_processed) {
    if (m_val == 0 || w_val == 0) return 0.0; 
    double V = Z_star_t / (double)(m_val * w_val);
    return hfx_binary_search_solve_n_dynamic(V, m_val, w_val, t_elements_processed);
}


// ===================================================================================
//               IMPLEMENTATIONS FOR INSERTION-ONLY (OR) ESTIMATION
// ===================================================================================

double hfx_calculate_Yn_insertion_only(double n_val, unsigned m_val, unsigned w_val) {
    if (n_val < 0) n_val = 0;
    if (m_val == 0 || w_val == 0) return 0.0;
    
    double expected_set_bits_in_one_row = 0.0;
    const double p_base = 1.0 / (double)m_val;

    for (unsigned j_col = 0; j_col < w_val; ++j_col) {
        double p_j;
        if (j_col < w_val - 1) {
            p_j = pow(2.0, -(double)(j_col + 1));
        } else {
            p_j = pow(2.0, -(double)(w_val - 1));
        }
        
        double p_hit_specific_bit = p_base * p_j;
        expected_set_bits_in_one_row += (1.0 - pow(1.0 - p_hit_specific_bit, n_val));
    }
    
    double total_expected_Z = (double)m_val * expected_set_bits_in_one_row;
    return total_expected_Z / (double)(m_val * w_val);
}

double hfx_binary_search_solve_n_insertion_only(double V_target, unsigned m_val, unsigned w_val, unsigned long long current_t_elements_processed) {
    if (V_target < 0.0) V_target = 0.0;
    if (V_target > 1.0) V_target = 1.0;

    double low_n = 0.0;
    double high_n = (double)current_t_elements_processed * 2.0 + 1000.0;
    if (high_n <= low_n) high_n = low_n + 1000.0;

    for(int iter = 0; iter < 100; ++iter) { 
        double mid_n = low_n + (high_n - low_n) / 2.0;
        double y_mid = hfx_calculate_Yn_insertion_only(mid_n, m_val, w_val);
        if (y_mid < V_target) {
            low_n = mid_n;
        } else {
            high_n = mid_n;
        }
    }
    double estimated_n = low_n + (high_n - low_n) / 2.0;
    return std::min(std::max(0.0, estimated_n), (double)current_t_elements_processed + 1.0);
}

double hfx_cardinality_estimate_insertion_only(double Z_star_t, unsigned m_val, unsigned w_val, unsigned long long t_elements_processed) {
    if (m_val == 0 || w_val == 0) return 0.0;
    double V = Z_star_t / (double)(m_val * w_val);
    return hfx_binary_search_solve_n_insertion_only(V, m_val, w_val, t_elements_processed);
}


// ===================================================================================
//                        MAIN EXPERIMENT FUNCTIONS
// ===================================================================================

double hfx_estimates(int stream_type, unsigned hash_seed)
{
    const unsigned hfx_w = 32;
    const unsigned hfx_m = number_of_bits / 32;
    unsigned mask_m = hfx_m - 1;
    
    auto hfx_sketch_B = new unsigned[hfx_m];
    double hfx_tree_alpha[40], hfx_tree_beta[40];
    memset(hfx_sketch_B, 0, hfx_m * sizeof(unsigned));
    memset(hfx_tree_alpha, 0, sizeof(hfx_tree_alpha));
    memset(hfx_tree_beta, 0, sizeof(hfx_tree_beta));
    
    unsigned long long i;
    unsigned true_cardi = 0;
    unsigned hash_out[4] = {0};
    unsigned data_size = stream_lengths[stream_type];
    unsigned tree_height = ceil(log2(data_size + 1.0));
    unsigned j, k;
    double noisy_count = 0, estimated, temp_error;

    double error_sum_sq = 0.0;
    long long samples_count = 0;
    const unsigned SAMPLING_RATE = 10000; // Sample every 10000 steps

    for (i = 0; i < data_size; ++i) {
        MurmurHash3_x64_128(data_array + i, 4, hash_seed, hash_out);
        unsigned m_index = hash_out[0] & mask_m;
        unsigned w_index = min((unsigned)std::countr_zero(hash_out[1]), (unsigned)31);
        
        double current_bt = 0.0;
        unsigned bit_mask = (1U << w_index);
        if (!(hfx_sketch_B[m_index] & bit_mask)) {
            current_bt = 1.0;
            hfx_sketch_B[m_index] |= bit_mask;
        }

        j = std::countr_zero(i + 1);
        hfx_tree_alpha[j] = current_bt;
        
        unsigned long long temp_t = (i << 8);
        MurmurHash3_x64_128(&temp_t, 8, hash_seed, hash_out);
        double temp_node = gene_laplace_32(hash_out[0], 2.0 * tree_height / epsilon);
        
        for (k = 0; k < j; ++k) {
            hfx_tree_alpha[j] += hfx_tree_alpha[k];
            noisy_count -= (hfx_tree_beta[k] + hfx_tree_alpha[k]);
            hfx_tree_alpha[k] = 0;
            hfx_tree_beta[k] += temp_node;
            temp_t += 1;
            MurmurHash3_x64_128(&temp_t, 8, hash_seed, hash_out);
            double temp_weight = pow(2.0, k + 1) / (pow(2.0, k + 2) - 1.0);
            temp_node = temp_weight * hfx_tree_beta[k] + (1.0 - temp_weight) * gene_laplace_32(hash_out[0], 2.0 * tree_height / epsilon);
            hfx_tree_beta[k] = 0;
        }
        hfx_tree_beta[j] = temp_node;
        noisy_count += (hfx_tree_beta[j] + hfx_tree_alpha[j]);

        true_cardi += (bool) (cardi_array[i / 32] & (1U << (i % 32)));
        
        // --- Performance Optimization: Sampled Error Calculation ---
        unsigned long long ignores = 1000;
        if (i >= ignores && (i % SAMPLING_RATE == 0)) {
            estimated = hfx_cardinality_estimate_insertion_only(noisy_count, hfx_m, hfx_w, i + 1);
            if (true_cardi > 0) {
                temp_error = (estimated - (double)true_cardi) / (double)true_cardi;
                error_sum_sq += pow(temp_error, 2.0);
                samples_count++;
            }
        }
    }

    delete[] hfx_sketch_B;
    if (samples_count > 0) {
        return pow(error_sum_sq / samples_count, 0.5);
    }
    return 0.0;
}

double hfx_estimates_dynamic(const std::vector<unsigned>& op_stream, unsigned hash_seed_param)
{
    const unsigned hfx_w = 32;
    const unsigned hfx_m = number_of_bits / hfx_w;
    unsigned mask_m = hfx_m - 1;

    auto hfx_sketch_B = new unsigned[hfx_m];
    memset(hfx_sketch_B, 0, hfx_m * sizeof(unsigned));
    
    double hfx_tree_alpha[40] = {0};
    double hfx_tree_beta[40] = {0};

    unsigned data_size = op_stream.size();
    unsigned tree_height = (data_size > 0) ? std::ceil(std::log2(data_size + 1.0)) : 1;
    double noisy_count = 0;

    for (unsigned long long i_time_step = 0; i_time_step < data_size; ++i_time_step)
    {
        unsigned element_data = op_stream[i_time_step];
        
        unsigned hash_out[4] = {0};
        MurmurHash3_x64_128(&element_data, sizeof(unsigned), hash_seed_param, hash_out);
        unsigned m_idx = hash_out[0] & mask_m;
        unsigned w_idx_in_row = std::min((unsigned)std::countr_zero(hash_out[1]), (unsigned)hfx_w - 1);
        unsigned bit_mask_for_sketch = (1U << w_idx_in_row);

        bool old_bit_is_0 = !(hfx_sketch_B[m_idx] & bit_mask_for_sketch);
        hfx_sketch_B[m_idx] ^= bit_mask_for_sketch;
        
        double current_bt = old_bit_is_0 ? 1.0 : -1.0;
        
        unsigned j_node_idx = std::countr_zero(i_time_step + 1);
        hfx_tree_alpha[j_node_idx] = current_bt;
        
        unsigned long long temp_t_for_noise_seed = (i_time_step << 8);
        MurmurHash3_x64_128(&temp_t_for_noise_seed, sizeof(unsigned long long), hash_seed_param, hash_out);
        double temp_node = gene_laplace_32(hash_out[0], 2.0 * (double)tree_height / epsilon);

        for (unsigned k = 0; k < j_node_idx; ++k)
        {
            hfx_tree_alpha[j_node_idx] += hfx_tree_alpha[k];
            noisy_count -= (hfx_tree_beta[k] + hfx_tree_alpha[k]);
            hfx_tree_alpha[k] = 0;
            hfx_tree_beta[k] += temp_node;
            
            temp_t_for_noise_seed += 1;
            MurmurHash3_x64_128(&temp_t_for_noise_seed, sizeof(unsigned long long), hash_seed_param, hash_out);
            double temp_weight = pow(2.0, k + 1) / (pow(2.0, k + 2) - 1.0);
            temp_node = temp_weight * hfx_tree_beta[k] + (1.0 - temp_weight) * gene_laplace_32(hash_out[0], 2.0 * (double)tree_height / epsilon);
            hfx_tree_beta[k] = 0;
        }
        hfx_tree_beta[j_node_idx] = temp_node;
        noisy_count += (hfx_tree_beta[j_node_idx] + hfx_tree_alpha[j_node_idx]);
    }

    delete[] hfx_sketch_B;
    // CRITICAL FIX: Ensure dynamic estimator is called here
    return hfx_cardinality_estimate_dynamic(noisy_count, hfx_m, hfx_w, data_size);
}

#endif // _HFX_H