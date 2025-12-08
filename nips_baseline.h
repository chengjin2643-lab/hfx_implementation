#ifndef _NIPS_BASELINE_H
#define _NIPS_BASELINE_H

#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_set> // Using unordered_set to represent the linear space complexity
#include "MurmurHash3.h"
#include "parameters.h"

using namespace std;

// This file implements a simplified baseline inspired by the NeurIPS 2023 paper
// "Counting Distinct Elements in the Turnstile Model with Differential Privacy under Continual Observation"
// by Jain et al.
//
// Core Idea:
// 1.  Reduce the CountDistinct problem to a summation problem.
// 2.  Maintain the set of all unique items seen so far. This has LINEAR space complexity.
// 3.  At each time step t, calculate the change in distinct count: delta = 1 if new element, 0 otherwise.
// 4.  Feed this delta stream into the same refined binary-tree mechanism used by FC/HFX.
// 5.  The noisy sum output by the binary tree is the direct estimate of the cardinality.

double nips_baseline_estimates(int stream_type, unsigned hash_seed)
{
    // Data structure to store all unique elements, simulating linear space complexity.
    std::unordered_set<unsigned> distinct_elements;

    // Use the same binary tree structure as other algorithms for a fair comparison.
    double tree_alpha[40] = {0};
    double tree_beta[40] = {0};

    unsigned long long i_time_step, temp_t_for_noise_seed;
    unsigned true_cardi = 0;
    unsigned hash_out[4] = {0}; // For noise generation, not for the sketch itself
    unsigned data_size = stream_lengths[stream_type];
    unsigned tree_height = (data_size > 0) ? std::ceil(std::log2(data_size + 1.0)) : 1;
    unsigned j_node_idx, k_refinement_loop_idx;
    
    double noisy_cardinality_estimate = 0, error_sum_sq = 0, temp_rel_error;
    double current_bt, temp_node, temp_weight;

    for (i_time_step = 0; i_time_step < data_size; ++i_time_step)
    {
        unsigned element_data = data_array[i_time_step];
        
        // --- Core Logic of the Linear-Space Baseline ---
        // Check if the element is new by looking it up in the set.
        if (distinct_elements.find(element_data) == distinct_elements.end())
        {
            // Element is new. Insert it and set the delta to 1.
            distinct_elements.insert(element_data);
            current_bt = 1.0;
        }
        else
        {
            // Element is a duplicate. The change in distinct count is 0.
            current_bt = 0.0;
        }

        // --- Use the same refined binary-tree mechanism for DP ---
        j_node_idx = countr_zero(i_time_step + 1);
        tree_alpha[j_node_idx] = current_bt;

        // The sensitivity of a {0, 1} stream is 2 under the event-level privacy model used.
        temp_t_for_noise_seed = (i_time_step << 8);
        MurmurHash3_x64_128(&temp_t_for_noise_seed, sizeof(unsigned long long), hash_seed, hash_out);
        temp_node = gene_laplace_32(hash_out[0], 2.0 * (double)tree_height / epsilon);

        for (k_refinement_loop_idx = 0; k_refinement_loop_idx < j_node_idx; ++k_refinement_loop_idx)
        {
            tree_alpha[j_node_idx] += tree_alpha[k_refinement_loop_idx];
            noisy_cardinality_estimate -= (tree_beta[k_refinement_loop_idx] + tree_alpha[k_refinement_loop_idx]);
            tree_alpha[k_refinement_loop_idx] = 0;
            tree_beta[k_refinement_loop_idx] += temp_node;
            
            temp_t_for_noise_seed += 1;
            MurmurHash3_x64_128(&temp_t_for_noise_seed, sizeof(unsigned long long), hash_seed, hash_out);
            temp_weight = pow(2.0, k_refinement_loop_idx + 1) / (pow(2.0, k_refinement_loop_idx + 2) - 1.0);
            temp_node = temp_weight * tree_beta[k_refinement_loop_idx] + (1.0 - temp_weight) * gene_laplace_32(hash_out[0], 2.0 * (double)tree_height / epsilon);
            tree_beta[k_refinement_loop_idx] = 0;
        }
        tree_beta[j_node_idx] = temp_node;
        noisy_cardinality_estimate += (tree_beta[j_node_idx] + tree_alpha[j_node_idx]);

        // The output of the tree is the direct estimate.
        double estimated_n_t = noisy_cardinality_estimate;

        // --- Error Calculation Logic (remains the same) ---
        if (i_time_step < data_size && (i_time_step / 32) < (stream_lengths[stream_type] / 32 + (bool)(stream_lengths[stream_type] % 32))) {
             true_cardi += (bool) (cardi_array[i_time_step / 32] & (1U << (i_time_step % 32)));
        }
        
        unsigned long long ignores = 1000; 
        if (i_time_step >= ignores) 
        {
            if (true_cardi > 0) {
                temp_rel_error = (estimated_n_t - (double)true_cardi) / (double) true_cardi;
                error_sum_sq += pow(temp_rel_error, 2.0);
            } else if (estimated_n_t > 0) {
                error_sum_sq += pow(estimated_n_t, 2.0);
            }
        }
    }

    if (data_size <= 1000) return 0.0; 
    error_sum_sq = pow(error_sum_sq / (double)(data_size - 1000), 0.5);
    return error_sum_sq;
}

#endif // _NIPS_BASELINE_H