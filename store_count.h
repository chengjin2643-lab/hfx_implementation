#ifndef _STORE_COUNT_H
#define _STORE_COUNT_H

#include <bit>
#include <bitset>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <string>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <time.h>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include "MurmurHash3.h"
#include "parameters.h"

using namespace std;

double store_count_tree_alpha[40];
double store_count_tree_beta[40];


// assume bounded time step, and use the refined binary-tree mechanism
double store_count_estimates(int stream_type, unsigned hash_seed)
{
    memset(store_count_tree_alpha, 0, 40 * 8);
    memset(store_count_tree_beta, 0, 40 * 8);
    unsigned long long i, temp_t, ignores = 1000;
    unsigned true_cardi = 0;
    unsigned hash_out[4] = {0};
    unsigned data_size = stream_lengths[stream_type];
    unsigned tree_height = ceil(log2(data_size + 1)), j, k;
    double noisy_cardi = 0, error = 0, temp_error, temp_node, temp_weight;
    unordered_set<unsigned> stored_data;
    unsigned temp_element;
    for (i = 0; i < data_size; ++i)
    {
        j = countr_zero(i + 1);
        temp_element = data_array[i];
        if (stored_data.find(temp_element) == stored_data.end())
        {
            stored_data.insert(temp_element);
            true_cardi += 1;
            store_count_tree_alpha[j] = 1;
        }
        temp_t = (i << 8);
        MurmurHash3_x64_128(&temp_t, 8, hash_seed, hash_out);
        temp_node = gene_laplace_32(hash_out[0], 2.0 * tree_height / epsilon);
        for (k = 0; k < j; ++ k)
        {
            store_count_tree_alpha[j] += store_count_tree_alpha[k];
            noisy_cardi -= (store_count_tree_beta[k] + store_count_tree_alpha[k]);
            store_count_tree_alpha[k] = 0;
            store_count_tree_beta[k] += temp_node;
            temp_t += 1;
            MurmurHash3_x64_128(&temp_t, 8, hash_seed, hash_out);
            temp_weight = pow(2.0, k + 1) / (pow(2.0, k + 2) - 1.0);
            temp_node = temp_weight * store_count_tree_beta[k] + (1.0 - temp_weight) * gene_laplace_32(hash_out[0], 2.0 * tree_height / epsilon);
            store_count_tree_beta[k] = 0;
        }
        store_count_tree_beta[j] = temp_node;
        noisy_cardi += (store_count_tree_beta[j] + store_count_tree_alpha[j]);
        if (i > ignores)
        {
            temp_error = (noisy_cardi - true_cardi) / (double) true_cardi;
            error += pow(temp_error, 2.0);
        }
    }
    error =  pow(error / (data_size - ignores), 0.5);
    return error;
}


#endif
