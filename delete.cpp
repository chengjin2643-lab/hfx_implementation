#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <map>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <fstream> 
#include <sstream> 

#include "parameters.h"
#include "hfx.h"
#include "nips_hfx.h"
#include "fm_deletable.h"
#include "hll_deletable.h"
#include "ll_deletable.h"
using namespace std;

// 定义全局变量（这些在 parameters.h 中声明为 extern）
unsigned number_of_bits;
double epsilon;
vector<string> stream_paths;
vector<unsigned> stream_lengths;
unsigned* data_array = nullptr;
unsigned* cardi_array = nullptr;

// 操作类型枚举
enum OperationType {
    INSERT = 1,
    DELETE = -1
};

// 操作结构体
struct Operation {
    unsigned element;
    OperationType type;
};

// 从 parameters.cpp 复制必要的函数实现
void load_unsigned_from_file(string file_name, unsigned * array, unsigned arraySize) {
    ifstream infile(file_name, std::ios::binary);
    if (!infile) { 
        cerr << "Error opening file for reading: " << file_name << endl; 
        return; 
    }
    infile.read(reinterpret_cast<char*>(array), arraySize * sizeof(unsigned));
    infile.close();
}

void load_dataset_from_file(string file_name, unsigned arraySize) {
    string dataset_file_name = file_name + "_dataset.bin";
    string cardinalities_file_name = file_name + "_cardinalities.bin";
    load_unsigned_from_file(dataset_file_name, data_array, arraySize);
    load_unsigned_from_file(cardinalities_file_name, cardi_array, (arraySize / 32 + (bool)(arraySize % 32)));
}

double gene_laplace_32(unsigned random_number, double b) {
    const unsigned max_unsigned = -1;
    if (random_number == 0 || random_number == max_unsigned) { 
        return sqrt(2) * b; 
    }
    double uni = (double)random_number / (double)max_unsigned - 0.5;
    if (uni >= 0) { 
        return b * log(1.0 - 2.0 * uni); 
    }
    else { 
        return -1.0 * b * log(1.0 + 2.0 * uni); 
    }
}

// 生成删除操作序列
vector<Operation> generate_delete_operations(unsigned initial_ndv, double delete_percentage, unsigned seed) {
    vector<Operation> operations;
    vector<unsigned> all_elements;
    
    // 收集所有元素
    for (unsigned i = 0; i < initial_ndv; ++i) {
        all_elements.push_back(data_array[i]);
        operations.push_back({data_array[i], INSERT});
    }
    
    // 计算要删除的元素数量
    unsigned delete_count = initial_ndv * delete_percentage;
    
    // 随机选择要删除的元素
    mt19937 gen(seed);
    shuffle(all_elements.begin(), all_elements.end(), gen);
    
    for (unsigned i = 0; i < delete_count && i < all_elements.size(); ++i) {
        operations.push_back({all_elements[i], DELETE});
    }
    
    return operations;
}

// HFX删除实验
double hfx_deletion_experiment(unsigned initial_ndv, double delete_percentage, unsigned hash_seed) {
    const unsigned hfx_w = 32;
    const unsigned hfx_m = number_of_bits / 32;
    unsigned mask_m = hfx_m - 1;
    
    auto hfx_sketch_B = new unsigned[hfx_m];
    double hfx_tree_alpha[40], hfx_tree_beta[40];
    memset(hfx_sketch_B, 0, hfx_m * sizeof(unsigned));
    memset(hfx_tree_alpha, 0, sizeof(hfx_tree_alpha));
    memset(hfx_tree_beta, 0, sizeof(hfx_tree_beta));
    
    // 生成操作序列
    vector<Operation> operations = generate_delete_operations(initial_ndv, delete_percentage, hash_seed);
    
    unsigned tree_height = ceil(log2(operations.size() + 1.0));
    double noisy_count = 0;
    
    // 跟踪活跃元素
    unordered_set<unsigned> active_elements;
    
    // 执行所有操作
    for (size_t op_idx = 0; op_idx < operations.size(); ++op_idx) {
        unsigned element = operations[op_idx].element;
        OperationType op_type = operations[op_idx].type;
        
        // 更新活跃元素集合
        if (op_type == INSERT) {
            active_elements.insert(element);
        } else {
            active_elements.erase(element);
        }
        
        // 执行sketch更新
        unsigned hash_out[4] = {0};
        MurmurHash3_x64_128(&element, 4, hash_seed, hash_out);
        unsigned m_index = hash_out[0] & mask_m;
        unsigned w_index = min((unsigned)std::countr_zero(hash_out[1]), (unsigned)31);
        
        double current_bt = 0.0;
        unsigned bit_mask = (1U << w_index);
        unsigned old_bit = (hfx_sketch_B[m_index] & bit_mask) ? 1 : 0;
        hfx_sketch_B[m_index] ^= bit_mask;
        unsigned new_bit = (hfx_sketch_B[m_index] & bit_mask) ? 1 : 0;
        
        if (old_bit == 0 && new_bit == 1) {
            current_bt = 1.0;
        } else if (old_bit == 1 && new_bit == 0) {
            current_bt = -1.0;
        }
        
        // 更新二叉树机制
        unsigned j = std::countr_zero(op_idx + 1);
        hfx_tree_alpha[j] = current_bt;
        
        unsigned long long temp_t = (op_idx << 8);
        MurmurHash3_x64_128(&temp_t, 8, hash_seed, hash_out);
        double temp_node = gene_laplace_32(hash_out[0], 2.0 * tree_height / epsilon);
        
        for (unsigned k = 0; k < j; ++k) {
            hfx_tree_alpha[j] += hfx_tree_alpha[k];
            noisy_count -= (hfx_tree_beta[k] + hfx_tree_alpha[k]);
            hfx_tree_alpha[k] = 0;
            hfx_tree_beta[k] += temp_node;
            temp_t += 1;
            MurmurHash3_x64_128(&temp_t, 8, hash_seed, hash_out);
            double temp_weight = pow(2.0, k + 1) / (pow(2.0, k + 2) - 1.0);
            temp_node = temp_weight * hfx_tree_beta[k] + (1.0 - temp_weight) * 
                        gene_laplace_32(hash_out[0], 2.0 * tree_height / epsilon);
            hfx_tree_beta[k] = 0;
        }
        hfx_tree_beta[j] = temp_node;
        noisy_count += (hfx_tree_beta[j] + hfx_tree_alpha[j]);
    }
    
    // 计算最终估计
    double estimated = hfx_cardinality_estimate_insertion_only(noisy_count, hfx_m, hfx_w, operations.size());
    unsigned true_cardi = active_elements.size();
    
    delete[] hfx_sketch_B;
    
    if (true_cardi > 0) {
        double error = (estimated - (double)true_cardi) / (double)true_cardi;
        return fabs(error);
    }
    return 0.0;
}

// NIPS HFX删除实验
double nips_hfx_deletion_experiment(unsigned initial_ndv, double delete_percentage, unsigned hash_seed) {
    const unsigned hfx_w = 32;
    const unsigned hfx_m = number_of_bits / hfx_w;
    unsigned mask_m = hfx_m - 1;
    
    auto hfx_sketch_B = new unsigned[hfx_m];
    memset(hfx_sketch_B, 0, hfx_m * sizeof(unsigned));
    
    double hfx_tree_alpha[40] = {0};
    double hfx_tree_beta[40] = {0};
    
    vector<Operation> operations = generate_delete_operations(initial_ndv, delete_percentage, hash_seed);
    
    unsigned tree_height = std::ceil(std::log2(operations.size() + 1.0));
    double noisy_count = 0;
    
    unordered_set<unsigned> active_elements;
    
    for (size_t op_idx = 0; op_idx < operations.size(); ++op_idx) {
        unsigned element = operations[op_idx].element;
        OperationType op_type = operations[op_idx].type;
        
        if (op_type == INSERT) {
            active_elements.insert(element);
        } else {
            active_elements.erase(element);
        }
        
        unsigned hash_out[4] = {0};
        MurmurHash3_x64_128(&element, sizeof(unsigned), hash_seed, hash_out);
        unsigned m_idx = hash_out[0] & mask_m;
        unsigned w_idx = std::min((unsigned)std::countr_zero(hash_out[1]), (unsigned)hfx_w - 1);
        unsigned bit_mask = (1U << w_idx);
        
        bool old_bit_is_0 = !(hfx_sketch_B[m_idx] & bit_mask);
        hfx_sketch_B[m_idx] ^= bit_mask;
        
        double current_bt = old_bit_is_0 ? 1.0 : -1.0;
        
        unsigned j_node_idx = std::countr_zero(op_idx + 1);
        hfx_tree_alpha[j_node_idx] = current_bt;
        
        unsigned long long temp_t_for_noise = (op_idx << 8);
        MurmurHash3_x64_128(&temp_t_for_noise, sizeof(unsigned long long), hash_seed, hash_out);
        double temp_node = gene_laplace_32(hash_out[0], 2.0 * (double)tree_height / epsilon);
        
        for (unsigned k = 0; k < j_node_idx; ++k) {
            hfx_tree_alpha[j_node_idx] += hfx_tree_alpha[k];
            noisy_count -= (hfx_tree_beta[k] + hfx_tree_alpha[k]);
            hfx_tree_alpha[k] = 0;
            hfx_tree_beta[k] += temp_node;
            
            temp_t_for_noise += 1;
            MurmurHash3_x64_128(&temp_t_for_noise, sizeof(unsigned long long), hash_seed, hash_out);
            double temp_weight = pow(2.0, k + 1) / (pow(2.0, k + 2) - 1.0);
            temp_node = temp_weight * hfx_tree_beta[k] + (1.0 - temp_weight) * 
                        gene_laplace_32(hash_out[0], 2.0 * (double)tree_height / epsilon);
            hfx_tree_beta[k] = 0;
        }
        hfx_tree_beta[j_node_idx] = temp_node;
        noisy_count += (hfx_tree_beta[j_node_idx] + hfx_tree_alpha[j_node_idx]);
    }
    
    delete[] hfx_sketch_B;
    
    double estimated = hfx_cardinality_estimate_dynamic(noisy_count, hfx_m, hfx_w, operations.size());
    unsigned true_cardi = active_elements.size();
    
    if (true_cardi > 0) {
        double error = (estimated - (double)true_cardi) / (double)true_cardi;
        return fabs(error);
    }
    return 0.0;
}

// 运行完整的删除实验
void run_deletion_experiments(unsigned random_seed, const string& output_filename) {
    
    vector<unsigned> initial_ndvs = {100, 10000, 1000000}; // 10^2, 10^4, 10^6
    vector<unsigned> memory_sizes_kb = {8, 32};
    vector<double> deletion_percentages = {0.1, 0.3, 0.5, 0.7, 0.9};
    
    // 设置epsilon
    epsilon = 2.0;
    
    ofstream results_file(output_filename);
    if (!results_file.is_open()) {
        cerr << "Error: Could not open output file " << output_filename << endl;
        return;
    }
    
    cout << "\n=================================================" << endl;
    cout << "     删除实验 (Delete Experiments)" << endl;
    cout << "=================================================" << endl;
    
    // 写入CSV表头
    results_file << "Initial_NDV,Memory_KB,Deletion_Percentage,HFX_Error,NIPS_HFX_Error,FM_Error,LL_Error,HLL_Error" << endl;
    
    for (unsigned initial_ndv : initial_ndvs) {
        // 生成初始数据集
        cout << "\n--- 生成NDV=" << initial_ndv << "的数据集 ---" << endl;
        
        stream_lengths = {initial_ndv, initial_ndv, initial_ndv};
        data_array = new unsigned[initial_ndv];
        cardi_array = new unsigned[initial_ndv / 32 + 1];
        
        // 生成不重复的随机数
        unordered_set<unsigned> unique_values;
        mt19937 gen(random_seed);
        uniform_int_distribution<unsigned> dis(0, UINT_MAX);
        
        while (unique_values.size() < initial_ndv) {
            unique_values.insert(dis(gen));
        }
        
        unsigned idx = 0;
        for (auto val : unique_values) {
            data_array[idx++] = val;
        }
        
        // 设置cardinality数组
        memset(cardi_array, 0xFF, (initial_ndv / 32 + 1) * sizeof(unsigned));
        
        for (unsigned mem_kb : memory_sizes_kb) {
            number_of_bits = mem_kb * 1024 * 8;
            
            cout << "\n内存大小: " << mem_kb << " KB" << endl;
            cout << "----------------------------------------" << endl;
            
            for (double del_pct : deletion_percentages) {
                cout << "\n删除比例: " << (del_pct * 100) << "%" << endl;
                
                stringstream result_line;
                result_line << initial_ndv << "," << mem_kb << "," << del_pct << ",";
                
                // 运行HFX删除实验
                double hfx_err = hfx_deletion_experiment(initial_ndv, del_pct, random_seed);
                cout << "  HFX Error: " << setw(12) << left << hfx_err << endl;
                result_line << hfx_err << ",";
                
                // 运行NIPS HFX删除实验
                double nips_hfx_err = nips_hfx_deletion_experiment(initial_ndv, del_pct, random_seed);
                cout << "  NIPS HFX Error: " << setw(12) << left << nips_hfx_err << endl;
                result_line << nips_hfx_err << ",";
                
                // 运行FM删除实验
                double fm_err = fm_count_estimates_with_delete(0, random_seed, del_pct);
                cout << "  FM Error: " << setw(12) << left << fm_err << endl;
                result_line << fm_err << ",";
                
                // 运行LL删除实验
                double ll_err = ll_count_estimates_with_delete(0, random_seed, del_pct);
                cout << "  LL Error: " << setw(12) << left << ll_err << endl;
                result_line << ll_err << ",";
                
                // 运行HLL删除实验
                double hll_err = hll_count_estimates_with_delete(0, random_seed, del_pct);
                cout << "  HLL Error: " << setw(12) << left << hll_err << endl;
                result_line << hll_err;
                
                results_file << result_line.str() << endl;
            }
        }
        
        delete[] data_array;
        delete[] cardi_array;
        data_array = nullptr;
        cardi_array = nullptr;
    }
    
    results_file.close();
    cout << "\n\n删除实验完成！结果已保存至 " << output_filename << endl;
}

// 主函数
int main(int argc, char* argv[])
{
    unsigned random_seed = 63242691;
    string output_filename = "deletion_experiment_results.csv";
    
    // 支持命令行参数
    if (argc > 1) {
        random_seed = atoi(argv[1]);
    }
    if (argc > 2) {
        output_filename = argv[2];
    }
    
    cout << "随机种子: " << random_seed << endl;
    cout << "输出文件: " << output_filename << endl;
    
    // 运行删除实验
    run_deletion_experiments(random_seed, output_filename);
    
    return 0;
}