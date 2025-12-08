

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
// #include "fc.h" // 不再需要FC
#include "hfx.h"
#include "nips_hfx.h"

// 引入基线方法
#include "fm_count.h"
#include "hll_count.h"
#include "ll_count.h"

using namespace std;

unsigned number_of_bits;

/**
 * @brief 针对给定的数据流类型（T）运行所有实验组合
 */
void run_experiments_for_stream(int stream_type, unsigned random_seed, const vector<string>& stream_paths, const vector<unsigned>& stream_lengths, ofstream& results_file) {
    
    data_array = new unsigned[stream_lengths[stream_type]];
    cardi_array = new unsigned[stream_lengths[stream_type] / 32 + 1];
    load_dataset_from_file(stream_paths[stream_type], stream_lengths[stream_type]);

    vector<double> epsilon_values = {0.5, 1.0, 2.0, 3.0, 4.0};
    vector<unsigned> memory_sizes_kb = {4, 8, 16, 32, 64};
    // vector<string> method_names = {"our HFX", "our FC", "NIPS HFX", "FM", "LL", "HLL"}; // FC已移除

    cout << "\n\n=================================================" << endl;
    cout << "--- 开始实验: T = " << stream_lengths[stream_type] << " ---" << endl;
    cout << "=================================================" << endl;

    for (unsigned mem_kb : memory_sizes_kb) {
        number_of_bits = mem_kb * 1024 * 8;
        
        // 由于FC已被移除，FMS估算表不再需要生成
        // cout << "\n--- 生成 FMS 估算表 for Memory = " << mem_kb << " KB... ---" << endl;
        // std::vector<double> fms_estimation_table = gene_fms_estimation_table();
        // cout << "--- 估算表生成完毕 ---" << endl;

        for (double current_epsilon : epsilon_values) {
            epsilon = current_epsilon;
            
            cout << "\n--- Testing Epsilon: " << epsilon << ", Memory: " << mem_kb << " KB ---" << endl;

            stringstream result_line;
            result_line << stream_lengths[stream_type] << "," << epsilon << "," << mem_kb << ",";

            // --- 运行各个算法并记录结果 ---
            
            // our HFX
            double hfx_err = hfx_estimates(stream_type, random_seed);
            cout << "RMSD of our HFX: " << setw(10) << left << hfx_err << endl;
            result_line << hfx_err << ",";
            
            // NIPS HFX
            double nips_hfx_err = nips_hfx_estimates(stream_type, random_seed);
            cout << "RMSD of NIPS HFX: " << setw(10) << left << nips_hfx_err << endl;
            result_line << nips_hfx_err << ",";

            // FM-Count
            double fm_err = fm_count_estimates(stream_type, random_seed);
            cout << "RMSD of FM:      " << setw(10) << left << fm_err << endl;
            result_line << fm_err << ",";

            // LL-Count
            double ll_err = ll_count_estimates(stream_type, random_seed);
            cout << "RMSD of LL:      " << setw(10) << left << ll_err << endl;
            result_line << ll_err << ",";

            // HLL-Count
            double hll_err = hll_count_estimates(stream_type, random_seed);
            cout << "RMSD of HLL:     " << setw(10) << left << hll_err << endl;
            result_line << hll_err;

            results_file << result_line.str() << endl;
        }
    }

    delete[] data_array;
    delete[] cardi_array;
    data_array = nullptr;
    cardi_array = nullptr;
}


int main()
{
    unsigned random_seed = 63242691;
    
    // 数据流路径和长度
    stream_paths = {
        "./datastreams/uniform_order_7", 
        "./datastreams/uniform_order_8", 
        "./datastreams/uniform_order_9"
    };
    stream_lengths = {
        10000000,   // T = 10^7
        100000000,  // T = 10^8
        1000000000  // T = 10^9
    };

    ofstream results_file("hfx_experiment_results.csv");
    if (!results_file.is_open()) {
        cerr << "Error: Could not open results file for writing." << endl;
        return 1;
    }

    // 更新CSV文件的表头，移除FC_RMSD
    results_file << "T,Epsilon,Memory_KB,HFX_RMSD,NIPS_HFX_RMSD,FM_RMSD,LL_RMSD,HLL_RMSD" << endl;

    // 循环遍历不同的T值
    for (size_t i = 0; i < stream_lengths.size(); ++i) {
        run_experiments_for_stream(i, random_seed, stream_paths, stream_lengths, results_file);
    }

    results_file.close();
    cout << "\n\n实验完成。结果已保存至 hfx_experiment_results.csv" << endl;
    
    return 0;
}