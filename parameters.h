#ifndef _PARAMETERS_H
#define _PARAMETERS_H

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <random> // For Laplace noise generation
#include "MurmurHash3.h"

using namespace std;

// --- Global Parameters (that are truly constant or configured once) ---
extern double epsilon; // Use extern to declare it's defined elsewhere (dpccr.cpp)
extern vector<string> stream_paths;
extern vector<unsigned> stream_lengths;
extern unsigned* data_array;  // These will be allocated in main
extern unsigned* cardi_array; // These will be allocated in main

// Use extern for the global variable that will be defined in dpccr.cpp
extern unsigned number_of_bits;

// --- Function Declarations ---

// This function now takes the estimation table as an argument
double fms_table_estimate(int count, unsigned long long t, const std::vector<double>& estimation_table);

// This function now returns the generated table instead of modifying a global one
std::vector<double> gene_fms_estimation_table();

// Other utility functions remain the same
double gene_laplace_32(unsigned random_number, double b);
void load_dataset_from_file(string file_name, unsigned arraySize);

enum OperationType {
    INSERT = 1,
    DELETE = -1
};

// 用于存储带操作类型的数据流
struct Operation {
    unsigned element;
    OperationType type;
};

#endif // _PARAMETERS_H