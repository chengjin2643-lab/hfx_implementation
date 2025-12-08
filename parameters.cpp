#include "parameters.h"
#include <iomanip>

// Define the global variables that were declared with extern in the header
double epsilon;
vector<string> stream_paths;
vector<unsigned> stream_lengths;
unsigned* data_array = nullptr;
unsigned* cardi_array = nullptr;


// --- Helper function implementations that were in parameters.h ---
void load_unsigned_from_file(string file_name, unsigned * array, unsigned arraySize) {
    ifstream infile(file_name, std::ios::binary);
    if (!infile) { cerr << "Error opening file for reading: " << file_name << endl; return; }
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
    if (random_number == 0 || random_number == max_unsigned) { return sqrt(2) * b; }
    double uni = (double)random_number / (double)max_unsigned - 0.5;
    if (uni >= 0) { return b * log(1.0 - 2.0 * uni); }
    else { return -1.0 * b * log(1.0 + 2.0 * uni); }
}

void save_double_to_file(string file_name, const std::vector<double>& array) {
    ofstream outfile(file_name, std::ios::binary);
    if(!outfile) {cerr << "Error opening file for writing" << endl; return;}
    outfile.write(reinterpret_cast<const char *>(array.data()), array.size() * sizeof(double));
    outfile.close();
}

void load_double_from_file(string file_name, std::vector<double>& array) {
    ifstream infile(file_name, std::ios::binary);
    if (!infile) {cerr << "Error opening file for reading" << endl; return;}
    infile.read(reinterpret_cast<char *>(array.data()), array.size() * sizeof(double));
    infile.close();
}

// --- FMS Estimation Table Logic (moved from header) ---

double fms_expected_z(unsigned m, unsigned w, double cardinality) {
    double expected_z = m * w;
    double quotient = 1.0;
    for (unsigned j = 0; j < w - 1; ++j) {
        quotient *= 2.0;
        expected_z -= (double)m * pow(1.0 - 1.0 / ((double)m * quotient), cardinality);
    }
    expected_z -= (double)m * pow(1.0 - 1.0 / ((double)m * quotient), cardinality);
    return expected_z;
}

double fms_estimate(unsigned m, unsigned w, double z) {
    double left_n = 0.0, right_n = pow(10.0, 11.2);
    if (z < 0.0) return 0.0;
    if (z > m * w) return right_n;
    
    for(int i=0; i<100; ++i) { // Use fixed iterations for binary search
        double mid = left_n + (right_n - left_n) / 2.0;
        if (fms_expected_z(m, w, mid) >= z) { right_n = mid; }
        else { left_n = mid; }
    }
    return (left_n + right_n) / 2.0;
}

std::vector<double> gene_fms_estimation_table() {
    unsigned fms_m = number_of_bits / 32;
    unsigned fms_w = 32;
    std::vector<double> estimation_table(fms_m * fms_w);

    string file_name = "datastreams/" + to_string(fms_m) + "_" + to_string(fms_w) + ".bin";
    
    if (filesystem::exists(file_name)) {
        load_double_from_file(file_name, estimation_table);
        return estimation_table;
    }

    unsigned long long max_cardi = pow(10, 11);
    unsigned i;
    double temp = 0;
    for (i = 1; i <= fms_m * fms_w; ++i) {
        temp = fms_estimate(fms_m, fms_w, (double)i);
        estimation_table[i - 1] = temp;
        if (temp > max_cardi) break;
    }

    for (unsigned j = i; j < fms_m * fms_w; ++j) {
        estimation_table[j] = temp;
    }

    save_double_to_file(file_name, estimation_table);
    return estimation_table;
}

double fms_table_estimate(int count, unsigned long long t, const std::vector<double>& estimation_table) {
    if (count <= 0) return 1.0;
    if ((unsigned)count <= number_of_bits) {
        return min(estimation_table[count - 1], (double)t);
    }
    return (double)t;
}