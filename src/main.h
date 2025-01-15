#ifndef __MAIN_H_
#define __MAIN_H_

// N is power of 2, size of squared matrix
// evaluate from N=16 to N=4096
#define MIN 16
#define MAX 4096
#define LOOP 10

// MAIN.C
void matTranspose(float* M, float* T, int size);
int checkSym(float* M, int size);
void matTransposeMPI(float *M, float *T, int matSize, int rank, int num_procs);
void rankTransposeMPI(float *M, float *T, int matSize, int rank, int num_procs);

// UTILS.C
float* allocate_sqr_matrix(int size);
void init_matrix(float* matrix, int size);
void print_matrix_utils(float *M, int rows, int cols);

int remove_outliers(float data[], float filtered_data[], int data_size, float threshold);
float calculate_std_dev(float arr[], int array_size, float mean);
float calculate_mean(float arr[], int array_size);

// TEST_FUNCTIONS.C
void test_functions();
#endif
