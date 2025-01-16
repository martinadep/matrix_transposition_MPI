#ifndef __MAIN_H_
#define __MAIN_H_

// N is power of 2, size of squared matrix
// evaluate from N=16 to N=4096
#define MIN 16
#define MAX 4096
#define LOOP 10
#include <stdbool.h>

// CHECKSYM_FUNCTIONS.C
int checkSym(float* M, int mat_size);
int checkSymOMP(float* M, int mat_size);
int checkSymMPI(float* M, int mat_size, int rank, int num_procs);
int check_sym_local(float *matrix, int mat_size, int start_row, int end_row);

// TRANSP_FUNCTIONS.C
void transpose_local(float* local_matrix, float* local_transposed, int rows, int cols);
void matTranspose(float* M, float* T, int mat_size);
void matTransposeOMP(float* M, float* T, int mat_size);
void matTransposeMPI(float* M, float* T, int mat_size, int rank, int num_procs);
void matTransposeMPI_Bcast(float* M, float* T, int mat_size, int rank, int num_procs);

// UTILS.C
float* allocate_sqr_matrix(int size);
void init_matrix(float* matrix, int size);
void print_matrix(float* matrix, int rows, int cols);

int remove_outliers(float data[], float filtered_data[], int data_size, float threshold);
float calculate_std_dev(float arr[], int array_size, float mean);
float calculate_mean(float arr[], int array_size);

int choose_omp_threads(int matrix_size);
int choose_block_size(int matrix_size);

#endif
