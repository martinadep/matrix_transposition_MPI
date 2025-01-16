#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "main.h"

/// Naive matrix transposition
void matTranspose(float *M, float *T, int mat_size) {
#pragma omp parallel num_threads(1)
    for (int i = 0; i < mat_size; i++) {
        for (int j = 0; j < mat_size; j++) {
            T[j * mat_size + i] = M[i * mat_size + j];
        }
    }
}

/// Parallel OMP matrix transposition
void matTransposeOMP(float *M, float *T, int mat_size) {
    int block_size = 16; //choose_block_size(size);

    /*
     * If you want to use a different number of threads
     * using export OMP_NUM_THREADS
     * you must comment the next two lines of code,
     * otherwise 'export' it is overwritten
    */
    int num_thr = 4; //choose_num_threads(size);
    //    omp_set_num_threads(num_thr);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < mat_size; i += block_size) {
        for (int j = 0; j < mat_size; j += block_size) {
            for (int bi = i; bi < i + block_size && bi < mat_size; bi += 2) {
                for (int bj = j; bj < j + block_size && bj < mat_size; bj += 2) {
                    T[bj * mat_size + bi] = M[bi * mat_size + bj];
                    T[bj * mat_size + bi + 1] = M[(bi + 1) * mat_size + bj];
                    T[(bj + 1) * mat_size + bi] = M[bi * mat_size + bj + 1];
                    T[(bj + 1) * mat_size + bi + 1] = M[(bi + 1) * mat_size + bj + 1];
                }
            }
        }
    }
}

void transpose_local(float *local_matrix, float *local_transposed, int rows, int cols) {
#pragma omp parallel num_threads(1)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            local_transposed[j * rows + i] = local_matrix[i * cols + j];
        }
    }
}

/// Parallel MPI matrix transposition using Scattering - Gathering with Datatypes
void matTransposeMPI(float *M, float *T, int mat_size, int rank, int num_procs) {
    int col_per_proc = mat_size / num_procs;

    // ------ Create datatype to scatter by columns ------
    MPI_Datatype columns_type, res_columns_type;
    MPI_Type_vector(mat_size, // count: num of blocks of contiguous elements (one for each row)
                    col_per_proc, // block length: num of elements in each block (columns assigned to each proc)
                    mat_size, // stride: elements between start of block N and start of block N+1 (matrix width)
                    MPI_FLOAT, &columns_type);
    MPI_Type_create_resized(columns_type, 0,
                            col_per_proc * sizeof(float),
                            // extend: bytes for each block (elements in block * size of each element)
                            &res_columns_type);
    MPI_Type_commit(&res_columns_type);

    // ------ Set displacement and count of res_columns_type for each proc ------
    int counts[num_procs], disp[num_procs];
    if (rank == 0) {
        for (int i = 0; i < num_procs; i++) {
            counts[i] = 1;
            disp[i] = i;
        }
    }

    // ------ Scatter the columns of the matrix to all processes ------
    float *local_M = malloc(col_per_proc * mat_size * sizeof(float));
    MPI_Scatterv(M, counts, disp, res_columns_type,
                 local_M, col_per_proc * mat_size, MPI_FLOAT,
                 0, MPI_COMM_WORLD);
    //printf("Process %d received in local_matrix after SCATTERV:\n", rank);
    //print_matrix(local_M, N, col_per_proc);

    // ------ Compute local transposition ------
    float *local_T = malloc(mat_size * col_per_proc * sizeof(float));
    transpose_local(local_M, local_T, mat_size, col_per_proc);
    //printf("Process %d local_T after transpose:\n", rank);
    //print_matrix(local_T, col_per_proc, N);

    // ------ Gather the transposed matrix ------
    MPI_Gather(local_T, mat_size * col_per_proc, MPI_FLOAT,
               T, mat_size * col_per_proc, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    // ------ Free datatype and local matrices ------
    free(local_M);
    free(local_T);
    MPI_Type_free(&res_columns_type);
}


/// Parallel MPI matrix transposition using Bcast
void matTransposeMPI_Bcast(float *M, float *T, int mat_size, int rank, int nproc){
    int rows_per_proc = mat_size / nproc;
    int start = rank * rows_per_proc;
    int end   = start + rows_per_proc;

    // --- Broadcast matrix to all processes ---
    if (rank != 0) {
        // Every process must allocate space for matrix M
        M = allocate_sqr_matrix(mat_size);
    }
    MPI_Bcast(M, mat_size*mat_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // --- Transpose locally ---
    float *local_T = malloc(rows_per_proc * mat_size * sizeof(float));
    for (int i = start; i < end; i++) {
        for (int j = 0; j < mat_size; j++) {
            local_T[(i - start) * mat_size + j] = M[j * mat_size + i];
        }
    }

    // --- Gather all local transposition ---
    MPI_Gather( local_T,  rows_per_proc * mat_size, MPI_FLOAT, //send
        T, rows_per_proc * mat_size, MPI_FLOAT, //recv
        0, MPI_COMM_WORLD);

    // --- Clean up ---
    free(local_T);
    if (rank != 0) {
        free(M);
    }
}
