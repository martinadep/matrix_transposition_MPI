#include "main.h"
#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <omp.h>


void transpose_local(float *local_matrix, float *local_transposed, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            local_transposed[j * rows + i] = local_matrix[i * cols + j];
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int N = 4096; // Size of the square matrix (N x N)
    if (nprocs > N && rank == 0) {
        MPI_Finalize();
        printf("Processes must be less then matrix size");
        exit(1);
    }

    float *M = NULL;
    float *Tseq = NULL;
    float *Tmpi = NULL;

    // Process 0 initializes the matrix
    if (rank == 0) {
        M = (float *)malloc(N * N * sizeof(float));
        Tseq = (float *)malloc(N * N * sizeof(float));
        Tmpi = (float *)malloc(N * N * sizeof(float));

        // Fill the matrix with some values
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                M[i * N + j] = i * 10 + j; // change this with random values
            }
        }

        //printf("Original matrix %d x %d:\n", N, N);
        //print_matrix(M, N, N);
    }

    if (rank == 0) {
        double start = MPI_Wtime();
        transpose_local(M, Tseq, N, N);
        double end = MPI_Wtime();
        //printf("\nTransposed matrix %d x %d sequential:\n", N, N);
        //print_matrix(Tseq, N, N);
        printf("Elapsed time sequential: %f s\n", end - start);
    }

    double start, end;
    if (rank == 0) {
       start = MPI_Wtime();
    }

    matTransposeMPI(M, Tmpi, N, rank, nprocs);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        end = MPI_Wtime();
        printf("Elapsed time MPI: %f s\n", end - start);
    }


    // Process 0 prints the transposed matrix
    if (rank == 0) {
        //printf("\nTransposed matrix %d x %d with MPI:\n", N, N);
        //print_matrix(Tmpi, N, N);
        free(M);
        free(Tseq);
        free(Tmpi);
    }

    MPI_Finalize();
    return 0;
}

void matTransposeMPI(float *M, float *T, int mat_size, int rank, int num_procs) {
    int col_per_proc = mat_size / num_procs;

    // ------ Create datatype to scatter by columns ------
    MPI_Datatype columns_type, res_columns_type;
    MPI_Type_vector( mat_size,      // count: num of blocks of contiguous elements (one for each row)
                     col_per_proc,  // block length: num of elements in each block (columns assigned to each proc)
                     mat_size,      // stride: elements between start of block N and start of block N+1 (matrix width)
                     MPI_FLOAT, &columns_type);
    MPI_Type_create_resized(columns_type, 0,
        col_per_proc * sizeof(float), // extend: bytes for each block (elements in block * size of each element)
        &res_columns_type); 
    MPI_Type_commit(&res_columns_type);

    // ------ Set displacement and count of res_columns_type for each proc ------
    int counts[num_procs], disp[num_procs];
    if (rank == 0) {
        for (int i = 0; i < num_procs; i++) {
            counts[i] = 1;
            disp[i] = i ;
        }
    }

    // ------ Scatter the columns of the matrix to all processes ------
    float *local_M = malloc(col_per_proc * mat_size * sizeof(float));
    MPI_Scatterv(M, counts, disp,res_columns_type,
        local_M,col_per_proc * mat_size, MPI_FLOAT,
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