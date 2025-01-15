#include "main.h"

#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void print_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%4.0f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

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

    int N = 8; // Size of the square matrix (N x N)
    float *matrix = NULL;
    float *transposed = NULL;

    // Process 0 initializes the matrix
    if (rank == 0) {
        matrix = (float *)malloc(N * N * sizeof(float));
        transposed = (float *)malloc(N * N * sizeof(float));

        // Fill the matrix with some values
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i * N + j] = i * 10 + j;
            }
        }

        printf("Original matrix:\n");
        print_matrix(matrix, N, N);
    }

    // Scatter the columns of the matrix to all processes
    int col_per_proc = N / nprocs;
    printf("col_per_proc: %d\n", col_per_proc);

    float *local_matrix = (float *)malloc(col_per_proc * N * sizeof(float));

    // count: num of blocks
    // block length: num of elements in each block
    // stride: elements between start of block N and start of block N+1
    MPI_Datatype columns_type, res_column_type;
    MPI_Type_vector( N, // count of blocks is row number
                    col_per_proc, // block length is column for each process
                     N, // stride num process * column = N for each process
                    MPI_FLOAT, &columns_type);
    //MPI_Type_create_resized(MPI_FLOAT, 0,    1*sizeof(float), // extend: length of a single block, so scatter knows where the next one starts&columns_type);
    MPI_Type_create_resized(columns_type, 0, col_per_proc * sizeof(float), &res_column_type);
    MPI_Type_commit(&res_column_type);
    //MPI_Type_commit(&columns_type);


   /* MPI_Scatter(matrix, 1, columns_type,
                local_matrix, col_per_proc * N, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    printf("Process %d received in local_matrix aftert SCATTER:\n", rank);
    print_matrix(local_matrix, N, col_per_proc);
   */
    int counts[nprocs], displs[nprocs];
    if (rank == 0) {
        for (int i = 0; i < nprocs; i++) {
            counts[i] = 1;
            displs[i] = i ;
        }
    }
    MPI_Scatterv(matrix, counts, displs,
        res_column_type, local_matrix,
        col_per_proc * N, MPI_FLOAT,
        0, MPI_COMM_WORLD);


    printf("Process %d received in local_matrix after SCATTERV:\n", rank);
    print_matrix(local_matrix, N, col_per_proc);
/*
    // Allocate space for the local transposed matrix
    float *local_transposed = (float *)malloc(rows_per_proc * N * sizeof(float));

    // Transpose locally (swap rows and columns for each chunk)
    transpose_local(local_matrix, local_transposed, rows_per_proc, N);

    // Gather the transposed chunks into the final matrix
    if (rank == 0) {
        float *temp_buffer = (float *)malloc(N * N * sizeof(float));
        MPI_Gather(local_transposed, rows_per_proc * N, MPI_FLOAT,
                   temp_buffer, rows_per_proc * N, MPI_FLOAT,
                   0, MPI_COMM_WORLD);

        // Reassemble the final transposed matrix
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                transposed[j * N + i] = temp_buffer[i * N + j];
            }
        }

        free(temp_buffer);
    } else {
        MPI_Gather(local_transposed, rows_per_proc * N, MPI_FLOAT,
                   NULL, rows_per_proc * N, MPI_FLOAT,
                   0, MPI_COMM_WORLD);
    }
*/
    // Process 0 prints the transposed matrix
    if (rank == 0) {
        // printf("\nTransposed matrix:\n");
       // print_matrix(transposed, N, N);
        free(matrix);
        free(transposed);
    }

    free(local_matrix);
    //free(local_transposed);

    MPI_Type_free(&res_column_type);
    MPI_Finalize();
    return 0;
}

/*
int main(int argc, char *argv[]) {
    int size = 8; // matrice 8x8
    float *M = allocate_sqr_matrix(size);
    int nprocs, rank; // 4 processi
    int i, j;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Matrix initialization
    if (rank == 0) {
        printf("running with %d processes", nprocs);
        for (i = 0; i < size; i++) {
            for (j = 0; j < size; j++) {
                M[i * size + j] = i * 10 + j;
            }
        }
        printf("Matrix M:\n");
        print_matrix(M, size, size);
    }


    int rows = size / nprocs;
    int columns = size;
    int elements_in_local_M = rows * size;
    if (rank == 0) {
        printf("Number of processes: %d\n", nprocs);
        printf("Matrix size: %d\n", size);
        printf("Rows per process: %d\n", rows);
        printf("Elements in each local_M: %d\n", elements_in_local_M);
    }

    // --- SCATTER ---
    float *local_M = allocate_sqr_matrix(elements_in_local_M);
    MPI_Scatter(M, elements_in_local_M, MPI_FLOAT, local_M, elements_in_local_M, MPI_FLOAT, 0, MPI_COMM_WORLD);
    printf("\nprocess %d received from root local_M :\n ", rank);
    print_matrix(local_M, rows, columns);

    // --- TYPE VECTOR ---
    int elem_in_block = rows * rows;
    int block_size = rows;
    MPI_Datatype BlockType;
    MPI_Type_vector(rows * rows, block_size, //count: num of blocks, blocklen: num of elements
        rows * rank, MPI_FLOAT, &BlockType);
    MPI_Type_commit(&BlockType);

    // --- SCATTER WITH DERIVED DATATYPE ---
    float *local_M_derived = allocate_sqr_matrix(elements_in_local_M);
    MPI_Scatter(M, nprocs, BlockType,
                local_M_derived, elem_in_block * nprocs,
         MPI_FLOAT, 0, MPI_COMM_WORLD);

    printf("\nprocess %d received from root local_M DERIVED :\n ", rank);
    print_matrix(local_M, rows, columns);


    float *received_M = allocate_sqr_matrix(elem_in_block);
    if (rank == 0) {
        MPI_Send(local_M, 1, BlockType, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(received_M, elem_in_block, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank == 1) {
        MPI_Send(local_M, 1, BlockType, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(received_M, elem_in_block, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    printf("\nprocess %d received received_M:\n", rank);
    print_matrix(received_M, block_size, block_size);

    if (rank == 0) free(M);
    MPI_Finalize();
    return 0;
}
*/
