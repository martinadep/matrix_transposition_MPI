#include "main.h"
#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    srand(time(NULL));
    if (argc != 2) {
        printf("Usage: ./main <matrix_size>\n");
        return 1;
    }
    int n = atoi(argv[1]);
    int mat_size = pow(2, n);

    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    //int mat_size = 4096; // Size of the square matrix (N x N)
    if (nprocs > mat_size) {
        if (rank == 0) {
            printf("Processes must be less then matrix size");
        }
        MPI_Finalize();
        exit(1);
    }

    /*
     * NOTE: Print info about processors
     *
     * char processor_name[MPI_MAX_PROCESSOR_NAME];
     * int name_len;
     * MPI_Get_processor_name(processor_name, &name_len);
     * printf("Processor name: %s\n", processor_name);
    */

    /*
     * NOTE: Results of the matrix transpositions can be printed:
     * note that for huge matrix it could be convenient to
     * print only a small block, like 5x5, to assess correctness
     *
     * int elem_to_print = 5;
     * print_matrix(M, elem_to_print);
    */

    // --- Matrix allocations and initialization
    float *M = NULL;
    float *T = NULL;
    if (rank == 0) {
        M = allocate_sqr_matrix(mat_size);
        T = allocate_sqr_matrix(mat_size);
        init_matrix(M, mat_size);
        //printf("Original matrix %d x %d:\n", N, N);
        //print_matrix(M, N, N);
    }



    if (rank == 0) {
        double start = MPI_Wtime();
        transpose_local(M, T, mat_size, mat_size);
        double end = MPI_Wtime();
        //printf("\nTransposed matrix %d x %d sequential:\n", N, N);
        //print_matrix(Tseq, N, N);
        printf("Elapsed time sequential: %f s\n", end - start);
    }

    double start, end;
    if (rank == 0) {
        start = MPI_Wtime();
    }
    matTransposeMPI(M, T, mat_size, rank, nprocs);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        end = MPI_Wtime();
        printf("Elapsed time MPI: %f s\n", end - start);
    }


    if (rank == 0) {
        start = MPI_Wtime();
    }
    matTransposeMPI_Bcast(M, T, mat_size, rank, nprocs);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        end = MPI_Wtime();
        printf("Elapsed time MPI Bcast: %f s\n", end - start);
    }

    // Root process free allocated matrices
    if (rank == 0) {
        //printf("\nTransposed matrix %d x %d with MPI:\n", N, N);
        //print_matrix(Tmpi, N, N);
        free(M);
        free(T);
    }

    MPI_Finalize();
    return 0;
}
