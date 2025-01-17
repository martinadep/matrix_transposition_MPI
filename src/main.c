#include "main.h"
#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>

/// This main performs two matrix transposition and check symmetry approaches:
/// - Sequential
/// - MPI parallelized
///
/// It takes <pow> as input and operates over [2^pow]x[2^pow] matrices

int main(int argc, char *argv[]) {
    srand(time(NULL));
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 2) {
      if (rank==0){
          printf("Usage: ./main <matrix_size>\n");
      }
      MPI_Finalize();
      return 1;
    }
    int n = atoi(argv[1]);
    int mat_size = pow(2, n);

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
     * print only a small block, like 5x5.
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

        //float checksum = partialChecksum(M, mat_size);
        //printf("checksum is %f\n", checksum);
        //printf("Original matrix %d x %d:\n", N, N);
        //print_matrix(M, N, N);
    }

    if (rank == 0) {
        double start, end;
        //------------------------ sequential --------------------------
        if (checkSym(M, mat_size)) {
            printf("Matrix is symmetric (seq), no need to transpose\n");
        } else {
            start = MPI_Wtime();
            transpose_local(M, T, mat_size, mat_size);
            end = MPI_Wtime();
            printf("%f s | Elapsed time transpose sequential\n", end - start);
            //print_matrix(T, 5, 5);

            //float checksum = partialChecksum(M, mat_size);
            //printf("checksum is %f\n", checksum);
        }

    }

    if (checkSymMPI(M, mat_size, rank, nprocs)) {
        if (rank == 0) {
            printf("Matrix is symmetric (MPI), no need to transpose\n");
        }
    } else {
        //-------------------------- MPI Scatterv -----------------------------
        double start, end;
        if (rank == 0) {
            start = MPI_Wtime();
        }
        matTransposeMPI(M, T, mat_size, rank, nprocs);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            end = MPI_Wtime();
            printf("%f s | Elapsed time transpose MPI (%d procs)\n", end - start, nprocs);
            //print_matrix(T, 5, 5);

            //float checksum = partialChecksum(M, mat_size);
            //printf("checksum is %f\n", checksum);
        }


        // -------------------------- MPI Bcast -------------------------------
        if (rank == 0) {
            start = MPI_Wtime();
        }
        matTransposeMPI_Bcast(M, T, mat_size, rank, nprocs);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            end = MPI_Wtime();
            printf("%f s | Elapsed time transpose MPI Bcast (%d procs)\n", end - start, nprocs);

            //float checksum = partialChecksum(M, mat_size);
            //printf("checksum is %f\n", checksum);
            //print_matrix(T, 5, 5);
        }
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
