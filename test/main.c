#include <stdio.h>
#include <stdbool.h>
#include "../src/main.h"

#include <mpi.h>
#include <stdlib.h>
#include <time.h>

/// This is a function used to test correctness of the checkSym()s and matTransp()s
/// It performs tests on two [8x8] matrices, one symmetric and one asymmetric
/// which is transposed
void main(int argc, char *argv[]) {
    srand(time(NULL));
    int mat_size = 8;

    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    if (nprocs > mat_size) {
        if (rank == 0) {
            printf("Processes must be less then matrix size");
        }
        MPI_Finalize();
        exit(1);
    }

    float *M = NULL;
    float *T = NULL;
    float *M_symmetric = NULL;

    // --- matrix transpositions ---
    if (rank == 0) {
        M = allocate_sqr_matrix(mat_size);
        T = allocate_sqr_matrix(mat_size);
        M_symmetric = allocate_sqr_matrix(mat_size);

        init_matrix(M, mat_size);
        printf("Matrix before transposition:\n");
        print_matrix(M, mat_size, mat_size);
        printf("\n");

        if (checkSym(M, mat_size)) {
            printf("Matrix is symmetric [check_Sym], no need to transpose\n");
        } else {
            printf("Matrix is NOT symmetric [check_Sym], transposing seq\n");
            matTranspose(M, T, mat_size);
            print_matrix(T, mat_size, mat_size);
        }
        printf("\n");

        if (checkSymOMP(M, mat_size)) {
            printf("Matrix is symmetric [check_SymOMP], no need to transpose\n");
        } else {
            printf("Matrix is NOT symmetric [check_SymOMP], transposing OMP\n");
            matTransposeOMP(M, T, mat_size);
            print_matrix(T, mat_size, mat_size);
        }
    }

    int checkMPI = checkSymMPI(M, mat_size, rank, nprocs);
    if (!checkMPI) {
        if (rank == 0) {
            printf("\nMatrix is NOT symmetric [check_SymMPI], transposing MPI\n");
        }
        matTransposeMPI(M, T, mat_size, rank, nprocs);
        if (rank == 0) {
            printf("MPI scatter - gather:\n");
            print_matrix(T, mat_size, mat_size);
        }
        matTransposeMPI_Bcast(M, T, mat_size, rank, nprocs);
        if (rank == 0) {
            printf("MPI broadcast:\n");
            print_matrix(T, mat_size, mat_size);
        }
    }

    // --- check symmetry---
    if (rank == 0) {
        printf("\n-----------------------------\nM_symmetric:\n");
        for (int i = 0; i < mat_size; i++) {
            for (int j = 0; j < mat_size; j++) {
                if (i == j - 1 || i - 1 == j) {
                    M_symmetric[i * mat_size + j] = 2.3;
                } else M_symmetric[i * mat_size + j] = 1.0;
            }
        }

        print_matrix(M_symmetric, mat_size, mat_size);
        if (checkSym(M_symmetric, mat_size)) printf("M_symmetric is symmetric [checkSym]\n");
        else printf("M_symmetric is NOT symmetric [checkSym]\n");

        if (checkSymOMP(M_symmetric, mat_size)) printf("M_symmetric is symmetric [check_SymOMP]\n");
        else printf("M_symmetric is NOT symmetric [checkOMP]\n");
    }

    checkMPI = checkSymMPI(M_symmetric, mat_size, rank, nprocs);
    if (checkMPI && rank == 0) printf(
        "M_symmetric is symmetric [check_SymMPI]\n");
    else if (rank == 0) printf("M_symmetric is NOT symmetric [checkMPI]\n");

    if (rank == 0) {
        free(M);
        free(T);
        free(M_symmetric);
    }
    MPI_Finalize();
}
