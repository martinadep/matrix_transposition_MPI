#include <stdio.h>
#include <omp.h>
#include <mpi.h>
#include "main.h"

#include <stdlib.h>
#include <tgmath.h>
#include <time.h>

/// This main performs three matrix transposition and check symmetry approaches:
/// - Sequential
/// - Implicitly parallelized
/// - Explicitly parallelized
///
/// It takes <pow> as input and operates over [2^pow]x[2^pow] matrices
int main(int argc, char **argv) {
    srand(time(NULL));
    if (argc != 2) {
        printf("Usage: ./main <matrix_size>\n");
        return 1;
    }
    int n = atoi(argv[1]);

    int my_rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    printf("%d out of %d procs\n", my_rank, num_procs);


    int matrix_size = pow(2, n);
    printf("matrix [%d]x[%d]...\n", matrix_size, matrix_size);
    float **M = allocate_sqr_matrix(matrix_size);
    float **T = allocate_sqr_matrix(matrix_size);
    init_matrix(M, matrix_size);
    /*
     * Results of the matrix transpositions can be printed:
     * note that for huge matrix it could be convenient to
     * print only a small block, like 5x5, to assess correctness
    */
    //int elem_to_print = 5;
    //print_matrix(M, elem_to_print);

    //------------------------sequential--------------------------
    if (checkSym(M, n)) {
        printf("Matrix is symmetric (seq), no need to transpose\n");
    } else {
        double wt1 = omp_get_wtime();
        matTranspose(M, T, n);
        double wt2 = omp_get_wtime();
        double elapsed = wt2 - wt1;
        printf("mat transpose: %f\n", elapsed);
        //print_matrix(T, elem_to_print);
    }

    //--------------------------OMP-------------------------------
    if (checkSymOMP(M, n)) {
        printf("Matrix is symmetric (omp), no need to transpose\n");
    } else {
        double wt1 = omp_get_wtime();
        matTransposeOMP(M, T, n);
        double wt2 = omp_get_wtime();
        double elapsed = wt2 - wt1;
        printf("mat transpose OMP: %f\n", elapsed);
        //print_matrix(T, elem_to_print);
    }

    /*
     * The correctness of the functions can be checked through
     * the "test_functions()": it exectutes all the functions
     * implemented in main.c, over two [4x4] matrices, one is
     * symmetric and one is asymmetric.
    */
   // test_functions();

    MPI_Finalize();
    return 0;

    return 0;
}


/// Naive matrix transposition
void matTranspose(float **M, float **T, int n) {
    int size = pow(2, n);
#pragma omp parallel num_threads(1)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            T[j][i] = M[i][j];
        }
    }
}

/// Naive check symmetry
int checkSym(float **M, int n) {
    int size = pow(2, n);
    int is_sym = 1; // assumed symmetric
#pragma omp parallel num_threads(1)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (M[i][j] != M[j][i]) {
                is_sym = 0; // non-symmetric
            }
        }
    }
    return is_sym;
}

/// Parallel OMP matrix transposition
void matTransposeOMP(float **M, float **T, int n) {
    int size = pow(2, n);
    omp_set_num_threads(4);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            T[j][i] = M[i][j];
        }
    }
}

/// Parallel OMP check symmetry
int checkSymOMP(float **M, int n) {
    int size = pow(2, n);
    int is_sym = 1; // assumed symmetric
#pragma omp parallel for collapse(2) reduction(&&:is_sym)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (M[i][j] != M[j][i]) {
                is_sym = 0; // non-symmetric
            }
        }
    }
    return is_sym;
}

