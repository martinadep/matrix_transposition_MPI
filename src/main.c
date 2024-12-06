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

    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    printf("%d out of %d procs\n", rank, num_procs);


    int size = pow(2, n);
    printf("matrix [%d]x[%d]...\n", size, size);

    float *M = NULL;
    float *T = NULL;
    if (rank == 0) {
        M = allocate_sqr_matrix(size);
        T = allocate_sqr_matrix(size);
        init_matrix(M, size);
        print_top_left_block(M, size, size);
        printf("\n");
    }

    //------------------------sequential--------------------------
    double wt1, wt2, elapsed;
    if (rank == 0) {
        if (checkSym(M, size)) {
            printf("Matrix is symmetric according to chechSym() - no need to transpose!\n");
        } else {
            wt1 = MPI_Wtime();
            matTranspose(M, T, size);
            wt2 = MPI_Wtime();
            elapsed = wt2 - wt1;
            printf("mat transpose: %f\n", elapsed);
            print_top_left_block(T, size, size);
        }

    }
    //------------------------parallel-MPI--------------------------
    wt1 = MPI_Wtime();
    matTransposeMPI(M, T, size, rank, num_procs);
    wt2 = MPI_Wtime();
    elapsed = wt2 - wt1;
    if (rank == 0) {
        printf("\nmat transposeMPI: %f\n", elapsed);
        print_top_left_block(T, size, size);
        free(M);
        free(T);
    }


    MPI_Finalize();
    return 0;
}


/// Naive matrix transposition
void matTranspose(float *M, float *T, int size) {
    int rows = size, cols = size;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            T[j * rows + i] = M[i * cols + j];
        }
    }
}

/// Naive check symmetry
int checkSym(float *M, int size) {
    int is_sym = 1; // assumed symmetric
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (M[j * size + i] != M[i * size + j]) {
                is_sym = 0; // non-symmetric
            }
        }
    }
    return is_sym;
}
void matTransposeMPI(float *M, float *T, int matSize, int rank, int size) {
    int rowsPerProcess = matSize / size;
    int extraRows = matSize % size;

    int startRow = rank * rowsPerProcess;
    int endRow = startRow + rowsPerProcess;
    if (rank == size - 1) {
        endRow += extraRows;
    }

    int localRows = endRow - startRow;

    // Allocate local buffers
    float *localM = (float *)malloc(localRows * matSize * sizeof(float));
    float *localT = (float *)malloc(matSize * matSize * sizeof(float));
    if (!localM || !localT) {
        fprintf(stderr, "Memory allocation failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int *sendCounts = NULL;
    int *displacements = NULL;

    if (rank == 0) {
        sendCounts = (int *)malloc(size * sizeof(int));
        displacements = (int *)malloc(size * sizeof(int));

        for (int i = 0; i < size; i++) {
            int currentRows = rowsPerProcess + (i == size - 1 ? extraRows : 0);
            sendCounts[i] = currentRows * matSize;
            displacements[i] = i * rowsPerProcess * matSize;
        }
    }

    // Scatter rows of M
    MPI_Scatterv(M, sendCounts, displacements, MPI_FLOAT,
                 localM, localRows * matSize, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    // Transpose the local block
    for (int i = 0; i < localRows; i++) {
        for (int j = 0; j < matSize; j++) {
            localT[j * matSize + (startRow + i)] = localM[i * matSize + j];
        }
    }

    // Gather transposed blocks into T
    MPI_Gatherv(localT, localRows * matSize, MPI_FLOAT,
                T, sendCounts, displacements, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // Cleanup
    free(localM);
    free(localT);
    if (rank == 0) {
        free(sendCounts);
        free(displacements);
    }
}

