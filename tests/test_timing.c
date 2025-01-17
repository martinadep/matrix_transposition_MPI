#include "test_timing.h"

#include <mpi.h>

#include "../src/main.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LOOP 10
#define MIN 16
#define MAX 4096
/// Execute this main with different <num_threads>.
/// It computes matrix transposition with two different MPI approaches
///
/// - MPI using MPI_Bcast()
///
/// - MPI using MPI_Scatterv()
///
/// You can store the result in a .csv file with the following format:
///
/// "matrix_size, mean, num_procs, approach"
int main(int argc, char **argv) {
    srand(time(NULL));

    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    float filtered_data_MpiScatt[LOOP];
    float data_MpiScatt[LOOP];
    float thrsd = 2.0;

    // ----- matTransposeMPI Scatter -----
    for (int matrix_size = MIN; matrix_size <= MAX; matrix_size *= 2) {
        if (matrix_size > nprocs) {
            FILE *file = NULL;
            // root stores values
            if (rank == 0) {
                // File settings to store results
                char filename[20];
                sprintf(filename, "./timing_out/%dmatrix.txt", matrix_size);
                file = fopen(filename, "a");
                if (file == NULL) {
                    perror("Error opening file");
                    return 1;
                }
            }
            float *M = NULL;
            float *T = NULL;
            if (rank == 0) {
                M = allocate_sqr_matrix(matrix_size);
                T = allocate_sqr_matrix(matrix_size);
                init_matrix(M, matrix_size);
            }

            double start = 0, end = 0;
            for (int i = 0; i < LOOP; i++) {
                // root stores timing for matTransposeMPI with Scatter
                if (rank == 0) { start = MPI_Wtime(); }

                matTransposeMPI(M, T, matrix_size, rank, nprocs);

                MPI_Barrier(MPI_COMM_WORLD);
                if (rank == 0) {
                    end = MPI_Wtime(); // Stop wall-clock time
                    fprintf(file, "(%d) %fs | %d procs | matTransposeMPI()\n",i, end - start, nprocs);
                    data_MpiScatt[i] = end - start;
                }
            }

            if (rank == 0) {
                // root computes data filtering and storing
                // MPI Scatter
                int count_filtered_MpiScatt = remove_outliers(data_MpiScatt, filtered_data_MpiScatt, LOOP, thrsd);
                if (count_filtered_MpiScatt > 0) {
                    double filtered_data_mean = calculate_mean(filtered_data_MpiScatt, count_filtered_MpiScatt);
                    // READABLE FORMAT
                    //printf("[%dx%d] mean for %d threads: %.7f (naive)\n", matrix_size, matrix_size, num_threads, filtered_data_mean);

                    // .CSV FORMAT
                    printf("%d,%.7f,%d,Scatter\n", matrix_size, filtered_data_mean, nprocs);
                } else {
                    printf("All values considered outliers in a row\n");
                }
            }
            if (rank == 0) {
                fclose(file);
                free(M);
                free(T);
            }
        }
    }

    if(rank == 0) printf("\n");

    float filtered_data_MpiBcast[LOOP];
    float data_MpiBcast[LOOP];
    // ----- matTransposeMPI Bcast -----
    for (int matrix_size = MIN; matrix_size <= MAX; matrix_size *= 2) {
        if (matrix_size > nprocs) {
            FILE *file = NULL;
            // root stores values
            if (rank == 0) {
                // File settings to store results
                char filename[20];
                sprintf(filename, "./timing_out/%dmatrix.txt", matrix_size);
                file = fopen(filename, "a");
                if (file == NULL) {
                    perror("Error opening file");
                    return 1;
                }
            }
            float *M = NULL;
            float *T = NULL;
            if (rank == 0) {
                M = allocate_sqr_matrix(matrix_size);
                T = allocate_sqr_matrix(matrix_size);
                init_matrix(M, matrix_size);
            }

            // ----- matTransposeMPI -----
            double start = 0, end = 0;
            for (int i = 0; i < LOOP; i++) {
                // root stores timing for matTransposeMPI with Bcast
                if (rank == 0) { start = MPI_Wtime(); }

                matTransposeMPI_Bcast(M, T, matrix_size, rank, nprocs);

                MPI_Barrier(MPI_COMM_WORLD);
                if (rank == 0) {
                    end = MPI_Wtime(); // Stop wall-clock time
                    fprintf(file, "(%d) %fs | %d procs | matTransposeMPI_Bcast()\n",i, end - start, nprocs);
                    data_MpiBcast[i] = end - start;
                }
            }

            if (rank == 0) {
                // root computes data filtering and storing
                // MPI Bcast
                int count_filtered_MpiBcast = remove_outliers(data_MpiBcast, filtered_data_MpiBcast, LOOP, thrsd);
                if (count_filtered_MpiBcast > 0) {
                    double filtered_data_mean = calculate_mean(filtered_data_MpiBcast, count_filtered_MpiBcast);
                    // READABLE FORMAT
                    //printf("[%dx%d] mean for %d threads: %.7f (block-based)\n", matrix_size, matrix_size, num_threads, filtered_data_mean);

                    // .CSV FORMAT
                    printf("%d,%.7f,%d,Bcast\n", matrix_size, filtered_data_mean, nprocs);
                } else {
                    printf("All values considered outliers in a row\n");
                }
            }

            if (rank == 0) {
                fclose(file);
                free(M);
                free(T);
            }
        }
    }
    if(rank == 0) printf("\n");
    return 0;
}
