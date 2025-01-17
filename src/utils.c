#include <stdio.h>
#include "main.h"
#include <math.h>
#include <stdlib.h>

#define ELEM_TO_PRINT 8
/// Dynamic allocation of a square [size] x [size] matrix
float *allocate_sqr_matrix(int size) {
    float *matrix = malloc(size * size * sizeof(float));
    if (matrix == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(1);
    }
    return matrix;
}

/// Random floats initialization of a square [size] x [size] matrix
void init_matrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i * size + j] = (float) rand() / (float) RAND_MAX;
        }
    }
}
/// Function that prints a square [size] x [size] matrix
void print_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < ELEM_TO_PRINT && i < rows; i++) {
        for (int j = 0; j < ELEM_TO_PRINT && j < cols; j++) {
            printf("%4.1f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    if(rows > ELEM_TO_PRINT || cols > ELEM_TO_PRINT) {
        printf(" ...\n");
    }
}


float partialChecksum(const float *matrix, int n)
{
    float sum = 0.0f;
    if (n <= 0) return sum;  // Edge case: no size

    // Identify up to 3 selected rows/columns
    int mid = n / 2;
    int selectedRows[3] = {0, mid, n - 1};
    int selectedCols[3] = {0, mid, n - 1};

    // 1) Sum the selected rows, weighting by (i+1)
    for (int r = 0; r < 3; r++) {
        int i = selectedRows[r];
        // Validate index in range
        if (i >= 0 && i < n) {
            float rowWeight = (float)(i + 1);
            for (int j = 0; j < n; j++) {
                sum += rowWeight * matrix[i * n + j];
            }
        }
    }
    // 2) Sum the selected columns, weighting by (j+1)
    for (int c = 0; c < 3; c++) {
        int j = selectedCols[c];
        // Validate index in range
        if (j >= 0 && j < n) {
            float colWeight = (float)(j + 1);
            for (int i = 0; i < n; i++) {
                sum += colWeight * matrix[i * n + j];
            }
        }
    }

    return sum;
}

// ------ FUNCTIONS TO REMOVE OUTLIERS AND CALCULATE MEANS -------
/// Function to calculate mean of the elements inside an array
float calculate_mean(float arr[], int array_size) {
    float sum = 0.0;
    for (int i = 0; i < array_size; i++) {
        sum += arr[i];
    }
    return sum / array_size;
}

/// Function to calculate standard deviation
float calculate_std_dev(float arr[], int array_size, float mean) {
    float quadratic_sum = 0.0;
    for (int i = 0; i < array_size; i++) {
        quadratic_sum += (arr[i] - mean) * (arr[i] - mean);
    }
    return sqrt(quadratic_sum / array_size);
}

/// Function that cleans an array from outliers based on a threshold from std dev
int remove_outliers(float data[], float filtered_data[], int data_size, float threshold) {
    float media = calculate_mean(data, data_size);
    float deviazione_standard = calculate_std_dev(data, data_size, media);

    int indice_output = 0;
    for (int i = 0; i < data_size; i++) {
        if (fabs(data[i] - media) <= threshold * deviazione_standard) {
            filtered_data[indice_output++] = data[i];
        }
    }
    return indice_output;
}

// ------ FUNCTIONS USEFUL FOR OMP ------
int choose_omp_threads(int matrix_size) {
    if (matrix_size < 64) return 1; // [0, 32]
    if (matrix_size < 512) return matrix_size / 8; // [64, 128, 256]
    return 96; // [512, inf)
}

int choose_block_size(int matrix_size) {
    if (matrix_size < 256) return 16; // [0, 128]
    if (matrix_size < 1024) return 64; // [256, 512]
    return 256; // [1024, inf)
}