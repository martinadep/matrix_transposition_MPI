#include <stdio.h>
#include "main.h"
#include <math.h>
#include <stdlib.h>

/// Dynamic allocation of a square [size] x [size] matrix
float **allocate_sqr_matrix(int size) {
    float **matrix = malloc(size * sizeof(float *));
    if (matrix == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        matrix[i] = (float *) malloc(size * sizeof(float));
        if (matrix[i] == NULL) {
            fprintf(stderr, "Memory allocation failed!\n");
            exit(1);
        }
    }
    return matrix;
}

/// Dynamic deallocation of a square [size] x [size] matrix
void free_matrix(float **matrix, int size) {
    if (matrix != NULL) {
        for (int i = 0; i < size; i++) {
            free(matrix[i]);
        }
        free(matrix);
    }
}

/// Random floats initialization of a square [size] x [size] matrix
void init_matrix(float **matrix, int size) {
    // Initialize the matrix with random values
    //printf("Initializing random %d x %d matrix...\n", size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }
    // printf("Initializaton done\n");
}
/// Function that prints a square [size] x [size] matrix
void print_matrix(float **matrix, int size) {
    for (int i = 0; i < size; i++) {
        printf("\t");
        for(int j = 0; j < size; j++)
            printf("%.3f ", matrix[i][j]);
        printf("\n");
    }
}

// ------ FUNCTIONS TO CANCEL OUTLIERS AND CALCULATE MEANS -------
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