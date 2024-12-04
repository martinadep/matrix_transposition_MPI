#include <math.h>
#include <stdio.h>

#include "../main.h"

/// This is a function used to test correctness of the checkSym()s and matTransp()s
/// It performs tests on [4x4] matrices, both symmetric and asymmetric
void test_functions() {
    int two_pow = 2;
    float **M = allocate_sqr_matrix(pow(2, two_pow));
    float **T = allocate_sqr_matrix(pow(2, two_pow));
    init_matrix(M, pow(2, two_pow));
    printf("M matrix before transposition:\n");
    print_matrix(M, pow(2, two_pow));
    printf("\n");

    bool check_Sym = checkSym(M, two_pow);
    if (check_Sym) {
        printf("Matrix is symmetric [check_Sym], no need to transpose\n");
    } else {
        printf("Matrix is NOT symmetric [check_Sym], transposing\n");
        matTranspose(M, T, two_pow);
        print_matrix(T, pow(2, two_pow));
    }
    printf("\n");

    bool check_SymImp = checkSymImp(M, two_pow);
    if (check_SymImp) {
        printf("Matrix is symmetric [check_SymImp], no need to transpose\n");
    } else {
        printf("Matrix is NOT symmetric [check_SymImp], transposing\n");
        matTransposeImp(M, T, two_pow);
        print_matrix(T, pow(2, two_pow));
    }
    printf("\n");

    bool check_SymOMP = checkSymOMP(M, two_pow);
    if (check_SymOMP) {
        printf("Matrix is symmetric [check_SymOMP], no need to transpose\n");
    } else {
        printf("Matrix is NOT symmetric [check_SymOMP], transposing\n");
        matTransposeOMP(M, T, two_pow);
        print_matrix(T, pow(2, two_pow));
    }

    printf("\n-----------------------------\nM_symmetric:\n");
    float symmetric[4][4] = {
        {1.0, 2.0, 3.0, 4.0},
        {2.0, 1.0, 0.0, 5.0},
        {3.0, 0.0, 1.0, 6.0},
        {4.0, 5.0, 6.0, 7.0}
    };
    float **M_symmetric = allocate_sqr_matrix(4);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            M_symmetric[i][j] = symmetric[i][j];
        }
    }

    print_matrix(M_symmetric, pow(2, 2));
    if (checkSym(M_symmetric, 2)) printf("Matrix is symmetric [checkSym]\n");
    else printf("Matrix is NOT symmetric [checkSym]\n");

    if (checkSymImp(M_symmetric, 2)) printf("Matrix is symmetric [checkSymImp]\n");
    else printf("Matrix is NOT symmetric [checkImp]\n");

    if (checkSymOMP(M_symmetric, 2)) printf("Matrix is symmetric [check_SymOMP]\n");
    else printf("Matrix is NOT symmetric [checkOMP]\n");
}