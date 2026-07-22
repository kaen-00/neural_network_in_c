#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int row;
    int col;
    float **m;
} MAT;


MAT* init_mat(int row, int col, int zero);
void free_mat(MAT *m);
MAT* mat_add(MAT *m1, MAT *m2);
MAT* mat_mul(MAT *m1, MAT *m2);
void mat_relu(MAT *m);
void mat_softmax(MAT *m);
MAT *mat_transpose(MAT *m);
void mat_zero(MAT *m);

#endif
