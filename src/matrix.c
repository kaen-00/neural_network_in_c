#include "matrix.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

MAT* init_mat(int row, int col, int zero) {
    MAT *mat = (MAT *)malloc(sizeof(MAT));
    mat->row = row;
    mat->col = col;
    float **m = (float **)malloc(row * sizeof(float*));
    for(int i=0;i<row;i++) m[i] = (float *)malloc(col * sizeof(float));
    mat->m = m;
    if(zero == 1) {
        for(int i=0;i<row;i++) {
            for(int j=0;j<col;j++) mat->m[i][j] = 0.0f;
        }
    }
    return mat;
}

void free_mat(MAT *m) {
    if(m == NULL) return;
    for(int i=0;i<m->row;i++) free(m->m[i]);
    free(m->m);
    free(m);
    return;
}

MAT* mat_add(MAT *m1, MAT *m2) {
    if(m1->row != m2->row || m1->col != m2->col) {
        printf("invalid addition\n"); return NULL;
    }
    MAT *new = init_mat(m1->row, m1->col, 0);
    for(int i=0;i<m1->row;i++) {
        for(int j=0;j<m1->col;j++) new->m[i][j] = m1->m[i][j] + m2->m[i][j];
    }
    return new;
}

MAT* mat_mul(MAT *m1, MAT *m2) {
   if(m1->col != m2->row) {
       printf("invalid multiplication"); return NULL;
   } 
   MAT *new = init_mat(m1->row, m2->col, 0);
   for(int i=0;i<new->row;i++) {
       for(int j=0;j<new->col;j++) {
           float sum = 0.0f;
           for(int k=0;k<m1->col;k++) {
               sum += m1->m[i][k] * m2->m[k][j]; 
           }
           new->m[i][j] = sum;
       }
   }
   return new;
}

void mat_relu(MAT *m) {
    for(int i=0;i<m->row;i++) {
        for(int j=0;j<m->col;j++) {
            if(m->m[i][j] < 0) m->m[i][j] = 0;
        }
    }
    return;
}

void mat_softmax(MAT *m) {
    float sum = 0.0f;
    float max = 0.0f;
    for(int i=0;i<m->row;i++) if(max < m->m[i][0]) max = m->m[i][0];
    for(int i=0;i<m->row;i++) {
        m->m[i][0] = expf(m->m[i][0] - max);
        sum += m->m[i][0];
    }
    for(int i=0;i<m->row;i++) {
        m->m[i][0] /= sum;
    }
    return;
}

MAT *mat_transpose(MAT *m) {
    MAT *new = init_mat(m->col, m->row, 0);
    for(int i=0;i<m->row;i++) {
        for(int j=i;j<m->col;j++) {
            new->m[j][i] = m->m[i][j];
        }
    }
    return new;
}

void mat_zero(MAT *m) {
    for(int i=0;i<m->row;i++) {
        for(int j=0;j<m->col;j++) {
            m->m[i][j] = 0.0f;
        }
    }
} 

void mat_print(MAT *c) {
    if(c == NULL) return;
    for(int i=0;i<c->row;i++) {
        for(int j=0;j<c->col;j++) printf("%f ", c->m[i][j]);
        printf("\n");
    }
}


