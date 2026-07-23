#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "parser.h"
#include "matrix.h"
#include "test.h"

typedef struct {
    int layers;
    int *layer_size;
    MAT **activations_v;
    MAT **bias_v;
    MAT **weights_m;
    MAT **error_v;
    MAT **del_weights_m;
    MAT **del_bias_v;
} NNET;

typedef struct {
    float prediction;
    float loss;
} FPROP_RES;

NNET* init_nn(int layers, int *layer_size) {
    NNET *nn = (NNET *)malloc(sizeof(NNET));
    nn->layers = layers;
    nn->layer_size = layer_size;

    // array of pointers containing pointers to the matrices
    MAT **activations_v = (MAT **)malloc(sizeof(MAT*) * layers);
    MAT **bias_v = (MAT **)malloc(sizeof(MAT*) * layers);
    MAT **weights_m = (MAT **)malloc(sizeof(MAT*) * layers);
    MAT **error_v = (MAT **)malloc(sizeof(MAT*) * layers);
    MAT **del_weights_m = (MAT **)malloc(sizeof(MAT*) * layers);
    MAT **del_bias_v = (MAT **)malloc(sizeof(MAT*) * layers);

    // create matrices
    int prev = -1;
    for(int i=0;i<layers;i++) {
        activations_v[i] = init_mat(layer_size[i], 1, 1);
        error_v[i] = init_mat(layer_size[i], 1, 1);
        if(prev == -1) {
            weights_m[i] = NULL;
            bias_v[i] = NULL;
            del_bias_v[i] = NULL; 
            del_weights_m[i] = NULL;
        }
        else {
            weights_m[i] = init_mat(layer_size[i], prev, 0);
            bias_v[i] = init_mat(layer_size[i], 1, 1);
            del_bias_v[i] = init_mat(layer_size[i], 1, 1);
            del_weights_m[i] = init_mat(layer_size[i], prev, 1);
        }
        prev = layer_size[i];
    }
    // randomize weights
    prev = layer_size[0];
    uint32_t seed = (uint32_t)time(NULL);
    for(int i=1;i<layers;i++) {
        float lim = sqrt(6.0/prev); 
        MAT *m = weights_m[i];
        for(int k=0;k<m->row;k++) {
            for(int j=0;j<m->col;j++) {
                float rin = (float)rand_r(&seed) / (float)RAND_MAX;
                rin *= 2;
                rin -= 1;
                rin *= lim;
                m->m[k][j] = rin;
            }
        }
        prev = nn->layer_size[i];
    }
    // assign to nn
    nn->activations_v = activations_v;
    nn->bias_v = bias_v;
    nn->weights_m = weights_m;
    nn->error_v = error_v;
    nn->del_weights_m = del_weights_m;
    nn->del_bias_v = del_bias_v;
    return nn;
}

void free_nn(NNET *nn) {
    for(int i=0;i<nn->layers;i++) {
        free_mat(nn->activations_v[i]);
        free_mat(nn->weights_m[i]);
        free_mat(nn->bias_v[i]);
        free_mat(nn->error_v[i]);
        free_mat(nn->del_bias_v[i]);
        free_mat(nn->del_weights_m[i]);
    }
    free(nn->activations_v);
    free(nn->bias_v);
    free(nn->weights_m);
    free(nn->error_v);
    free(nn->del_weights_m);
    free(nn->del_bias_v);
    free(nn);
}

FPROP_RES* fprop(NNET *nn) {
    MAT *next = NULL;
    for(int i=1;i<nn->layers;i++) {
        MAT *W = nn->weights_m[i];
        MAT *a = nn->activations_v[i-1];
        MAT *b = nn->bias_v[i];
        MAT *mul_res = mat_mul(W, a);
        MAT *add_res = mat_add(mul_res, b);
        if(i != nn->layers-1) mat_relu(add_res);
        else mat_softmax(add_res);
        MAT *erase = nn->activations_v[i];
        nn->activations_v[i] = add_res;
        free_mat(erase);
        free_mat(mul_res);
    }
    // cost calculation
    float cost = 0.0f;
    float max = 0.0f;
    int idx_last_layer = nn->layers-1;
    int size_of_last_layer = nn->layer_size[idx_last_layer];
    for(int i=0;i<size_of_last_layer;i++) {
        float act = nn->activations_v[idx_last_layer]->m[i][0];
        cost += -logf(act + 1e-7j);
        if(act > max) max = act;
    }
    printf("Loss: %f\n", cost);

    FPROP_RES *res = (FPROP_RES *)malloc(sizeof(FPROP_RES));
    res->loss = cost;
    res->prediction = max;
    return res;
}

void bprop(NNET *nn, int dataset_idx, MNIST_LAB *lab) {
    FPROP_RES *res = fprop(nn);
    // do something with res
    free(res);
    int y = (int)lab->dataset[dataset_idx];  
    int idx_last_layer = nn->layers-1;
    int size_of_last_layer = nn->layer_size[idx_last_layer];
    // equation 1
    for(int i=0;i<size_of_last_layer;i++) {
        float target = (i == y) ? 1.0f : 0.0f;
        nn->error_v[idx_last_layer]->m[i][0] = nn->activations_v[idx_last_layer]->m[i][0] - target;
    }
    // equation 2 
    for(int i=idx_last_layer-1;i>0;i--) {
        MAT *transpose = mat_transpose(nn->weights_m[i+1]);
        MAT *mul = mat_mul(transpose, nn->error_v[i]);
        for(int j=0;j<mul->row;j++) {
            if(nn->activations_v[i]->m[j][0] <= 0.0f) nn->error_v[i]->m[j][0] = 0.0f;
            else nn->error_v[i]->m[j][0] = mul->m[j][0];
        }
        free_mat(mul);
        free_mat(transpose);
        nn->error_v[i] = mul;
    }
    // equation 3 and 4
    for(int i=1;i<idx_last_layer;i++) {
        for(int j=0;j<nn->del_bias_v[i]->row;j++) nn->del_bias_v[i]->m[j][0] += nn->error_v[i]->m[j][0];
        MAT *mul = mat_mul(nn->error_v[i], nn->activations_v[i]);
        MAT *add = mat_add(nn->del_weights_m[i], mul);
        free_mat(mul);
        free_mat(nn->del_weights_m[i]);
        nn->del_weights_m[i] = add;
    } 
}


void feed_MNIST_input(MNIST_IMG *imgset, NNET *nn, int img_idx) {
    int offset = imgset->num_rows * imgset->num_cols;
    if(nn->layer_size[0] != offset) {
        printf("input missmatch\n");
        return;
    }
    MAT *input_layer = nn->activations_v[0];
    for(int i=0;i<offset;i++) {
        input_layer->m[i][0] = (float)imgset->dataset[i + img_idx * offset]/255.0f;
    }
}

void train_loop(int epoch, int batch_size, MNIST_IMG *imgset, MNIST_LAB *labset, NNET *x) {
    for(int epoch_no = 0; epoch_no < epoch ; epoch_no++) {
        // copy the current nn into a backup nn
        for(int spe = 0; spe < batch_size ; spe++) {
            /*
               load the backup nn
                get randomized indexes
                feed input and calculate cost
            */
        }
        // calculate average cost
        // backpropogate and find error terms and hence the gradients wrt w and b
        // update w and b based on learning rate
    }
}

int main() {
    
    test_matrices();
    test_parser();
    return 0;
}
