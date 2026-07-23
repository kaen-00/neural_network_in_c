#ifndef NNET_H
#define NNET_H

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

NNET* init_nn(int layers, int *layer_size);
void free_nn(NNET *nn);
int fprop(NNET *nn);
void bprop(NNET *nn, int dataset_idx, MNIST_LAB *lab);
void feed_MNIST_input(MNIST_IMG *imgset, NNET *nn, int img_idx);
void train_loop(int epoch, int batch_size, MNIST_IMG *imgset, MNIST_LAB *labset, NNET *x);
void print_NN(NNET *nn);

#endif
