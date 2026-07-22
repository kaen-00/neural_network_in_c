#include "test.h"
#include "matrix.h"
#include <stdio.h>
#include "parser.h"

void test_matrices() {
    MAT *b = init_mat(2, 3, 1);
    MAT *a = init_mat(2, 3, 1);
    MAT *c = mat_add(a, b);
    for(int i=0;i<c->row;i++) {
        for(int j=0;j<c->col;j++) printf("%f ", c->m[i][j]);
        printf("\n");
    }
}


void test_parser() {

    char *train_img = "/home/kaen/Desktop/neuralNetwork/dataset/t10k-images.idx3-ubyte";
    MNIST_IMG *img = load_image_ds(train_img);

    printf("\n");
    for(int i=0;i<28;i++) {
        for(int j=0;j<28;j++) {
            if(GET(img, 2, i, j) != 0) printf("1 ");
            else printf("0 "); 
        }
        printf("\n");
    }

    char *train_lab = "/home/kaen/Desktop/neuralNetwork/dataset/t10k-labels.idx1-ubyte";
    MNIST_LAB *lab = load_label_ds(train_lab);
    printf("%d\n", lab->dataset[2]);
}
