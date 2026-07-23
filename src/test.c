#include "test.h"
#include "matrix.h"
#include <stdio.h>
#include "parser.h"
#include "NNet.h"

void test_matrices() {
    MAT *b = init_mat(2, 3, 1);
    MAT *a = init_mat(2, 3, 1);
    a->m[0][0] = 1;
    a->m[0][1] = 2;
    a->m[0][2] = 3;
    a->m[1][0] = 4;
    a->m[1][1] = 5;
    a->m[1][2] = 6;
    b->m[0][0] = 1;
    b->m[0][1] = 2;
    b->m[0][2] = 3;
    b->m[1][0] = 4;
    b->m[1][1] = 5;
    b->m[1][2] = 6;
    MAT *c = mat_add(a, b);
    printf("mat add:\n");
    mat_print(c);
    free_mat(c);
    free_mat(b);
    b = init_mat(3, 2, 1);
    b->m[0][0] = 2;
    b->m[0][1] = -2;
    b->m[1][0] = -2;
    b->m[1][1] = 2;
    c = mat_mul(a,b);
    printf("mat mul\n");
    mat_print(c);
    printf("mat relu\n");
    mat_relu(c);
    mat_print(c);
    printf("mat transpose\n");
    free_mat(b);
    b = mat_transpose(c);
    mat_print(b);
    printf("mat zero\n");
    mat_zero(c);
    mat_print(c);
    printf("softmax:\n");
    free_mat(a);
    free_mat(b);
    free_mat(c);
    a = init_mat(3, 1, 1);
    a->m[0][0] = 2;
    a->m[1][0] = 1;
    a->m[2][0] = 3;
    mat_softmax(a);
    mat_print(a);
    free_mat(a);
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

    char *test_img = "/home/kaen/Desktop/neuralNetwork/dataset/train-images.idx3-ubyte";
    MNIST_IMG *timg = load_image_ds(test_img);

    printf("\n");
    for(int i=0;i<28;i++) {
        for(int j=0;j<28;j++) {
            if(GET(timg, 2, i, j) != 0) printf("1 ");
            else printf("0 "); 
        }
        printf("\n");
    }

    char *test_lab = "/home/kaen/Desktop/neuralNetwork/dataset/train-labels.idx1-ubyte";
    MNIST_LAB *tlab = load_label_ds(test_lab);
    printf("%d\n", tlab->dataset[2]);
    free(img);
    free(tlab);
}

void test_NN_init() {
    int arr[3] = {3, 2, 3};
    NNET *nn = init_nn(3, arr);
    print_NN(nn);
    free_nn(nn);
}

void test_NN_fprop() {
    int arr[3] = {3, 2, 3};
    NNET *nn = init_nn(3, arr);
    nn->activations_v[0]->m[0][0] = 1;
    nn->activations_v[0]->m[1][0] = 2;
    nn->activations_v[0]->m[2][0] = 3;
    int class = fprop(nn);
    printf("predicted class\n: %d", class);
    print_NN(nn);
    free_nn(nn);
}
