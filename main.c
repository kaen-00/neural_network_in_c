#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define GET(ds, img, r, c) \
    ((ds)->dataset[((size_t)(img) * (ds)->num_rows * (ds)->num_cols) + ((size_t)(r) * (ds)->num_cols) + (c)])

typedef struct {
    uint32_t num_images;
    uint32_t num_rows;
    uint32_t num_cols;
    uint8_t *dataset;
} MNIST_IMG;

typedef struct {
    uint32_t num_labels;
    uint8_t *dataset;
} MNIST_LAB;

typedef struct {
    int row;
    int col;
    float **m;
} MAT;

typedef struct {
    int layers;
    int *layer_size;
    MAT **activations_v;
    MAT **bias_v;
    MAT **weights_m;
} NNET;

MAT* init_mat(int row, int col) {
    MAT *mat = (MAT *)malloc(sizeof(MAT));
    mat->row = row;
    mat->col = col;
    float **m = (float **)malloc(row * sizeof(float*));
    for(int i=0;i<row;i++) m[i] = (float *)malloc(col * sizeof(float));
    mat->m = m;
    return mat;
}

void free_mat(MAT *m) {
    if(m == NULL) return;
    for(int i=0;i<m->row;i++) free(m->m[i]);
    free(m->m);
    free(m);
    return;
}

NNET* init_nn(int layers, int *layer_size) {
    NNET *nn = (NNET *)malloc(sizeof(NNET));
    nn->layers = layers;
    nn->layer_size = layer_size;
    MAT **activations_v = (MAT **)malloc(sizeof(MAT*) * layers);
    MAT **bias_v = (MAT **)malloc(sizeof(MAT*) * layers);
    MAT **weights_m = (MAT **)malloc(sizeof(MAT*) * layers);
    int prev = -1;
    for(int i=0;i<layers;i++) {
        activations_v[i] = init_mat(layer_size[i], 1);
        if(prev == -1) {
            weights_m[i] = NULL;
            bias_v[i] = NULL;
        }
        else {
            weights_m[i] = init_mat(layer_size[i], prev);
            bias_v[i] = init_mat(layer_size[i], 1);
        }
        prev = layer_size[i];
    }
    nn->activations_v = activations_v;
    nn->bias_v = bias_v;
    nn->weights_m = weights_m;
    return nn;
}

void free_nn(NNET *nn) {
    for(int i=0;i<nn->layers;i++) {
        free_mat(nn->activations_v[i]);
        free_mat(nn->weights_m[i]);
        free_mat(nn->bias_v[i]);
    }
    free(nn->activations_v);
    free(nn->bias_v);
    free(nn->weights_m);
    free(nn);
}

MAT* mat_mul(MAT *m1, MAT *m2) {
    
}

uint32_t swap_uint32(uint32_t val) {
    return ((val & 0xFF000000) >> 24) |
           ((val & 0x00FF0000) >> 8)  |
           ((val & 0x0000FF00) << 8)  |
           ((val & 0x000000FF) << 24);
}

MNIST_IMG* load_image_ds(char *fpath) {

    FILE *img = fopen(fpath , "rb");
    if(img == NULL) return NULL;

    uint32_t header[4];
    fread(header, sizeof(uint32_t), 4, img);
    for(int i=0;i<4;i++) header[i] = swap_uint32(header[i]);

    MNIST_IMG *ds =(MNIST_IMG *) malloc(sizeof(MNIST_IMG));
    ds->num_images = header[1];
    ds->num_rows = header[2];
    ds->num_cols = header[3];
    ds->dataset = (uint8_t *)malloc((size_t)header[1] * header[2] * header[3] * sizeof(uint8_t));
    fread(ds->dataset, sizeof(uint8_t), header[1] * header[2] * header[3], img);

    fclose(img);
    return ds;
}

MNIST_LAB* load_label_ds(char *fpath) {

    FILE *lab = fopen(fpath, "rb");
    if(lab == NULL) return NULL;

    uint32_t header[2];
    fread(header, sizeof(uint32_t), 2, lab);
    for(int i=0;i<2;i++) header[i] = swap_uint32(header[i]);

    MNIST_LAB *ds = (MNIST_LAB *)malloc(sizeof(MNIST_LAB));
    ds->num_labels = header[1];
    ds->dataset = (uint8_t *)malloc(header[1] * sizeof(uint8_t));
    fread(ds->dataset, sizeof(uint8_t), header[1], lab);

    fclose(lab);
    return ds;
}

void train_loop(int epoch, int size_per_epoch, MNIST_IMG imgset, MNIST_LAB labset, NNET x) {

}

int main() {

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
    return 0;
}
