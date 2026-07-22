#include "parser.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

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
