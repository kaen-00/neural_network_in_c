#ifndef PARSER_H
#define PARSER_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

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

MNIST_LAB* load_label_ds(char *fpath);
MNIST_IMG* load_image_ds(char *fpath);

#endif
