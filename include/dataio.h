//
// Created by MAC on 2019/12/27.
//

#ifndef CNN_DATAIO_H
#define CNN_DATAIO_H

#include <stdio.h>
#include <assert.h>
#include "vector.h"

int ReverseInt(int i) {
    unsigned char ch1 = i & 255;
    unsigned char ch2 = i >> 8 & 255;
    unsigned char ch3 = i >> 16 & 255;
    unsigned char ch4 = i >> 24 & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
Vector1D read_Mnist_Label(const char * fileName) {
    Vector1D labels = Vector1D_();
    FILE* file = fopen(fileName, "rb");
    assert(file != NULL);
    int tmp;
    if (file) {
        int magic_number = 0, number_of_images = 0;
        tmp = fread(&magic_number, sizeof(magic_number), 1, file);
        tmp = fread(&number_of_images, sizeof(number_of_images), 1, file);
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        labels.new(&labels, number_of_images);
        for (int i = 0; i < number_of_images; i++) {
            unsigned char label = 0;
            tmp = fread(&label, sizeof(label), 1, file);
            labels.data[i] = (double)label;
        }
    }
    return labels;
}
Vector2D read_Mnist_Images(const char * fileName) {
    Vector2D images = Vector2D_();
    FILE* file = fopen(fileName, "rb");
    assert(file != NULL);
    int tmp;
    if (file) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        tmp = fread(&magic_number, sizeof(magic_number), 1, file);
        tmp = fread(&number_of_images, sizeof(number_of_images), 1, file);
        tmp = fread(&n_rows, sizeof(n_rows), 1, file);
        tmp = fread(&n_cols, sizeof(n_cols), 1, file);
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        n_rows = ReverseInt(n_rows);
        n_cols = ReverseInt(n_cols);
        images.new(&images, number_of_images, n_rows * n_cols);
        for (int i = 0; i < number_of_images; i++) {
            for (int r = 0; r < n_rows; r++) {
                for (int c = 0; c < n_cols; c++) {
                    unsigned char image = 0;
                    tmp = fread(&image, sizeof(image), 1, file);
                    images.data[i][r * n_rows + c] = (double)image;
                }
            }
        }
    }
    return images;
}

#endif //CNN_DATAIO_H
