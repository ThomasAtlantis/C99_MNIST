//
// Created by MAC on 2019/12/1.
//

#ifndef CNN_VECTOR_H
#define CNN_VECTOR_H

#include <stdio.h>
#include <stdlib.h>
#include "../include/mytype.h"
#include "../include/memtool.h"

typedef struct vector1D {
    _vector1D_t * data;
    int lens;
    int (*new)(struct vector1D *, int);
    void (*destroy)(struct vector1D *);
} Vector1D;

int constructVector1D(Vector1D * this, int len) {
    this->data = (_vector1D_t *)malloc(len * sizeof(_vector1D_t));
    if (this->data) this->lens = len;
    return this->data != NULL;
}

void destroyVector1D(Vector1D * this) {
    free_p(this->data);
}

Vector1D Vector1D_() {
    Vector1D vector1D;
    vector1D.data = NULL;
    vector1D.lens = 0;
    vector1D.new = constructVector1D;
    vector1D.destroy = destroyVector1D;
    return vector1D;
}

typedef struct vector2D {
    _vector2D_t ** data;
    int rows, cols;
    int (*new)(struct vector2D *, int, int);
    void (*destroy)(struct vector2D *);
} Vector2D;

int constructVector2D(Vector2D * this, int r, int c) {
    int error = 0;
    this->data = (_vector2D_t **)malloc(r * sizeof(_vector2D_t *));
    for (int i = 0; i < r; ++ i) {
        this->data[i] = (_vector2D_t *)malloc(c * sizeof(_vector2D_t));
        if (!this->data[i]) error = 1;
    }
    if (this->data && !error) {
        this->rows = r;
        this->cols = c;
        return 1;
    } else {
        for (int i = 0; i < r; ++ i)
            if (this->data[i]) free_p(this->data[i]);
        free_p(this->data);
        return 0;
    }
}

void destroyVector2D(Vector2D * this) {
    for (int i = 0; i < this->rows; ++ i)
        free_p(this->data[i]);
    free_p(this->data);
}

Vector2D Vector2D_() {
    Vector2D vector2D;
    vector2D.data = NULL;
    vector2D.rows = 0;
    vector2D.cols = 0;
    vector2D.new = constructVector2D;
    vector2D.destroy = destroyVector2D;
    return vector2D;
}

#endif //CNN_VECTOR_H
