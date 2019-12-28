//
// Created by MAC on 2019/12/28.
//

#ifndef CNN_NETWORK_H
#define CNN_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/vector.h"
#include "../include/memtool.h"
#include "../include/mytype.h"

#define _max(a,b) (((a)>(b))?(a):(b))
#define _min(a,b) (((a)>(b))?(b):(a))

// ReLU函数
_type ReLU(_type x) {
    return _max(0.0, x);
}

_type sigmoid(_type x) {
    return 1.0 / (1.0 + exp(-x));
}

const double pi = 3.141592654;
double GaussRand(double mean, double sigma) {
    static double U, V;
    static int phase = 0;
    double Z;
    if (phase == 0) {
        U = rand() / (RAND_MAX + 1.0);
        V = rand() / (RAND_MAX + 1.0);
        Z = sqrt(-2.0 * log(U)) * sin(2.0 * pi * V);
    } else {
        Z = sqrt(-2.0 * log(U)) * cos(2.0 * pi * V);
    }
    phase = 1 - phase;
    return mean + Z * sigma;
}

typedef struct {
    double para_1, para_2;
} Alpha;

Alpha * ExpDecayLR(double start, int epoch, double end) {
    Alpha * alpha = (Alpha *)malloc(sizeof(Alpha));
    alpha->para_2 = log(end / start) / epoch;
    alpha->para_1 = start;
    return alpha;
}

double expDecayLR(Alpha * alpha, int step) {
    return alpha->para_1 * exp(alpha->para_2 * step);
}

typedef struct CVLayer {
    int L, W, H;
    _type bias;
    _type *** values;
    _type *** deltas;
    void (*destroy)(struct CVLayer *);
} CVLayer;

void killCVLayer(CVLayer * this) {
    free3D(this->values, this->H, this->L);
    free3D(this->deltas, this->H, this->L);
    free_p(this);
}

/**
 * @param h 图片的通道数
 * @param l 图片的长/行数
 * @param w 图片的宽/列数
 * @return 卷积层结构
 * values是存值的矩阵，deltas是梯度下降的误差敏感项
 */
CVLayer * Convol2D_(int h, int l, int w) {
    CVLayer * layer = (CVLayer *)malloc(sizeof(CVLayer));
    layer->destroy = killCVLayer;
    layer->L = l; layer->W = w; layer->H = h;
    layer->values = (_type ***)malloc(h * sizeof(_type **));
    layer->deltas = (_type ***)malloc(h * sizeof(_type **));
    for (int k = 0; k < h; ++ k) {
        layer->values[k] = (_type **) malloc(l * sizeof(_type *));
        layer->deltas[k] = (_type **) malloc(l * sizeof(_type *));
        for (int i = 0; i < l; ++ i) {
            layer->values[k][i] = (_type *) malloc(w * sizeof(_type));
            layer->deltas[k][i] = (_type *) malloc(w * sizeof(_type));
            for (int j = 0; j < w; ++ j) {
                layer->values[k][i][j] = 0.01 * (rand() % 100);
            }
        }
    }
    return layer;
}

CVLayer * Filter2D_(int h, int l, int w) {
    CVLayer * layer = (CVLayer *)malloc(sizeof(CVLayer));
    layer->destroy = killCVLayer; layer->bias = 0;
    layer->L = l; layer->W = w; layer->H = h; layer->deltas = NULL;
    layer->values = (_type ***)malloc(h * sizeof(_type **));
    for (int k = 0; k < h; ++ k) {
        layer->values[k] = (_type **) malloc(l * sizeof(_type *));
        for (int i = 0; i < l; ++ i) {
            layer->values[k][i] = (_type *) malloc(w * sizeof(_type));
            for (int j = 0; j < w; ++ j) {
                layer->values[k][i][j] = 0.01 * (rand() % 100);
            }
        }
    }
    return layer;
}

// 全连接网络层
typedef struct FCLayer {
    int L, W;
    _type **weights;
    _type * bias;
    _type * values;
    _type * deltas;
    void (*destroy)(struct FCLayer *);
} FCLayer;

void killFCLayer(FCLayer * this) {
    free2D(this->weights, this->L);
    free1D(this->bias);
    free1D(this->values);
    free1D(this->deltas);
    free_p(this);
}

FCLayer * FCLayer_(int length) {
    FCLayer * layer = (FCLayer *)malloc(sizeof(FCLayer));
    layer->destroy = killFCLayer;
    layer->values = (_type *)malloc(length * sizeof(_type));
    layer->deltas = (_type *)malloc(length * sizeof(_type));
    layer->L = length; layer->W = 1;
    layer->weights = NULL; layer->bias = NULL;
    return layer;
}

FCLayer * FCWeight_(int from, int to) {
    FCLayer * layer = (FCLayer *)malloc(sizeof(FCLayer));
    layer->destroy = killFCLayer;
    layer->bias = (_type *)malloc(to * sizeof(_type));
    layer->weights = (_type **)malloc(to * sizeof(_type));
    for (int i = 0; i < to; ++ i) {
        layer->weights[i] = (_type *)malloc(from * sizeof(_type));
        for (int j = 0; j < from; ++ j) {
            layer->weights[i][j] = GaussRand(0, 0.1);
        }
    }
    layer->values = layer->deltas = NULL;
    layer->L = to; layer->W = from;
    return layer;
}

void softmax(FCLayer * A) {
    _type sum = 0.0; _type maxi = -100000000;
    for (int i = 0; i < A->L; ++ i) {
        maxi = _max(maxi, A->values[i]);
    }
    for (int i = 0; i < A->L; ++ i) {
        A->values[i] = exp(A->values[i] - maxi);
        sum += A->values[i];
    }
    for (int i = 0; i < A->L; ++ i) {
        A->values[i] /= sum;
    }
}

#endif //CNN_NETWORK_H
