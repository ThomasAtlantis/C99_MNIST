//
// Created by MAC on 2019/12/31.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define filter_num 5
#define class_num 10
#define _max(a,b) (((a)>(b))?(a):(b))
int test_num  = 4000;
int show_flag = 0;

typedef double _type;

_type ReLU(_type x) {
    return _max(0.0, x);
}

_type sigmoid(_type x) {
    return 1.0 / (1.0 + exp(-x));
}

typedef struct CVLayer {
    int L, W, H;
    _type bias;
    _type *** values;
    _type *** deltas;
} CVLayer;

CVLayer * loadFilter(FILE * fp) {
    CVLayer * layer = (CVLayer *)malloc(sizeof(CVLayer));
    fread((char *)&layer->H, sizeof(int), 1, fp);
    fread((char *)&layer->L, sizeof(int), 1, fp);
    fread((char *)&layer->W, sizeof(int), 1, fp);
    fread((char *)&layer->bias, sizeof(_type), 1, fp);
    layer->deltas = NULL;
    layer->values = (_type ***)malloc(layer->H * sizeof(_type **));
    for (int k = 0; k < layer->H; ++ k) {
        layer->values[k] = (_type **) malloc(layer->L * sizeof(_type *));
        for (int i = 0; i < layer->L; ++ i) {
            layer->values[k][i] = (_type *) malloc(layer->W * sizeof(_type));
            fread((char *)layer->values[k][i], sizeof(_type), layer->W, fp);
        }
    }
    return layer;
}

CVLayer * Convol2D_(int h, int l, int w) {
    CVLayer * layer = (CVLayer *)malloc(sizeof(CVLayer));
    layer->bias = 0;
    layer->L = l; layer->W = w; layer->H = h;
    layer->values = (_type ***)malloc(h * sizeof(_type **));
    layer->deltas = NULL;
    for (int k = 0; k < h; ++ k) {
        layer->values[k] = (_type **) malloc(l * sizeof(_type *));
        for (int i = 0; i < l; ++ i) {
            layer->values[k][i] = (_type *) malloc(w * sizeof(_type));
        }
    }
    return layer;
}

typedef struct FCLayer {
    int L, W;
    _type **weights;
    _type * bias;
    _type * values;
    _type * deltas;
} FCLayer;

FCLayer * loadFCWeight(FILE * fp) {
    FCLayer * layer = (FCLayer *)malloc(sizeof(FCLayer));
    fread((char *)&layer->L, sizeof(int), 1, fp);
    fread((char *)&layer->W, sizeof(int), 1, fp);
    layer->values = layer->deltas = NULL;
    layer->bias = (_type *)malloc(layer->L * sizeof(_type));
    layer->weights = (_type **)malloc(layer->L * sizeof(_type));
    fread((char *)layer->bias, sizeof(_type), layer->L, fp);
    for (int i = 0; i < layer->L; ++ i) {
        layer->weights[i] = (_type *)malloc(layer->W * sizeof(_type));
        fread((char *)layer->weights[i], sizeof(_type), layer->W, fp);
    }
    return layer;
}

FCLayer * FCLayer_(int length) {
    FCLayer * layer = (FCLayer *)malloc(sizeof(FCLayer));
    layer->values = (_type *)malloc(length * sizeof(_type));
    layer->deltas = (_type *)malloc(length * sizeof(_type));
    layer->L = length; layer->W = 1;
    layer->weights = NULL; layer->bias = NULL;
    return layer;
}

typedef struct Network {
    CVLayer * input_layer;
    CVLayer * filter[filter_num];
    CVLayer * conv_layer;
    CVLayer * pool_layer;
    FCLayer * fc_input;
    FCLayer * fc_weight;
    FCLayer * fc_output;
} Network;

Network * CNN;
int fact;

void loadNetwork() {
    FILE * fp = fopen("../model-20191231-8975.sav", "rb");
    FILE * fp_1 = fopen("../image.mem", "rb");
    CNN = (Network *)malloc(sizeof(Network));
    CNN->input_layer = Convol2D_(1, 28, 28);
    for (int i = 0; i < 28; ++ i)
        for (int j = 0; j < 28; ++ j)
            fread((char *) &CNN->input_layer->values[0][i][j],
                sizeof(_type), 1, fp_1);
    fread((char *)&fact, sizeof(int), 1, fp_1);
    for (int i = 0; i < filter_num; ++ i)
        CNN->filter[i] = loadFilter(fp);
    CNN->conv_layer = Convol2D_(5, 24, 24);
    CNN->pool_layer = Convol2D_(5, 12, 12);
    CNN->fc_input = FCLayer_(720);
    CNN->fc_weight = loadFCWeight(fp);
    CNN->fc_output = FCLayer_(10);
    fclose(fp);
    fclose(fp_1);
}

void conv(CVLayer * input, CVLayer * filter[], CVLayer * conv) {
    for (int p = 0; p < conv->H; ++ p) {
        for (int i = 0; i < conv->L; ++ i) {
            for (int j = 0; j < conv->W; ++ j) {
                conv->values[p][i][j] = 0;
                for (int k = 0; k < input->H; ++ k)
                    for (int a = 0; a < filter[0]->L; ++ a)
                        for (int b = 0; b < filter[0]->W; ++ b)
                            conv->values[p][i][j] += input->values[k][i + a][j + b] * filter[p]->values[k][a][b];
                conv->values[p][i][j] = ReLU(conv->values[p][i][j] + filter[p]->bias);
            }
        }
    }
}

void maxPooling(CVLayer * conv, CVLayer * pool) {
    for (int k = 0; k < conv->H; ++ k)
        for (int i = 0; i < conv->L; i += 2)
            for (int j = 0; j < conv->W; j += 2)
                pool->values[k][i / 2][j / 2] = _max(
                    _max(conv->values[k][i][j], conv->values[k][i][j + 1]),
                    _max(conv->values[k][i + 1][j],conv->values[k][i + 1][j + 1])
                );
}

void FCInput(CVLayer * pool, FCLayer * fcInput) {
    int x = 0;
    for (int k = 0; k < pool->H; ++ k)
        for (int i = 0; i < pool->L; ++ i)
            for (int j = 0; j < pool->W; ++ j)
                fcInput->values[x ++] = sigmoid(pool->values[k][i][j]);
}

void FCMultiply(FCLayer * input, FCLayer * weight, FCLayer * output) {
    for (int i = 0; i < output->L; ++ i) {
        output->values[i] = 0;
        for (int j = 0; j < input->L; ++ j)
            output->values[i] += weight->weights[i][j] * input->values[j];
        output->values[i] += weight->bias[i];
    }
}

void forePropagation() {
    conv(CNN->input_layer, CNN->filter, CNN->conv_layer);
    maxPooling(CNN->conv_layer, CNN->pool_layer);
    FCInput(CNN->pool_layer, CNN->fc_input);
    FCMultiply(CNN->fc_input, CNN->fc_weight, CNN->fc_output);
}

int predict() {
    forePropagation();
    int ans = -1; _type sign = -1;
    for (int i = 0; i < class_num; ++ i) {
        if (CNN->fc_output->values[i] > sign) {
            sign = CNN->fc_output->values[i];
            ans = i;
        }
    }
    return ans;
}

int main() {
    loadNetwork();
    printf("Ground Truth: %d, Predicted: %d", fact, predict());
    return 0;
}
