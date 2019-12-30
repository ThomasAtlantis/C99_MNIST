#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../include/vector.h"
#include "../include/dataio.h"

#define _type double

#define _max(a,b) (((a)>(b))?(a):(b))
#define _min(a,b) (((a)>(b))?(b):(a))

const int train_num = 100;
const int test_number = 100;
const int out = 10;
const int epoch_num = 10;
int step;
Vector1D labels_train;
Vector2D images_train;
Vector1D labels_test;
Vector2D images_test;

const double PI = 3.141592654;
const double mean = 0;
const double sigma = 0.1;
double gaussrand() {
    static double U, V;
    static int phase = 0;
    double Z;
    if (phase == 0) {
        U = rand() / (RAND_MAX + 1.0);
        V = rand() / (RAND_MAX + 1.0);
        Z = sqrt(-2.0 * log(U)) * sin(2.0 * PI * V);
    } else {
        Z = sqrt(-2.0 * log(U)) * cos(2.0 * PI * V);
    }
    phase = 1 - phase;
    return mean + Z * sigma;
}
double alpha() {
    return 0.04 * exp(-0.023 * step);
}

typedef struct {
    int L, W, H;
    _type values[30][30][5];
    _type bias[30][30][5];
    _type deltas[30][30][5];
} Layer;
Layer * Layer_() {
    Layer * layer = (Layer *)malloc(sizeof(Layer));
    for (int i = 0; i < 30; ++ i)
        for (int j = 0; j < 30; ++j)
            for (int k = 0; k < 5; ++k)
                layer->values[i][j][k] = 0.01 * (rand() % 100);
    return layer;
}
typedef struct {
    int length;
    _type values[1000];
    _type bias[1000];
    _type deltas[1000];
    _type weights[20][1000];
} Fcnn_layer;
Fcnn_layer * Fcnn_layer_() {
    Fcnn_layer * fcnn_layer = (Fcnn_layer *)malloc(sizeof(Fcnn_layer));
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 1000; ++j)
            fcnn_layer->weights[i][j] = gaussrand();
    return fcnn_layer;
}
typedef struct {
    Layer * Input_layer;
    Layer * conv_layer1;
    Layer * pool_layer1;
    Layer * filter1[5];
    Fcnn_layer * fcnn_input;
    Fcnn_layer * fcnn_w;
    Fcnn_layer * fcnn_output;
} Network;

Network * Network_() {
    Network * CNN = (Network *)malloc(sizeof(Network));
    CNN->Input_layer = Layer_();
    CNN->conv_layer1 = Layer_();
    CNN->pool_layer1 = Layer_();
    for (int i = 0; i < 5; i++) CNN->filter1[i] = Layer_();
    CNN->fcnn_input = Fcnn_layer_();
    CNN->fcnn_w = Fcnn_layer_();
    CNN->fcnn_output = Fcnn_layer_();
    return CNN;
}

Network * CNN;


_type ReLU(_type x) {
    return _max(0.0, x);
}

_type sigmoid(_type x) {
    return 1.0 / (1.0 + exp(-x));
}

/**
 * 卷积函数，表示卷积层A与number个filterB相卷积
 * @param A         输入层
 * @param B         卷积核数组
 * @param number    卷积核的个数
 * @param C         得到的卷积结果
 * @return
 */

void conv(Layer * A, Layer * B[], int number, Layer * C) {
    memset(C->values, 0, sizeof(C->values));

    for (int i = 0; i < number; ++ i) {
        B[i]->L = B[i]->W = 5; B[i]->H = 1;
    }
    C->L = A->L - B[0]->L + 1;
    C->W = A->W - B[0]->W + 1;
    C->H = number;

    for (int num = 0; num < number; ++ num) {
        for (int i = 0; i < C->L; ++ i) {
            for (int j = 0; j < C->W; ++ j) {
                for (int a = 0; a < B[0]->L; ++ a)
                    for (int b = 0; b < B[0]->W; ++ b)
                        for (int k = 0; k < A->H; ++ k)
                            C->values[i][j][num] += A->values[i + a][j + b][k] * B[num]->values[a][b][k];
                C->values[i][j][num] = ReLU(C->values[i][j][num] + C->bias[i][j][num]);
            }
        }
    }
}

/**
 * 将图像输入卷积层
 * @param num   样本序号
 * @param A     输入层
 * @param flag  0代表训练，1代表测试
 * @return
 */
void convInput(int index, Layer * input, Vector2D images) {
    input->L = input->W = 28; input->H = 1;
    int x = 0;
    for (int k = 0; k < input->H; ++ k)
        for (int i = 0; i < input->L; ++ i)
            for (int j = 0; j < input->W; ++ j)
                input->values[i][j][k] = images.data[index][x ++];
}

/**
 * 将卷积提取特征输入到全连接神经网络
 * @param A 池化层的输出
 * @param B 全连接层的输入
 * @return
 */
void FCInput(Layer * A, Fcnn_layer * B) {
    int x = 0;
    for (int i = 0; i < A->L; ++ i)
        for (int j = 0; j < A->W; ++ j)
            for (int k = 0; k < A->H; ++ k)

                B->values[x ++] = sigmoid(A->values[i][j][k]);
    B->length = x;
}


void pool_input(Layer * A, Fcnn_layer * B) {

    int x = 0;
    for (int i = 0; i < A->L; ++ i)
        for (int j = 0; j < A->W; ++ j)
            for (int k = 0; k < A->H; ++ k)
                A->deltas[i][j][k] = B->deltas[x ++];
}

void pool_delta(Layer * A, Layer * B) {
    for (int k = 0; k < A->H; ++ k) {
        for (int i = 0; i < A->L; i += 2) {
            for (int j = 0; j < A->W; j += 2) {

                if (fabs(A->values[i][j][k] - B->values[i / 2][j / 2][k]) < 0.001)
                    A->deltas[i][j][k] = B->deltas[i / 2][j / 2][k];
                else A->deltas[i][j][k] = 0;
            }
        }
    }
}
/**
 * 2x2的滤波器最大池化
 * @param conv_layer 卷积层输出
 * @param A 池化层输出
 * @return
 */

void maxPooling(Layer * conv_layer, Layer * A) {
    A->L = conv_layer->L / 2;
    A->W = conv_layer->W / 2;
    A->H = conv_layer->H;
    for (int k = 0; k < conv_layer->H; ++ k)
        for (int i = 0; i < conv_layer->L; i += 2)
            for (int j = 0; j < conv_layer->W; j += 2)
                A->values[i / 2][j / 2][k] = _max(_max(conv_layer->values[i][j][k], conv_layer->values[i + 1][j][k]),
                                                  _max(conv_layer->values[i][j + 1][k],conv_layer->values[i + 1][j + 1][k]));
}
/**
 * 全连接层参数乘法
 * @param A 输入
 * @param B 参数矩阵
 * @param C 输出
 * @return
 */
void FCMultiply(Fcnn_layer * A, Fcnn_layer * B, Fcnn_layer * C) {
    memset(C->values, 0, sizeof(C->values));
    C->length = out;
    for (int i = 0; i < C->length; ++ i) {
        for (int j = 0; j < A->length; ++ j) {
            C->values[i] += B->weights[i][j] * A->values[j];
        }
        C->values[i] += B->bias[i];
    }
}

_type sum(Layer * A, int z) {
    _type a = 0;
    for (int i = 0; i < A->L; ++ i)
        for (int j = 0; j < A->W; ++ j)
            a += A->deltas[i][j][z];
    return a;
}

_type sum1(Layer * A, Layer * B, int x, int y, int z) {
    _type a = 0;
    for (int i = 0; i < A->L; ++ i)
        for (int j = 0; j < A->W; ++ j)
            a += A->deltas[i][j][z] * B->values[i + x][j + y][0];
    return a;
}

void Update(Layer * A, Layer * B, Layer * C, int z) {
    _type sum_C = sum(C, z);
    for (int i = 0; i < A->L; ++ i) {
        for (int j = 0; j < A->W; ++ j) {
            for (int k = 0; k < A->H; ++ k) {
                A->values[i][j][k] -= alpha() * sum1(C, B, i, j, z);
                C->bias[i][j][k] -= alpha() * sum_C;
            }
        }
    }
}

/**
 * softmax函数
 * @param A
 * @return
 */

void softmax(Fcnn_layer * A) {
    _type sum = 0.0; _type maxi = -100000000;
    for (int i = 0; i < out; ++ i) maxi = _max(maxi,A->values[i]);
    for (int i = 0; i < out; ++ i) sum += exp(A->values[i] - maxi);
    for (int i = 0; i < out; ++ i) A->values[i] = exp(A->values[i] - maxi) / sum;
}
/**
 * 做一次前向输出
 * @param num   样本序号
 * @param flag  0代表训练，1代表测试
 */
void forePropagation(Network * CNN, int index, Vector2D images) {
    convInput(index, CNN->Input_layer, images);
    conv(CNN->Input_layer, CNN->filter1, 5, CNN->conv_layer1);
    maxPooling(CNN->conv_layer1, CNN->pool_layer1);
    FCInput(CNN->pool_layer1, CNN->fcnn_input);
    FCMultiply(CNN->fcnn_input, CNN->fcnn_w, CNN->fcnn_output);
    softmax(CNN->fcnn_output);
}

void backPropagation(Network * CNN, int label) {
    for (int i = 0; i < out; ++ i) {
        CNN->fcnn_output->deltas[i] = CNN->fcnn_output->values[i];
        if (i == label) CNN->fcnn_output->deltas[i] -= 1.0;
    }
    memset(CNN->fcnn_input->deltas, 0, sizeof(CNN->fcnn_input->deltas));
    for (int i = 0; i < CNN->fcnn_input->length; ++ i) {
        for (int j = 0; j < out; ++ j) {
            CNN->fcnn_input->deltas[i] += CNN->fcnn_input->values[i] * (1.0 - CNN->fcnn_input->values[i])
                * CNN->fcnn_w->weights[j][i] * CNN->fcnn_output->deltas[j];
        }
    }
    for (int i = 0; i < CNN->fcnn_input->length; ++ i) {
        for (int j = 0; j < out; ++ j) {
            CNN->fcnn_w->weights[j][i] -= alpha() * CNN->fcnn_output->deltas[j] * CNN->fcnn_input->values[i];
            CNN->fcnn_w->bias[j] -= alpha() * CNN->fcnn_output->deltas[j];
        }
    }
    pool_input(CNN->pool_layer1, CNN->fcnn_input);
    pool_delta(CNN->conv_layer1, CNN->pool_layer1);
    for (int i = 0; i < 5; ++ i)
        Update(CNN->filter1[i], CNN->Input_layer, CNN->conv_layer1, i);
}

int predict(Network * CNN, int t) {
    forePropagation(CNN, t, images_test);
    int ans = -1;
    _type sign = -1;
    for (int i = 0; i < out; ++ i) {
        if (CNN->fcnn_output->values[i] > sign) {
            sign = CNN->fcnn_output->values[i];
            ans = i;
        }
    }
    return ans;
}

double test(Network * CNN) {
    int sum = 0;
    for (int i = 0; i < test_number; ++ i) {
        if (predict(CNN, i) == (int)labels_test.data[i]) sum ++;
    }
    return 1.0 * sum / test_number;
}

void train() {

    printf("Begin Training ...\n");
    for (step = 0; step < epoch_num; ++ step) {
        _type err = 0;
        for (int i = 0; i < train_num; i ++) {
            forePropagation(CNN, i, images_train);
            err -= log(CNN->fcnn_output->values[(int)labels_train.data[i]]);
            backPropagation(CNN, (int)labels_train.data[i]);
        }
        printf("step: %3d loss: %.5f prec: %.5f\n", step, err / train_num, test(CNN));
    }
}

int main() {

    srand((unsigned)time(NULL));

    CNN = Network_();

    labels_train = read_Mnist_Label("../dataset/train-labels.idx1-ubyte");
    images_train = read_Mnist_Images("../dataset/train-images.idx3-ubyte");
    labels_test = read_Mnist_Label("../dataset/t10k-labels.idx1-ubyte");
    images_test = read_Mnist_Images("../dataset/t10k-images.idx3-ubyte");

    for (int i = 0; i < images_train.rows; ++ i)
        for (int j = 0; j < images_train.cols; ++ j)
            images_train.data[i][j] /= 255.0;
    for (int i = 0; i < images_test.rows; ++ i)
        for (int j = 0; j < images_test.cols; ++ j)
            images_test.data[i][j] /= 255.0;

    train();

    labels_train.destroy(&labels_train);
    images_train.destroy(&images_train);
    labels_test.destroy(&labels_test);
    images_test.destroy(&images_test);
    return 0;
}
