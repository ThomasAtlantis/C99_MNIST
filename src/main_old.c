#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../include/vector.h"
#include "../include/dataio.h"
#include "../include/memtool.h"
#include "../include/network.h"

#define _max(a,b) (((a)>(b))?(a):(b))
#define _min(a,b) (((a)>(b))?(b):(a))

const int train_num = 100;
const int test_num = 100;
const int epoch_num = 10;

Vector1D labels_train;
Vector2D images_train;
Vector1D labels_test;
Vector2D images_test;

#define filter_num 5
#define class_num 10

typedef struct Network {
    CVLayer * input_layer;
    CVLayer * filter[filter_num];
    CVLayer * conv_layer;
    #define ConvActivate ReLU
    CVLayer * pool_layer;
    #define poolActivate sigmoid
    FCLayer * fc_input;
    FCLayer * fc_weight;
    FCLayer * fc_output;
    void (*destroy)(struct Network *);
} Network;

void killNetwork(Network * this) {
    delete_p(this->input_layer);
    delete_p(this->conv_layer);
    delete_p(this->pool_layer);
    delete_p(this->fc_input);
    delete_p(this->fc_weight);
    delete_p(this->fc_output);
    for (int i = 0; i < filter_num; ++ i)
        delete_p(this->filter[i]);
}

Network * Network_() {
    Network * CNN = (Network *)malloc(sizeof(Network));
    CNN->destroy = killNetwork;
    CNN->input_layer = Convol2D_(1, 28, 28);
    for (int i = 0; i < 5; i++)
        CNN->filter[i] = Filter2D_(1, 5, 5);
    CNN->conv_layer = Convol2D_(5, 24, 24);
    CNN->pool_layer = Convol2D_(5, 12, 12);
    CNN->fc_input = FCLayer_(720);
    CNN->fc_weight = FCWeight_(720, 10);
    CNN->fc_output = FCLayer_(10);
    return CNN;
}

/**
 * 卷积函数，表示卷积层A与filter_num个filterB相卷积
 * @param A         输入层
 * @param B         卷积核数组
 * @param filter_num    卷积核的个数
 * @param C         得到的卷积结果
 * @return
 */
void conv(CVLayer * input, CVLayer * filter[], CVLayer * conv) {
    for (int p = 0; p < conv->H; ++ p) {
        for (int i = 0; i < conv->L; ++ i) {
            for (int j = 0; j < conv->W; ++ j) {
                conv->values[p][i][j] = 0;
                for (int k = 0; k < input->H; ++ k)
                    for (int a = 0; a < filter[0]->L; ++ a)
                        for (int b = 0; b < filter[0]->W; ++ b)
                            conv->values[p][i][j] += input->values[k][i + a][j + b] * filter[p]->values[k][a][b];
                conv->values[p][i][j] = ConvActivate(conv->values[p][i][j] + filter[p]->bias);
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
void convInput(int index, CVLayer * input, Vector2D images) {
    int x = 0;
    for (int k = 0; k < input->H; ++ k)
        for (int i = 0; i < input->L; ++ i)
            for (int j = 0; j < input->W; ++ j)
                input->values[k][i][j] = images.data[index][x ++];
}

/**
 * 将卷积提取特征输入到全连接神经网络
 * @param A 池化层的输出
 * @param B 全连接层的输入
 * @return
 */
void FCInput(CVLayer * A, FCLayer * B) {
    int x = 0;
    for (int k = 0; k < A->H; ++ k)
        for (int i = 0; i < A->L; ++ i)
            for (int j = 0; j < A->W; ++ j)
                B->values[x ++] = sigmoid(A->values[k][i][j]);
    B->L = x;
}


void pool_input(CVLayer * A, FCLayer * B) {

    int x = 0;
    for (int k = 0; k < A->H; ++ k)
        for (int i = 0; i < A->L; ++ i)
            for (int j = 0; j < A->W; ++ j)
                A->deltas[k][i][j] = B->deltas[x ++];
}

void pool_delta(CVLayer * A, CVLayer * B) {
    for (int k = 0; k < A->H; ++ k) {
        for (int i = 0; i < A->L; i += 2) {
            for (int j = 0; j < A->W; j += 2) {
                if (fabs(A->values[k][i][j] - B->values[k][i / 2][j / 2]) < 0.001)
                    A->deltas[k][i][j] = B->deltas[k][i / 2][j / 2];
                else A->deltas[k][i][j] = 0;
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

void maxPooling(CVLayer * conv_layer, CVLayer * A) {
    A->L = conv_layer->L / 2;
    A->W = conv_layer->W / 2;
    A->H = conv_layer->H;
    for (int k = 0; k < conv_layer->H; ++ k)
        for (int i = 0; i < conv_layer->L; i += 2)
            for (int j = 0; j < conv_layer->W; j += 2)
                A->values[k][i / 2][j / 2] = _max(
                    _max(conv_layer->values[k][i][j], conv_layer->values[k][i + 1][j]),
                    _max(conv_layer->values[k][i][j + 1],conv_layer->values[k][i + 1][j + 1])
                );
}
/**
 * 全连接层参数乘法
 * @param A 输入
 * @param B 参数矩阵
 * @param C 输出
 * @return
 */
void FCMultiply(FCLayer * A, FCLayer * B, FCLayer * C) {
    for (int i = 0; i < C->L; ++ i) {
        C->values[i] = 0;
        for (int j = 0; j < A->L; ++ j) {
            C->values[i] += B->weights[i][j] * A->values[j];
        }
        C->values[i] += B->bias[i];
    }
}

_type sum(CVLayer * A, int z) {
    _type a = 0;
    for (int i = 0; i < A->L; ++ i)
        for (int j = 0; j < A->W; ++ j)
            a += A->deltas[z][i][j];
    return a;
}

_type sum1(CVLayer * A, CVLayer * B, int x, int y, int z) {
    _type a = 0;
    for (int i = 0; i < A->L; ++ i)
        for (int j = 0; j < A->W; ++ j)
            a += A->deltas[z][i][j] * B->values[0][i + x][j + y];
    return a;
}

void Update(CVLayer * A, CVLayer * B, CVLayer * C, int z, double alpha) {
    B->bias -= alpha * sum(C, z);
    for (int k = 0; k < A->H; ++ k) {
        for (int i = 0; i < A->L; ++ i) {
            for (int j = 0; j < A->W; ++ j) {
                A->values[k][i][j] -= alpha * sum1(C, B, i, j, z);
            }
        }
    }
}

/**
 * 做一次前向输出
 * @param num   样本序号
 * @param flag  0代表训练，1代表测试
 */
void forePropagation(Network * CNN, int index, Vector2D images) {
    convInput(index, CNN->input_layer, images);
    conv(CNN->input_layer, CNN->filter, CNN->conv_layer);
    maxPooling(CNN->conv_layer, CNN->pool_layer);
    FCInput(CNN->pool_layer, CNN->fc_input);
    FCMultiply(CNN->fc_input, CNN->fc_weight, CNN->fc_output);
    softmax(CNN->fc_output);
}

void backPropagation(Network * CNN, int step, Alpha * alpha, int label) {
    for (int i = 0; i < class_num; ++ i) {
        CNN->fc_output->deltas[i] = CNN->fc_output->values[i];
        if (i == label) CNN->fc_output->deltas[i] -= 1.0;
    }
    for (int i = 0; i < CNN->fc_input->L; ++ i) {
        CNN->fc_input->deltas[i] = 0;
        for (int j = 0; j < class_num; ++ j) {
            CNN->fc_input->deltas[i] += CNN->fc_input->values[i] * (1.0 - CNN->fc_input->values[i])
                * CNN->fc_weight->weights[j][i] * CNN->fc_output->deltas[j];
        }
    }
    for (int i = 0; i < CNN->fc_input->L; ++ i) {
        for (int j = 0; j < class_num; ++ j) {
            CNN->fc_weight->weights[j][i] -= expDecayLR(alpha, step) * CNN->fc_output->deltas[j] * CNN->fc_input->values[i];
            CNN->fc_weight->bias[j] -= expDecayLR(alpha, step) * CNN->fc_output->deltas[j];
        }
    }
    pool_input(CNN->pool_layer, CNN->fc_input);
    pool_delta(CNN->conv_layer, CNN->pool_layer);
    for (int i = 0; i < 5; ++ i)
        Update(CNN->filter[i], CNN->input_layer, CNN->conv_layer, i, expDecayLR(alpha, step));
}

int predict(Network * CNN, int t) {
    forePropagation(CNN, t, images_test);
    int ans = -1;
    _type sign = -1;
    for (int i = 0; i < class_num; ++ i) {
        if (CNN->fc_output->values[i] > sign) {
            sign = CNN->fc_output->values[i];
            ans = i;
        }
    }
    return ans;
}

double test(Network * CNN) {
    int sum = 0;
    for (int i = 0; i < test_num; ++ i) {
        if (predict(CNN, i) == (int)labels_test.data[i]) sum ++;
    }
    return 1.0 * sum / test_num;
}

void train(Network * CNN, Alpha * alpha) {
    printf("Begin Training ...\n");
    for (int step = 0; step < epoch_num; ++ step) {
        _type err = 0;
        for (int i = 0; i < train_num; i ++) {
            forePropagation(CNN, i, images_train);
            err -= log(CNN->fc_output->values[(int)labels_train.data[i]]);
            backPropagation(CNN, step, alpha, (int)labels_train.data[i]);
        }
        printf("step: %3d loss: %.5f prec: %.5f\n", step, err / train_num, test(CNN));
    }
}

int main() {

    srand((unsigned)time(NULL));

    Alpha * alpha = ExpDecayLR(0.1, 200, 0.01);

    Network * CNN = Network_();

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

    train(CNN, alpha);

    delete_p(CNN);
    delete(labels_train);
    delete(images_train);
    delete(labels_test);
    delete(images_test);
    return 0;
}
