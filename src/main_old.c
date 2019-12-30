#include <stdio.h>
#include <stdlib.h>
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

Vector1D labels_train;
Vector2D images_train;
Vector1D labels_test;
Vector2D images_test;

const double PI = 3.141592654;

double GaussRand(double mean, double sigma) {
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

CVLayer * Convol2D_(int h, int l, int w) {
    CVLayer * layer = (CVLayer *)malloc(sizeof(CVLayer));
//    layer->destroy = killCVLayer;
    layer->bias = 0;
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
                layer->values[k][i][j] = GaussRand(0.5, 0.1);
            }
        }
    }
    return layer;
}


CVLayer * Filter2D_(int h, int l, int w) {
    CVLayer * layer = (CVLayer *)malloc(sizeof(CVLayer));
//    layer->destroy = killCVLayer;
    layer->bias = 0;
    layer->L = l; layer->W = w; layer->H = h; layer->deltas = NULL;
    layer->values = (_type ***)malloc(h * sizeof(_type **));
    for (int k = 0; k < h; ++ k) {
        layer->values[k] = (_type **) malloc(l * sizeof(_type *));
        for (int i = 0; i < l; ++ i) {
            layer->values[k][i] = (_type *) malloc(w * sizeof(_type));
            for (int j = 0; j < w; ++ j) {
                layer->values[k][i][j] = GaussRand(0.5, 0.1);
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

FCLayer * FCLayer_(int length) {
    FCLayer * layer = (FCLayer *)malloc(sizeof(FCLayer));
//    layer->destroy = killFCLayer;
    layer->values = (_type *)malloc(length * sizeof(_type));
    layer->deltas = (_type *)malloc(length * sizeof(_type));
    layer->L = length; layer->W = 1;
    layer->weights = NULL; layer->bias = NULL;
    return layer;
}

FCLayer * FCWeight_(int from, int to) {
    FCLayer * layer = (FCLayer *)malloc(sizeof(FCLayer));
//    layer->destroy = killFCLayer;
    layer->bias = (_type *)malloc(to * sizeof(_type));
    layer->weights = (_type **)malloc(to * sizeof(_type));
    for (int i = 0; i < to; ++ i) {
        layer->weights[i] = (_type *)malloc(from * sizeof(_type));
        layer->bias[i] = 0.01 * (rand() % 100);
        for (int j = 0; j < from; ++ j) {
            layer->weights[i][j] = GaussRand(0.5, 0.1);
        }
    }
    layer->values = layer->deltas = NULL;
    layer->L = to; layer->W = from;
    return layer;
}

typedef struct {
    CVLayer * Input_layer;
    CVLayer * conv_layer1;
    CVLayer * pool_layer1;
    CVLayer * filter1[5];
    FCLayer * fcnn_input;
    FCLayer * fcnn_w;
    FCLayer * fcnn_output;
} Network;

Network * Network_() {
    Network * CNN = (Network *)malloc(sizeof(Network));
    CNN->Input_layer = Convol2D_(1, 28, 28);
    CNN->conv_layer1 = Convol2D_(5, 24, 24);
    CNN->pool_layer1 = Convol2D_(5, 12, 12);
    for (int i = 0; i < 5; i++)
        CNN->filter1[i] = Filter2D_(1, 5, 5);
    CNN->fcnn_input = FCLayer_(720);
    CNN->fcnn_w = FCWeight_(720, 10);
    CNN->fcnn_output = FCLayer_(10);
    return CNN;
}

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

void conv(CVLayer * A, CVLayer * B[], int number, CVLayer * C) {
    for (int i = 0; i < number; ++ i) {
        B[i]->L = B[i]->W = 5; B[i]->H = 1;
    }
    C->L = A->L - B[0]->L + 1;
    C->W = A->W - B[0]->W + 1;
    C->H = number;

    for (int num = 0; num < number; ++ num) {
        for (int i = 0; i < C->L; ++ i) {
            for (int j = 0; j < C->W; ++ j) {
                C->values[num][i][j] = 0;
                for (int k = 0; k < A->H; ++ k)
                    for (int a = 0; a < B[0]->L; ++ a)
                        for (int b = 0; b < B[0]->W; ++ b)
                            C->values[num][i][j] += A->values[k][i + a][j + b] * B[num]->values[k][a][b];
                C->values[num][i][j] = ReLU(C->values[num][i][j] + B[num]->bias);
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
    input->L = input->W = 28; input->H = 1;
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
    C->L = out;
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
 * softmax函数
 * @param A
 * @return
 */

void softmax(FCLayer * A) {
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

void backPropagation(Network * CNN, int step, Alpha * alpha, int label) {
    for (int i = 0; i < out; ++ i) {
        CNN->fcnn_output->deltas[i] = CNN->fcnn_output->values[i];
        if (i == label) CNN->fcnn_output->deltas[i] -= 1.0;
    }
    for (int i = 0; i < CNN->fcnn_input->L; ++ i) {
        CNN->fcnn_input->deltas[i] = 0;
        for (int j = 0; j < out; ++ j) {
            CNN->fcnn_input->deltas[i] += CNN->fcnn_input->values[i] * (1.0 - CNN->fcnn_input->values[i])
                * CNN->fcnn_w->weights[j][i] * CNN->fcnn_output->deltas[j];
        }
    }
    for (int i = 0; i < CNN->fcnn_input->L; ++ i) {
        for (int j = 0; j < out; ++ j) {
            CNN->fcnn_w->weights[j][i] -= expDecayLR(alpha, step) * CNN->fcnn_output->deltas[j] * CNN->fcnn_input->values[i];
            CNN->fcnn_w->bias[j] -= expDecayLR(alpha, step) * CNN->fcnn_output->deltas[j];
        }
    }
    pool_input(CNN->pool_layer1, CNN->fcnn_input);
    pool_delta(CNN->conv_layer1, CNN->pool_layer1);
    for (int i = 0; i < 5; ++ i)
        Update(CNN->filter1[i], CNN->Input_layer, CNN->conv_layer1, i, expDecayLR(alpha, step));
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

void train(Network * CNN, Alpha * alpha) {
    printf("Begin Training ...\n");
    for (int step = 0; step < epoch_num; ++ step) {
        _type err = 0;
        for (int i = 0; i < train_num; i ++) {
            forePropagation(CNN, i, images_train);
            err -= log(CNN->fcnn_output->values[(int)labels_train.data[i]]);
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

    labels_train.destroy(&labels_train);
    images_train.destroy(&images_train);
    labels_test.destroy(&labels_test);
    images_test.destroy(&images_test);
    return 0;
}
