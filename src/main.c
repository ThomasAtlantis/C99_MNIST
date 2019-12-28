#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/vector.h"
#include "../include/dataio.h"
#include "../include/network.h"
#include "../include/model.h"
#define _type double

const int train_number = 20000; // 训练样本数
const int test_number = 4000; // 测试样本数
const int epoch = 200; // 训练轮数
const int class = 10; // 分类类别数
int step;

Vector1D labels_train;
Vector2D images_train;
Vector1D labels_test;
Vector2D images_test;

Alpha * alpha;
Network * CNN;

/**
 * 卷积函数，表示卷积层A与number个filterB相卷积
 * @param A         输入层
 * @param B         卷积核数组
 * @param number    卷积核的个数
 * @param C         得到的卷积结果
 * @return
 */
// CNN->conv_layer = conv(CNN->input_layer, CNN->filter, 5, CNN->conv_layer);
void conv(CVLayer * A, CVLayer * B[], int number, CVLayer * C) {
    memset(C->values, 0, sizeof(C->values));
    C->L = A->L - B[0]->L + 1;
    C->W = A->W - B[0]->W + 1;
    C->H = number;
    // TODO: 这里没有考虑步长，默认步长值为1
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
void CNN_Input(int num, CVLayer * A, int flag) {
    A->L = A->W = 28; A->H = 1;
    int x = 0;
    // 数据集中的data是一维存储的，需要变回二维（实际是三维，通道那一维的长度为1）
    // TODO: 统一reshape模块
    for (int i = 0; i < 28; ++ i) {
        for (int j = 0; j < 28; ++ j) {
            for (int k = 0; k < 1; ++ k) {
                if (flag == 0) A->values[i][j][k] = images_train.data[num][x];
                else if (flag == 1) A->values[i][j][k] = images_test.data[num][x];
                x++;
            }
        }
    }
}

/**
 * 将卷积提取特征输入到全连接神经网络
 * @param A 池化层的输出
 * @param B 全连接层的输入
 * @return
 */
void Classify_input(CVLayer * A, FCLayer * B) {
    int x = 0;
    // 这里又是一个reshape操作，把三维参数展成一维的
    // TODO: B->values[0] = 1.0是什么意思
    B->values[x ++] = 1.0;
    for (int i = 0; i < A->L; ++ i)
        for (int j = 0; j < A->W; ++ j)
            for (int k = 0; k < A->H; ++ k)
                // TODO: 这里用sigmoid激活会不会不太好
                B->values[x ++] = sigmoid(A->values[i][j][k]);
    B->L = x;
}

// 全连接层的误差项传递到CNN中
void pool_input(CVLayer * A, FCLayer * B) {
    // 这里又是一个reshape操作，把fc_input层的参数reshape回pool_layer的维度
    int x = 1;
    for (int i = 0; i < A->L; ++ i)
        for (int j = 0; j < A->W; ++ j)
            for (int k = 0; k < A->H; ++ k)
                A->deltas[i][j][k] = B->deltas[x ++];
}
// 当前层为池化层的敏感项传递
void pool_delta(CVLayer * A, CVLayer * B) {
    for (int k = 0; k < A->H; ++ k) {
        for (int i = 0; i < A->L; i += 2) {
            for (int j = 0; j < A->W; j += 2) {
                // 如果输入输出之差的绝对值小于0.01，认为该位置时max，传递delta；否则delta为0，不更新参数
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
// TODO: 池化窗口大小不通用
void maxpooling(CVLayer * conv_layer, CVLayer * A) {
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
void fcnn_Mul(FCLayer * A, FCLayer * B, FCLayer * C) {
    memset(C->values, 0, sizeof(C->values));
    C->L = class;
    for (int i = 0; i < C->L; ++ i) {
        for (int j = 0; j < A->L; ++ j) {
            C->values[i] += B->weights[i][j] * A->values[j];
        }
        C->values[i] += B->bias[i];
    }
}

// 矩阵求和，此处用于敏感项求和
// 注意卷积的偏置是对于每个卷积核而言的
// 5个卷积核，那么偏置就是长度为5的向量
_type sum(CVLayer * A, int z) {
    _type a = 0;
    for (int i = 0; i < A->L; ++ i)
        for (int j = 0; j < A->W; ++ j)
            a += A->deltas[i][j][z];
    return a;
}

// 其实这里就是B->values * A->delta卷积的结果
_type sum1(CVLayer * A, CVLayer * B, int x, int y, int z) {
    _type a = 0;
    for (int i = 0; i < A->L; ++ i)
        for (int j = 0; j < A->W; ++ j)
            a += A->deltas[i][j][z] * B->values[i + x][j + y][z];
    return a;
}

// filter更新
// TODO: 这里的sum(C)函数明显欠优化
void Update(CVLayer * A, CVLayer * B, CVLayer * C, int z) {
    int sum_C = sum(C, z);
    for (int i = 0; i < A->L; ++ i) {
        for (int j = 0; j < A->W; ++ j) {
            for (int k = 0; k < A->H; ++ k) {
                A->values[i][j][k] -= expDecayLR(alpha, step) * sum1(C, B, i, j, z);
                C->bias[i][j][k] -= expDecayLR(alpha, step) * sum_C;
            }
        }
    }
}

/**
 * softmax函数
 * @param A
 * @return
 */
// TODO: 这里的softmax公式减了一个最大值，不知道为啥这么做
void softmax(FCLayer * A) {
    _type sum = 0.0; _type maxi = -100000000;
    for (int i = 0; i < class; ++ i) maxi = _max(maxi,A->values[i]);
    for (int i = 0; i < class; ++ i) sum += exp(A->values[i] - maxi);
    for (int i = 0; i < class; ++ i) A->values[i] = exp(A->values[i] - maxi) / sum;
}
/**
 * 做一次前向输出
 * @param index   样本序号
 * @param isTest  0代表训练，1代表测试
 */
void forePropagation(int index, int isTest) {
    CNN_Input(index, CNN->input_layer, isTest);
    conv(CNN->input_layer, CNN->filter, 5, CNN->conv_layer);
    maxpooling(CNN->conv_layer, CNN->pool_layer);
    Classify_input(CNN->pool_layer, CNN->fc_input);
    fcnn_Mul(CNN->fc_input, CNN->fc_weight, CNN->fc_output);
    softmax(CNN->fc_output);
    for (int i = 0; i < class; ++ i) {
        if (i == (int)labels_train.data[index]) CNN->fc_output->deltas[i] = CNN->fc_output->values[i] - 1.0;
        else CNN->fc_output->deltas[i] = CNN->fc_output->values[i];
    }
}

/**
 * 反向传播算法
 */
void backPropagation() {
    memset(CNN->fc_input->deltas, 0, CNN->fc_input->L * sizeof(_type));
    for (int i = 0; i < CNN->fc_input->L; ++ i) {
        for (int j = 0; j < class; ++ j) {
            CNN->fc_input->deltas[i] += CNN->fc_input->values[i] * (1.0 - CNN->fc_input->values[i])
                * CNN->fc_weight->weights[j][i] * CNN->fc_output->deltas[j];
        }
    }
    for (int i = 0; i < CNN->fc_input->L; ++ i) {
        for (int j = 0; j < class; ++ j) {
            CNN->fc_weight->weights[j][i] -= expDecayLR(alpha, step) * CNN->fc_output->deltas[j] * CNN->fc_input->values[i];
            CNN->fc_weight->bias[j] -= expDecayLR(alpha, step) * CNN->fc_output->deltas[j];
        }
    }
    pool_input(CNN->pool_layer, CNN->fc_input);
    pool_delta(CNN->conv_layer, CNN->pool_layer);
    for (int i = 0; i < 5; ++ i)
        Update(CNN->filter[i], CNN->input_layer, CNN->conv_layer, i);
}

int predict(int t) {
    forePropagation(t, 1);
    // argmax获得预测的类别序号
    int ans = -1; _type sign = -1;
    for (int i = 0; i < class; ++ i) {
        if (CNN->fc_output->values[i] > sign) {
            sign = CNN->fc_output->values[i];
            ans = i;
        }
    }
    return ans;
}

double test() {
    int sum = 0;
    for (int i = 0; i < test_number; ++ i)
        if (predict(i) == (int)labels_test.data[i]) sum ++;
    return 1.0 * sum / test_number;
}

void train() {
    printf("Begin Training ...\n");
    for (step = 0; step < epoch; ++ step) {
        _type err = 0;
        for (int i = 0; i < train_number; i ++) {
            forePropagation(i, 0);
            err -= log(CNN->fc_output->values[(int)labels_train.data[i]]);
            backPropagation();
        }
        printf("step: %3d loss: %.5f prec: %.5f\n",
            step, err / train_number, test());
    }
}

int main() {
    // 初始化学习率
    alpha = ExpDecayLR(0.1, 200, 0.01);

    // 初始化CNN存储结构
    CNN = Network_();

    // 读取数据集
    labels_train = read_Mnist_Label("../dataset/train-labels.idx1-ubyte");
    images_train = read_Mnist_Images("../dataset/train-images.idx3-ubyte");
    labels_test = read_Mnist_Label("../dataset/t10k-labels.idx1-ubyte");
    images_test = read_Mnist_Images("../dataset/t10k-images.idx3-ubyte");

    // 对图片像素进行归一化
    for (int i = 0; i < images_train.rows; ++ i)
        for (int j = 0; j < images_train.cols; ++ j)
            images_train.data[i][j] /= 255.0;
    for (int i = 0; i < images_test.rows; ++ i)
        for (int j = 0; j < images_test.cols; ++ j)
            images_test.data[i][j] /= 255.0;

    // 训练
    train();

    // 释放内存
    // TODO：这里不全
    labels_train.destroy(&labels_train);
    images_train.destroy(&images_train);
    labels_test.destroy(&labels_test);
    images_test.destroy(&images_test);
    return 0;
}
