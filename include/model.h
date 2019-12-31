//
// Created by MAC on 2019/12/28.
//

#ifndef CNN_MODEL_H
#define CNN_MODEL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/network.h"

#define filter_num 5
#define class_num 10

// 定义Network结构
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

void saveNetwork(Network * CNN, const char *fileName) {
    FILE * fp = fopen(fileName, "wb");
    if (!fp) {printf("Failed to write file!\n"); return;}
    for (int i = 0; i < filter_num; ++ i)
        saveFilter(fp, CNN->filter[i]);
    saveFCWeight(fp, CNN->fc_weight);
    fclose(fp);
}

Network * loadNetwork(const char * fileName) {
    FILE * fp = fopen(fileName, "rb");
    Network * CNN = (Network *)malloc(sizeof(Network));
    CNN->destroy = killNetwork;
    CNN->input_layer = Convol2D_(1, 28, 28);
    for (int i = 0; i < filter_num; ++ i)
        CNN->filter[i] = loadFilter(fp);
    CNN->conv_layer = Convol2D_(5, 24, 24);
    CNN->pool_layer = Convol2D_(5, 12, 12);
    CNN->fc_input = FCLayer_(720);
    CNN->fc_weight = loadFCWeight(fp);
    CNN->fc_output = FCLayer_(10);
    fclose(fp);
    return CNN;
}

Network * Network_() {
    Network * CNN = (Network *)malloc(sizeof(Network));
    CNN->destroy = killNetwork;
    CNN->input_layer = Convol2D_(1, 28, 28);
    for (int i = 0; i < filter_num; ++ i)
        CNN->filter[i] = Filter2D_(1, 5, 5);
    CNN->conv_layer = Convol2D_(5, 24, 24);
    CNN->pool_layer = Convol2D_(5, 12, 12);
    CNN->fc_input = FCLayer_(720);
    CNN->fc_weight = FCWeight_(720, 10);
    CNN->fc_output = FCLayer_(10);
    return CNN;
}

/**
 * 将从文件中读出的图像reshape到输入层
 * @param index  样本序号
 * @param A      神经网络的输入层
 * @param isTest 0代表训练，1代表测试
 */
void convInput(int index, CVLayer * input, Vector2D images) {
    int x = 0;
    for (int k = 0; k < input->H; ++ k)
        for (int i = 0; i < input->L; ++ i)
            for (int j = 0; j < input->W; ++ j)
                input->values[k][i][j] = images.data[index][x ++];
}

/**
 * input与filter做卷积
 * @param input     输入层
 * @param filter    卷积核数组
 * @param conv      卷积层存储结果
 * 默认步长值为1
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
 * shape=2x2, stride=2 最大池化
 * @param conv 卷积层输出
 * @param pool 池化层输出
 */
// TODO: 池化窗口大小不通用
void maxPooling(CVLayer * conv, CVLayer * pool) {
    for (int k = 0; k < conv->H; ++ k)
        for (int i = 0; i < conv->L; i += 2)
            for (int j = 0; j < conv->W; j += 2)
                pool->values[k][i / 2][j / 2] = _max(
                    _max(conv->values[k][i][j], conv->values[k][i][j + 1]),
                    _max(conv->values[k][i + 1][j],conv->values[k][i + 1][j + 1])
                );
}

/**
 * 池化层的输出reshape到全连接神经网络
 * @param pool    池化层的输出
 * @param fcInput 全连接层的输入
 */
void FCInput(CVLayer * pool, FCLayer * fcInput) {
    int x = 0;
    for (int k = 0; k < pool->H; ++ k)
        for (int i = 0; i < pool->L; ++ i)
            for (int j = 0; j < pool->W; ++ j)
                fcInput->values[x ++] = poolActivate(pool->values[k][i][j]);
}

/**
 * 全连接层参数矩阵乘法
 * @param input  输入
 * @param weight 参数矩阵
 * @param output 输出
 * @return
 */
void FCMultiply(FCLayer * input, FCLayer * weight, FCLayer * output) {
    for (int i = 0; i < output->L; ++ i) {
        output->values[i] = 0;
        for (int j = 0; j < input->L; ++ j)
            output->values[i] += weight->weights[i][j] * input->values[j];
        output->values[i] += weight->bias[i];
    }
}

/**
 * 全连接层的误差项reshape回池化层中
 * @param fcInput 全连接的输入层
 * @param pool    池化层
 */
void back_fcIn2pool(FCLayer * fcInput, CVLayer * pool) {
    int x = 0;
    for (int k = 0; k < pool->H; ++ k)
        for (int i = 0; i < pool->L; ++ i)
            for (int j = 0; j < pool->W; ++ j)
                pool->deltas[k][i][j] = fcInput->deltas[x ++];
}
/**
 * 池化层的敏感项传递回卷积层中
 * @param A
 * @param B
 */
void back_pool2conv(CVLayer * pool, CVLayer * conv) {
    for (int k = 0; k < conv->H; ++ k) {
        for (int i = 0; i < conv->L; ++ i) {
            for (int j = 0; j < conv->W; ++ j) {
                // 如果输入输出之差的绝对值小于0.001，认为该位置时max，传递delta；否则delta为0，不更新参数
                if (fabs(conv->values[k][i][j] - pool->values[k][i / 2][j / 2]) == 0)
                    conv->deltas[k][i][j] = pool->deltas[k][i / 2][j / 2];
                else conv->deltas[k][i][j] = 0;
            }
        }
    }
}

// 矩阵求和，此处用于敏感项求和
// 注意卷积的偏置是对于每个卷积核而言的
// 5个卷积核，那么偏置就是长度为5的向量
_type sumMatrix(CVLayer * conv, int index) {
    _type a = 0;
    for (int i = 0; i < conv->L; ++ i)
        for (int j = 0; j < conv->W; ++ j)
            a += conv->deltas[index][i][j];
    return a;
}

// 其实这里就是input->values * conv->delta卷积的结果
_type convMatrix(CVLayer * input, CVLayer * conv, int k, int x, int y, int index) {
    _type a = 0;
    for (int i = 0; i < conv->L; ++ i)
        for (int j = 0; j < conv->W; ++ j)
            a += conv->deltas[index][i][j] * input->values[k][i + x][j + y];
    return a;
}

/**
 * 卷积层反向传播更新卷积核参数
 * @param filter 卷积核
 * @param input  输入层
 * @param conv   卷积层
 * @param index  卷积核序号
 */
void UpdateFilter(CVLayer * filter[], CVLayer * input, CVLayer * conv, double alpha) {
    for (int index = 0; index < filter_num; ++ index) {
        filter[index]->bias -= alpha * sumMatrix(conv, index);
        for (int k = 0; k < filter[index]->H; ++ k)
            for (int i = 0; i < filter[index]->L; ++ i)
                for (int j = 0; j < filter[index]->W; ++ j)
                    filter[index]->values[k][i][j] -= alpha * convMatrix(input, conv, k, i, j, index);
    }
}

/**
 * 做一次反向传播
 * @param CNN   神经网络
 * @param step  时间步
 * @param alpha 学习率参数
 * @param label Ground Truth标签
 */
void backPropagation(Network * CNN, int step, Alpha * alpha) {
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
    back_fcIn2pool(CNN->fc_input, CNN->pool_layer);
    back_pool2conv(CNN->pool_layer, CNN->conv_layer);
    UpdateFilter(CNN->filter, CNN->input_layer, CNN->conv_layer, expDecayLR(alpha, step));
}

/**
 * 做一次前向输出
 * @param CNN    神经网络
 * @param index  训练样本序号
 * @param images 训练样本集合
 */
void forePropagation(Network * CNN, int index, Vector2D images) {
    convInput(index, CNN->input_layer, images);
    conv(CNN->input_layer, CNN->filter, CNN->conv_layer);
    maxPooling(CNN->conv_layer, CNN->pool_layer);
    FCInput(CNN->pool_layer, CNN->fc_input);
    FCMultiply(CNN->fc_input, CNN->fc_weight, CNN->fc_output);
    softmax(CNN->fc_output);
}

#endif //CNN_MODEL_H