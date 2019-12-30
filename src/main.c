#include <stdio.h>
#include "../include/vector.h"
#include "../include/dataio.h"
#include "../include/network.h"
#include "../include/model.h"

const int train_num = 1000; // 训练样本数
const int test_num  = 400; // 测试样本数
const int epoch_num = 10; // 训练轮数

Vector1D labels_train;
Vector2D images_train;
Vector1D labels_test;
Vector2D images_test;

int predict(Network * CNN, int t) {
    forePropagation(CNN, t, images_test);
    // argmax获得预测的类别序号
    int ans = -1; _type sign = -1;
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
    for (int i = 0; i < test_num; ++ i)
        if (predict(CNN, i) == (int)labels_test.data[i]) sum ++;
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
        printf("step: %3d loss: %.5f prec: %.5f\n",
            step, err / train_num, test(CNN));
    }
}

int main() {
    // 初始化学习率
    Alpha * alpha = ExpDecayLR(0.1, 200, 0.01);

    // 初始化CNN存储结构
    Network * CNN = Network_();

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
    train(CNN, alpha);

    // 释放内存
    delete_p(CNN);
    delete(labels_train);
    delete(images_train);
    delete(labels_test);
    delete(images_test);
    return 0;
}
