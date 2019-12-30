//
// Created by MAC on 2019/12/31.
//

#include <stdio.h>
#include "../include/vector.h"
#include "../include/dataio.h"
#include "../include/network.h"
#include "../include/model.h"

const int test_num  = 4000; // 测试样本数

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

int main() {
    // 初始化CNN存储结构
    Network * CNN = loadNetwork("../model.sav");

    // 读取数据集
    labels_test = read_Mnist_Label("../dataset/t10k-labels.idx1-ubyte");
    images_test = read_Mnist_Images("../dataset/t10k-images.idx3-ubyte");

    // 对图片像素进行归一化
    for (int i = 0; i < images_test.rows; ++ i)
        for (int j = 0; j < images_test.cols; ++ j)
            images_test.data[i][j] /= 255.0;

    printf("Precision: %f", test(CNN));

    // 释放内存
    delete_p(CNN);
    delete(labels_test);
    delete(images_test);
    return 0;
}
