#include <stdio.h>
#include <string.h>
#include <time.h>
#include "../include/vector.h"
#include "../include/dataio.h"
#include "../include/network.h"
#include "../include/model.h"

#define ParamError do{ \
    printf("Parameter syntax error!\n"); \
    return 0; \
}while(0)

int train_num = 6000; // 训练样本数
int test_num  = 600; // 测试样本数
int epoch_num = 300; // 训练轮数
int batch_size = 1;

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

void train(Network * CNN, Alpha * alpha, const char * fileName) {
    double best_score = 0, new_score;
    printf("Begin Training ...\n");
    for (int step = 0; step < epoch_num; ++ step) {
        _type err = 0;
        for (int m = 0; m < train_num / batch_size; m ++) {
            int i = rand() % (train_num / batch_size);
            for (int k = 0; k < class_num; ++ k)
                CNN->fc_output->deltas[k] = 0;
            for (int j = 0; j < batch_size; ++ j) {
                forePropagation(CNN, i * batch_size + j, images_train);
                int label = (int)labels_train.data[i * batch_size + j];
                for (int k = 0; k < class_num; ++ k)
                    CNN->fc_output->deltas[k] += CNN->fc_output->values[k];
                CNN->fc_output->deltas[label] -= 1.0;
                err -= log(CNN->fc_output->values[label]);
            }
            for (int k = 0; k < class_num; ++ k)
                CNN->fc_output->deltas[k] /= batch_size;
            backPropagation(CNN, step, alpha);
        }
        new_score = test(CNN);
        printf("step: %3d loss: %.5f prec: %.5f\n",
            step, err / train_num, new_score);
        if (new_score > best_score) {
            saveNetwork(CNN, fileName);
            best_score = new_score;
        }
    }
    printf("Best Score: %f\nmodel saved to model.sav!\n", best_score);
}

int main(int argc, char * argv[]) {

    srand((unsigned)time(NULL));

    for (int i = 1; i < argc; ++ i) {
        if (!strcmp(argv[i], "--epoch")
            || !strcmp(argv[i], "--test_num")
            || !strcmp(argv[i], "--train_num")) {
            if (i == argc - 1) ParamError;
            for (int j = 0; j < strlen(argv[i + 1]); ++ j)
                if (! (argv[i + 1][j] >= '0' && argv[i + 1][j] <= '9')) ParamError;
            if (!strcmp(argv[i], "--epoch")) epoch_num = atoi(argv[++ i]);
            else if (!strcmp(argv[i], "--test_num")) test_num = atoi(argv[++ i]);
            else if (!strcmp(argv[i], "--train_num")) train_num = atoi(argv[++ i]);
        } else
        if (!strcmp(argv[i], "other")) {
            (void)"pass";
        }
    }

    // 初始化学习率
    Alpha * alpha = ExpDecayLR(0.03, 100, 0.003);

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
    train(CNN, alpha, "../model.sav");

    // 释放内存
    delete_p(CNN);
    delete(labels_train);
    delete(images_train);
    delete(labels_test);
    delete(images_test);
    return 0;
}
