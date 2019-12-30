//
// Created by MAC on 2019/12/31.
//

#include <stdio.h>
#include <string.h>
#include "../include/vector.h"
#include "../include/dataio.h"
#include "../include/network.h"
#include "../include/model.h"

#define ParamError do{printf("Parameter syntax error!\n");return 0;}while(0)

const char * chs = ".`\\\":I!>+~[{)|/frnvzYJL0Zwpbh#W%@";
int test_num  = 4000; // 测试样本数
int show_flag = 0;

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

int test(Network * CNN) {
    int sum = 0;
    for (int i = 0; i < test_num; ++ i) {
        int ans = predict(CNN, i), fact = (int)labels_test.data[i];
        if (show_flag) {
            printf("Index: %5d\n", i);
            int x = 0;
            for (int a = 0; a < 28; ++ a) {
                for (int b = 0; b < 28; ++ b)
//                    printf("%3d ", (int)(images_test.data[i][x ++] * 255));
                    printf("%c ", chs[(int)(images_test.data[i][x ++] * 32)]);
                printf("\n");
            }
            printf("Ground Truth: %2d, Result Predicted: %2d: ", fact, ans);
            printf(ans == fact ? "CORRECT\n": "  WRONG\n");
            if (i != test_num - 1) {
                getchar();
                for (int j = 0; j < 3999; ++ j) printf("\b");
            }
        }
        if (ans == fact) sum ++;
    }
    return sum;
}

int main(int argc, char * argv[]) {

    for (int i = 1; i < argc; ++ i) {
        if (strcmp(argv[i], "--show") == 0) {
            show_flag = 1;
        } else if (strcmp(argv[i], "--num") == 0) {
            if (i == argc - 1) ParamError;
            for (int j = 0; j < strlen(argv[i + 1]); ++ j)
                if (! (argv[i + 1][j] >= '0' && argv[i + 1][j] <= '9')) ParamError;
            test_num = atoi(argv[++ i]);
        }
    }

    // 初始化CNN存储结构
    Network * CNN = loadNetwork("../model.sav");

    // 读取数据集
    labels_test = read_Mnist_Label("../dataset/t10k-labels.idx1-ubyte");
    images_test = read_Mnist_Images("../dataset/t10k-images.idx3-ubyte");

    // 对图片像素进行归一化
    for (int i = 0; i < images_test.rows; ++ i)
        for (int j = 0; j < images_test.cols; ++ j)
            images_test.data[i][j] /= 255.0;
    int sum = test(CNN);
    printf("Total: %5d, Correct: %5d, Precision: %f\n",
        test_num, sum, 1.0 * sum / test_num);

    // 释放内存
    delete_p(CNN);
    delete(labels_test);
    delete(images_test);
    return 0;
}
