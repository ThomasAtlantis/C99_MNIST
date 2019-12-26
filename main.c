#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "vector.h"

#define _type double

#define _max(a,b) (((a)>(b))?(a):(b))
#define _min(a,b) (((a)>(b))?(b):(a))

const int train_number = 20000; // 训练样本数
const int test_number = 5000; // 测试样本数
const int out = 10; // 分类种类数
const _type alpha = 0.01; // 学习率
int step;

Vector1D labels_train;
Vector2D images_train;
Vector1D labels_test;
Vector2D images_test;

typedef struct { // 定义卷积网络中的层
    // L, W, H 分别代表m三个维度的大小
    // L x W x H的卷积核，H是通道数
    // m应该是参数矩阵，b可能是偏置吧，delta是梯度下降的变化量
    int L, W, H;
    _type m[30][30][5];
    _type b[30][30][5];
    _type delta[30][30][5];
} Layer;
Layer * Layer_() {
    Layer * layer = (Layer *)malloc(sizeof(Layer));
    for (int i = 0; i < 30; ++ i)
        for (int j = 0; j < 30; ++j)
            for (int k = 0; k < 5; ++k)
                layer->m[i][j][k] = 0.01 * (rand() % 100);
    return layer;
}
typedef struct { //定义全连接网络中的层
    int length;
    _type m[1000]; // 貌似是分类结果
    _type b[1000];
    _type delta[1000];
    _type w[20][1000];
} Fcnn_layer;
Fcnn_layer * Fcnn_layer_() {
    Fcnn_layer * fcnn_layer = (Fcnn_layer *)malloc(sizeof(Fcnn_layer));
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 1000; ++j)
            fcnn_layer->w[i][j] = 0.01 * (rand() % 100);
    return fcnn_layer;
}
typedef struct { //定义CNN
    Layer * Input_layer;
    Layer * conv_layer1;
    Layer * pool_layer1;
    Layer * filter1[5];
    Fcnn_layer * fcnn_input;
    Fcnn_layer * fcnn_w;
    Fcnn_layer * fcnn_output;
} Network;

Network * Network_() { //权重初始化
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

Layer * conv(Layer * A, Layer * B[], int number, Layer * C);
Layer * CNN_Input(int num, Layer * A, int flag);
Fcnn_layer * Classify_input(Layer * A, Fcnn_layer * B);//将卷积提取特征输入到全连接神经网络
Layer * pool_input(Layer * A, Fcnn_layer * B);//全连接层的误差项传递到CNN中
Layer * pool_delta(Layer * A, Layer * B);//当前层为池化层的敏感项传递
Fcnn_layer * softmax(Fcnn_layer * A);//softmax函数
_type Relu(_type x);//Relu函数
Layer * Update(Layer * A, Layer * B, Layer * C);//filter更新
Layer * maxpooling(Layer * conv_layer, Layer * A);//池化前向输出
Fcnn_layer * fcnn_Mul(Fcnn_layer * A, Fcnn_layer * B, Fcnn_layer * C);//全连接层前向输出
_type sum(Layer * A);//矩阵求和，此处用于敏感项求和
_type sum1(Layer * A, Layer * B, int x, int y);
_type sigmod(_type x);
void test();
int test_out(int t);

/**************************此段为读取MNIST数据集模块**************/
int ReverseInt(int i) {
    unsigned char ch1 = i & 255;
    unsigned char ch2 = i >> 8 & 255;
    unsigned char ch3 = i >> 16 & 255;
    unsigned char ch4 = i >> 24 & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
Vector1D read_Mnist_Label(const char * fileName) {
    Vector1D labels = Vector1D_();
    FILE* file = fopen(fileName, "rb");
    assert(file != NULL);
    if (file) {
        int magic_number = 0, number_of_images = 0;
        fread(&magic_number, sizeof(magic_number), 1, file);
        fread(&number_of_images, sizeof(number_of_images), 1, file);
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        labels.new(&labels, number_of_images);
        for (int i = 0; i < number_of_images; i++) {
            unsigned char label = 0;
            fread(&label, sizeof(label), 1, file);
            labels.data[i] = (_type)label;
        }
    }
    return labels;
}
Vector2D read_Mnist_Images(const char * fileName) {
    Vector2D images = Vector2D_();
    FILE* file = fopen(fileName, "rb");
    assert(file != NULL);
    if (file) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        fread(&magic_number, sizeof(magic_number), 1, file);
        fread(&number_of_images, sizeof(number_of_images), 1, file);
        fread(&n_rows, sizeof(n_rows), 1, file);
        fread(&n_cols, sizeof(n_cols), 1, file);
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        n_rows = ReverseInt(n_rows);
        n_cols = ReverseInt(n_cols);
        images.new(&images, number_of_images, n_rows * n_cols);
        for (int i = 0; i < number_of_images; i++) {
            for (int r = 0; r < n_rows; r++) {
                for (int c = 0; c < n_cols; c++) {
                    unsigned char image = 0;
                    fread(&image, sizeof(image), 1, file);
                    images.data[i][r * n_rows + c] = (_type)image;
                }
            }
        }
    }
    return images;
}
/**************************************************************/

/**
 * 卷积函数，表示卷积层A与number个filterB相卷积
 * @param A         输入层
 * @param B         卷积核数组
 * @param number    卷积核的个数
 * @param C         得到的卷积结果
 * @return
 */
// CNN->conv_layer1 = conv(CNN->Input_layer, CNN->filter1, 5, CNN->conv_layer1);
Layer * conv(Layer * A, Layer * B[], int number, Layer * C) {
    memset(C->m, 0, sizeof(C->m));
    // 5个5x5x1的卷积核
    for (int i = 0; i < number; ++ i) {
        B[i]->L = B[i]->W = 5; B[i]->H = 1;
    }
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
                            C->m[i][j][num] += A->m[i + a][j + b][k] * B[num]->m[a][b][k];
                C->m[i][j][num] = Relu(C->m[i][j][num] + C->b[i][j][num]);
            }
        }
    }
    return C;
}

/**
 * 将图像输入卷积层
 * @param num   样本序号
 * @param A     输入层
 * @param flag  0代表训练，1代表测试
 * @return
 */
Layer * CNN_Input(int num, Layer * A, int flag) {
    A->L = A->W = 28; A->H = 1;
    int x = 0;
    // 数据集中的data是一维存储的，需要变回二维（实际是三维，通道那一维的长度为1）
    // TODO: 统一reshape模块
    for (int i = 0; i < 28; ++ i) {
        for (int j = 0; j < 28; ++ j) {
            for (int k = 0; k < 1; ++ k) {
                if (flag == 0) A->m[i][j][k] = images_train.data[num][x];
                else if (flag == 1) A->m[i][j][k] = images_test.data[num][x];
                x++;
            }
        }
    }
    return A;
}

/**
 * 将卷积提取特征输入到全连接神经网络
 * @param A 池化层的输出
 * @param B 全连接层的输入
 * @return
 */
Fcnn_layer * Classify_input(Layer * A, Fcnn_layer * B) {
    int x = 0;
    // 这里又是一个reshape操作，把三维参数展成一维的
    // TODO: B->m[0] = 1.0是什么意思
    B->m[x ++] = 1.0;
    for (int i = 0; i < A->L; ++ i)
        for (int j = 0; j < A->W; ++ j)
            for (int k = 0; k < A->H; ++ k)
                // TODO: 这里用sigmoid激活会不会不太好
                B->m[x ++] = sigmod(A->m[i][j][k]);
    B->length = x;
    return B;
}

// 全连接层的误差项传递到CNN中
Layer * pool_input(Layer * A, Fcnn_layer * B) {
    // 这里又是一个reshape操作，把fcnn_input层的参数reshape回pool_layer1的维度
    int x = 1;
    for (int i = 0; i < A->L; ++ i)
        for (int j = 0; j < A->W; ++ j)
            for (int k = 0; k < A->H; ++ k)
                A->delta[i][j][k] = B->delta[x ++];
    return A;
}
// 当前层为池化层的敏感项传递
Layer * pool_delta(Layer * A, Layer * B) {
    for (int k = 0; k < A->H; ++ k) {
        for (int i = 0; i < A->L; i += 2) {
            for (int j = 0; j < A->W; j += 2) {
                // 如果输入输出之差的绝对值小于0.01，认为该位置时max，传递delta；否则delta为0，不更新参数
                if (fabs(A->m[i][j][k] - B->m[i / 2][j / 2][k]) < 0.01)
                    A->delta[i][j][k] = B->delta[i / 2][j / 2][k];
                else A->delta[i][j][k] = 0;
            }
        }
    }
    return A;
}
/**
 * 2x2的滤波器最大池化
 * @param conv_layer 卷积层输出
 * @param A 池化层输出
 * @return
 */
// TODO: 池化窗口大小不通用
Layer * maxpooling(Layer * conv_layer, Layer * A) {
    A->L = conv_layer->L / 2;
    A->W = conv_layer->W / 2;
    A->H = conv_layer->H;
    for (int k = 0; k < conv_layer->H; ++ k)
        for (int i = 0; i < conv_layer->L; i += 2)
            for (int j = 0; j < conv_layer->W; j += 2)
                A->m[i / 2][j / 2][k] = _max(_max(conv_layer->m[i][j][k], conv_layer->m[i + 1][j][k]),
                    _max(conv_layer->m[i][j + 1][k],conv_layer->m[i + 1][j + 1][k]));
    return A;
}
/**
 * 全连接层参数乘法
 * @param A 输入
 * @param B 参数矩阵
 * @param C 输出
 * @return
 */
Fcnn_layer * fcnn_Mul(Fcnn_layer * A, Fcnn_layer * B, Fcnn_layer * C) {
    memset(C->m, 0, sizeof(C->m));
    C->length = out;
    for (int i = 0; i < C->length; ++ i) {
        for (int j = 0; j < A->length; ++ j) {
            C->m[i] += B->w[i][j] * A->m[j];
        }
        C->m[i] += B->b[i];
    }
    return C;
}
_type sigmod(_type x) {
    return 1.0 / (1.0 + exp(-x));
}

// 矩阵求和，此处用于敏感项求和
// 注意卷积的偏置是对于每个卷积核而言的
// 5个卷积核，那么偏置就是长度为5的向量
_type sum(Layer * A) {
    _type a = 0;
    for (int i = 0; i < A->L; ++ i)
        for (int j = 0; j < A->W; ++ j)
            for (int k = 0; k < A->H; ++ k)
                a += A->delta[i][j][k];
    return a;
}

// 其实这里就是B->m * A->delta卷积的结果
_type sum1(Layer * A, Layer * B, int x, int y) {
    _type a = 0;
    for (int i = 0; i < A->L; ++ i)
        for (int j = 0; j < A->W; ++ j)
            for (int k = 0; k < A->H; ++ k)
                a += A->delta[i][j][k] * B->m[i + x][j + y][k];
    return a;
}

// filter更新
// TODO: 这里的sum(C)函数明显欠优化
Layer * Update(Layer * A, Layer * B, Layer * C) {
    for (int i = 0; i < A->L; ++ i) {
        for (int j = 0; j < A->W; ++ j) {
            for (int k = 0; k < A->H; ++ k) {
//                A->m[i][j][k] -= alpha * sum1(A, B, i, j);
                A->m[i][j][k] -= alpha * sum1(C, B, i, j);
//                C->b[i][j][k] -= alpha * sum(A);
                C->b[i][j][k] -= alpha * sum(C);
            }
        }
    }
    return A;
}
// ReLU函数
_type Relu(_type x) {
    return _max(0.0, x);
}

/**
 * softmax函数
 * @param A
 * @return
 */
// TODO: 这里的softmax公式减了一个最大值，不知道为啥这么做
Fcnn_layer * softmax(Fcnn_layer * A) {
    _type sum = 0.0; _type maxi = -100000000;
    for (int i = 0; i < out; ++ i) maxi = _max(maxi,A->m[i]);
    for (int i = 0; i < out; ++ i) sum += exp(A->m[i] - maxi);
    for (int i = 0; i < out; ++ i) A->m[i] = exp(A->m[i] - maxi) / sum;
    return A;
}
/**
 * 做一次前向输出
 * @param num   样本序号
 * @param flag  0代表训练，1代表测试
 */
void forward_propagation(int num, int flag) {
    CNN->Input_layer = CNN_Input(num, CNN->Input_layer, flag);
    CNN->conv_layer1 = conv(CNN->Input_layer, CNN->filter1, 5, CNN->conv_layer1);
    CNN->pool_layer1 = maxpooling(CNN->conv_layer1, CNN->pool_layer1);
    //CNN->conv_layer2=conv(CNN->pool_layer1,CNN->filter2,3,CNN->conv_layer2,0);
    //CNN->pool_layer2=maxpooling(CNN->conv_layer2,CNN->pool_layer2);
    CNN->fcnn_input = Classify_input(CNN->pool_layer1, CNN->fcnn_input);
    //for(int i=0;i<CNN->fcnn_input->length;i++) printf("%.5f ",CNN->fcnn_input->m[i]);
    CNN->fcnn_output = fcnn_Mul(CNN->fcnn_input, CNN->fcnn_w, CNN->fcnn_output);
    CNN->fcnn_output = softmax(CNN->fcnn_output);
    // 这个delta原来算的是预测分布与真实分布之差
    for (int i = 0; i < out; ++ i) {
        if (i == (int)labels_train.data[num]) CNN->fcnn_output->delta[i] = CNN->fcnn_output->m[i] - 1.0;
        else CNN->fcnn_output->delta[i] = CNN->fcnn_output->m[i];
        //printf("%.5f ",CNN->fcnn_output->m[i]);
    }
    //printf(" %.0f\n",labels[num]);
}

/**
 * 反向传播算法
 */
void back_propagation() {
    memset(CNN->fcnn_input->delta, 0, sizeof(CNN->fcnn_input->delta));
    for (int i = 0; i < CNN->fcnn_input->length; ++ i) {
        for (int j = 0; j < out; ++ j) {
            CNN->fcnn_input->delta[i] += CNN->fcnn_input->m[i] * (1.0 - CNN->fcnn_input->m[i])
                * CNN->fcnn_w->w[j][i] * CNN->fcnn_output->delta[j];
        }
    }
    for (int i = 0; i < CNN->fcnn_input->length; ++ i) {
        for (int j = 0; j < out; ++ j) {
            CNN->fcnn_w->w[j][i] -= alpha * CNN->fcnn_output->delta[j] * CNN->fcnn_input->m[i];
            CNN->fcnn_w->b[j] -= alpha * CNN->fcnn_output->delta[j];
        }
    }
    CNN->pool_layer1 = pool_input(CNN->pool_layer1, CNN->fcnn_input);
    CNN->conv_layer1 = pool_delta(CNN->conv_layer1, CNN->pool_layer1); // pooling误差传递
    //CNN->pool_layer1=conv(CNN->conv_layer1,CNN->filter2,3,CNN->pool_layer1,1);//conv误差传递
    for (int i = 0; i < 5; ++ i)
        CNN->filter1[i] = Update(CNN->filter1[i], CNN->Input_layer, CNN->conv_layer1);
}
void train() {
    step = 0;
    for(int time = 0; time < 100; ++ time) {
        _type err = 0;
        for(int i = 0; i < train_number; i ++) { // 遍历每一个训练样本
            forward_propagation(i, 0);
            // 交叉熵损失函数，P((int)labels_train.data[i]) = 1, P(其它类别) = 0, 所以只剩下Q(x)
            err -= log(CNN->fcnn_output->m[(int)labels_train.data[i]]);
            back_propagation();
        }
        printf("step: %d   loss:%.5f\n", time, 1.0 * err / train_number);//每次记录一遍数据集的平均误差
        test();
    }
}
void test() {
    int sum=0;
    for(int i=0;i<test_number;i++) {
        int ans=test_out(i);
        int label = (int)labels_test.data[i];
        if(ans==label) sum++;
        //printf("%d %d\n",ans,label);
    }
    //printf("\n");
    //for(int i=0;i<out;i++) printf("%.5f ",data_out[i]);
    //printf("\n");
    printf("step:%d   precision: %.5f\n",++step,1.0*sum/test_number);
}
int test_out(int t) {
    forward_propagation(t,1);
    int ans=-1;
    _type sign=-1;
    for(int i=0;i<out;i++)
    {
        if(CNN->fcnn_output->m[i]>sign)
        {
            sign=CNN->fcnn_output->m[i];
            ans=i;
        }
    }
    return ans;
}

int main() {
    // 初始化CNN存储结构
    CNN = Network_();

    // 读取数据集
    labels_train = read_Mnist_Label("..\\train-labels.idx1-ubyte");
    images_train = read_Mnist_Images("..\\train-images.idx3-ubyte");
    labels_test = read_Mnist_Label("..\\t10k-labels.idx1-ubyte");
    images_test = read_Mnist_Images("..\\t10k-images.idx3-ubyte");

    // 对图片像素进行归一化
    for (int i = 0; i < images_train.rows; ++ i)
        for (int j = 0; j < images_train.cols; ++ j)
            images_train.data[i][j] /= 255.0;
    for (int i = 0; i < images_test.rows; ++ i)
        for (int j = 0; j < images_test.cols; ++ j)
            images_test.data[i][j] /= 255.0;

    // 开始训练
    printf("Begin Training ...\n");
    train();

    // 释放内存
    labels_train.destroy(&labels_train);
    images_train.destroy(&images_train);
    labels_test.destroy(&labels_test);
    images_test.destroy(&images_test);
    return 0;
}