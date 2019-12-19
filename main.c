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
const int out = 10;
const _type alpha = 0.01; // 学习率
int step;

Vector1D labels;
Vector2D images;
Vector1D labels_test;
Vector2D images_test;

typedef struct { // 定义卷积网络中的层
    int L, W, H; // L x W x H的卷积核，H是通道数
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
    _type m[1000];
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
    Fcnn_layer * fcnn_outpot;
} Network;

Network * Network_() { //权重初始化
    Network * CNN = (Network *)malloc(sizeof(Network));
    CNN->Input_layer = Layer_();
    CNN->conv_layer1 = Layer_();
    CNN->pool_layer1 = Layer_();
    for (int i = 0; i < 5; i++) CNN->filter1[i] = Layer_();
    CNN->fcnn_input = Fcnn_layer_();
    CNN->fcnn_w = Fcnn_layer_();
    CNN->fcnn_outpot = Fcnn_layer_();
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
        unsigned char label;
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

// 卷积函数，表示卷积层A与number个filterB相卷积
Layer * conv(Layer * A, Layer * B[], int number, Layer * C) {
    memset(C->m, 0, sizeof(C->m));
    for (int i = 0; i < number; ++i) {
        B[i]->L = B[i]->W = 5;
        B[i]->H = 1;
    }
    C->L=A->L-B[0]->L+1;
    C->W=A->W-B[0]->W+1;
    C->H=number;
    for(int num=0;num<number;num++)
        for(int i=0;i<C->L;i++)
            for(int j=0;j<C->W;j++)
            {
                for(int a=0;a<B[0]->L;a++)
                    for(int b=0;b<B[0]->W;b++)
                        for(int k=0;k<A->H;k++)
                            C->m[i][j][num]+=A->m[i+a][j+b][k]*B[num]->m[a][b][k];
                C->m[i][j][num]=Relu(C->m[i][j][num]+C->b[i][j][num]);
            }
    return C;
}

// 将图像输入卷积层
Layer * CNN_Input(int num, Layer * A, int flag) {
    A->L = A->W = 28; A->H = 1;
    int x = 0;
    for (int i = 0; i < 28; ++ i) {
        for (int j = 0; j < 28; ++ j) {
            for (int k = 0; k < 1; ++ k) {
                if (flag == 0) A->m[i][j][k] = images.data[num][x];
                else if (flag == 1) A->m[i][j][k] = images_test.data[num][x];
                x++;
            }
        }
    }
    return A;
}

// 将卷积提取特征输入到全连接神经网络
Fcnn_layer * Classify_input(Layer * A,Fcnn_layer * B)
{
    int x=0;
    B->m[x++]=1.0;
    for(int i=0;i<A->L;i++)
        for(int j=0;j<A->W;j++)
            for(int k=0;k<A->H;k++)
                B->m[x++]=sigmod(A->m[i][j][k]);
    B->length=x;
    return B;
}

// 全连接层的误差项传递到CNN中
Layer * pool_input(Layer * A,Fcnn_layer * B)
{
    int x=1;
    for(int i=0;i<A->L;i++)
        for(int j=0;j<A->W;j++)
            for(int k=0;k<A->H;k++)
                A->delta[i][j][k]=B->delta[x++];
    return A;
}
// 当前层为池化层的敏感项传递
Layer * pool_delta(Layer * A,Layer * B)
{
    for(int k=0;k<A->H;k++)
        for(int i=0;i<A->L;i+=2)
            for(int j=0;j<A->W;j+=2)
            {
                if(fabs(A->m[i][j][k]-B->m[i/2][j/2][k])<0.01) A->delta[i][j][k]=B->delta[i/2][j/2][k];
                else A->delta[i][j][k]=0;
            }
    return A;
}
Layer * maxpooling(Layer * conv_layer, Layer * A) {
    A->L=conv_layer->L/2;
    A->W=conv_layer->W/2;
    A->H=conv_layer->H;
    for(int k=0;k<conv_layer->H;k++)
        for(int i=0;i<conv_layer->L;i+=2)
            for(int j=0;j<conv_layer->W;j+=2)
                A->m[i/2][j/2][k]=_max(_max(conv_layer->m[i][j][k],conv_layer->m[i+1][j][k]),
                                     _max(conv_layer->m[i][j+1][k],conv_layer->m[i+1][j+1][k]));
    return A;
}
Fcnn_layer * fcnn_Mul(Fcnn_layer * A,Fcnn_layer * B,Fcnn_layer * C)
{
    memset(C->m,0,sizeof(C->m));
    C->length=out;
    for(int i=0;i<C->length;i++)
    {
        for(int j=0;j<A->length;j++)
        {
            C->m[i]+=B->w[i][j]*A->m[j];
        }
        C->m[i]+=B->b[i];
    }
    return C;
}
_type sigmod(_type x)
{
    return 1.0/(1.0+exp(-x));
}
// 矩阵求和，此处用于敏感项求和
_type sum(Layer * A)
{
    _type a=0;
    for(int i=0;i<A->L;i++)
        for(int j=0;j<A->W;j++)
            for(int k=0;k<A->H;k++)
                a+=A->delta[i][j][k];
    return a;
}
_type sum1(Layer * A,Layer * B,int x,int y)
{
    _type a=0;
    for(int i=0;i<A->L;i++)
        for(int j=0;j<A->W;j++)
            for(int k=0;k<A->H;k++)
                a+=A->delta[i][j][k]*B->m[i+x][j+y][k];
    return a;
}

// filter更新
Layer * Update(Layer * A,Layer * B,Layer * C)
{
    for(int i=0;i<A->L;i++)
        for(int j=0;j<A->W;j++)
            for(int k=0;k<A->H;k++)
            {
                A->m[i][j][k]-=alpha*sum1(A,B,i,j);
                C->b[i][j][k]-=alpha*sum(A);
            }
    return A;
}
// Relu函数
_type Relu(_type x)
{
    return _max(0.0, x);
}
// softmax函数
Fcnn_layer * softmax(Fcnn_layer * A)
{
    _type sum=0.0;_type maxx=-100000000;
    for(int i=0;i<out;i++) maxx=_max(maxx,A->m[i]);
    for(int i=0;i<out;i++) sum+=exp(A->m[i]-maxx);
    for(int i=0;i<out;i++) A->m[i]=exp(A->m[i]-maxx)/sum;
    return A;
}
//做一次前向输出
void forward_propagation(int num, int flag) {
    CNN->Input_layer = CNN_Input(num, CNN->Input_layer, flag);
    CNN->conv_layer1 = conv(CNN->Input_layer, CNN->filter1, 5, CNN->conv_layer1);
    CNN->pool_layer1=maxpooling(CNN->conv_layer1,CNN->pool_layer1);
    //CNN->conv_layer2=conv(CNN->pool_layer1,CNN->filter2,3,CNN->conv_layer2,0);
    //CNN->pool_layer2=maxpooling(CNN->conv_layer2,CNN->pool_layer2);
    CNN->fcnn_input=Classify_input(CNN->pool_layer1,CNN->fcnn_input);
    //for(int i=0;i<CNN->fcnn_input->length;i++) printf("%.5f ",CNN->fcnn_input->m[i]);
    CNN->fcnn_outpot=fcnn_Mul(CNN->fcnn_input,CNN->fcnn_w,CNN->fcnn_outpot);
    CNN->fcnn_outpot=softmax(CNN->fcnn_outpot);
    for (int i = 0; i < out; ++i) {
        if (i == (int)labels.data[num]) CNN->fcnn_outpot->delta[i]=CNN->fcnn_outpot->m[i]-1.0;
        else CNN->fcnn_outpot->delta[i]=CNN->fcnn_outpot->m[i];
        //printf("%.5f ",CNN->fcnn_outpot->m[i]);
    }
    //printf(" %.0f\n",labels[num]);
}
void back_propagation()//反向传播算法
{
    memset(CNN->fcnn_input->delta,0,sizeof(CNN->fcnn_input->delta));
    for(int i=0;i<CNN->fcnn_input->length;i++)
    {
        for(int j=0;j<out;j++)
        {
            CNN->fcnn_input->delta[i]+=CNN->fcnn_input->m[i]*(1.0-CNN->fcnn_input->m[i])*CNN->fcnn_w->w[j][i]*CNN->fcnn_outpot->delta[j];
        }
    }
    for(int i=0;i<CNN->fcnn_input->length;i++)
    {
        for(int j=0;j<out;j++)
        {
            CNN->fcnn_w->w[j][i]-=alpha*CNN->fcnn_outpot->delta[j]*CNN->fcnn_input->m[i];
            CNN->fcnn_w->b[j]-=alpha*CNN->fcnn_outpot->delta[j];
            //printf("%.5f ",CNN->fcnn_w->w[j][i]);
        }
    }
    CNN->pool_layer1=pool_input(CNN->pool_layer1,CNN->fcnn_input);
    CNN->conv_layer1=pool_delta(CNN->conv_layer1,CNN->pool_layer1);//pooling误差传递
    //CNN->pool_layer1=conv(CNN->conv_layer1,CNN->filter2,3,CNN->pool_layer1,1);//conv误差传递
    for(int i=0;i<5;i++) CNN->filter1[i]=Update(CNN->filter1[i],CNN->Input_layer,CNN->conv_layer1);
}
void train() {
    step = 0;
    for(int time = 0; time < 100; ++ time) {
        _type err = 0;
        for(int i = 0; i < train_number; i ++) { // 遍历每一个训练样本
            forward_propagation(i, 0);
            err -= log(CNN->fcnn_outpot->m[(int)labels.data[i]]);
            back_propagation();
        }
        printf("step: %d   loss:%.5f\n", time, 1.0 * err / train_number);//每次记录一遍数据集的平均误差
        test();
    }
}
void test() {
    int sum=0;
    for(int i=0;i<test_number;i++)
    {
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
        if(CNN->fcnn_outpot->m[i]>sign)
        {
            sign=CNN->fcnn_outpot->m[i];
            ans=i;
        }
    }
    return ans;
}

int main() {
    // 初始化CNN存储结构
    CNN = Network_();

    // 读取数据集
    labels = read_Mnist_Label("..\\train-labels.idx1-ubyte");
    images = read_Mnist_Images("..\\train-images.idx3-ubyte");
    labels_test = read_Mnist_Label("..\\t10k-labels.idx1-ubyte");
    images_test = read_Mnist_Images("..\\t10k-images.idx3-ubyte");

    // 对图片像素进行归一化
    for (int i = 0; i < images.rows; ++ i)
        for (int j = 0; j < images.cols; ++ j)
            images.data[i][j] /= 255.0;
    for (int i = 0; i < images_test.rows; ++ i)
        for (int j = 0; j < images_test.cols; ++ j)
            images_test.data[i][j] /= 255.0;

    // 开始训练
    printf("Begin Training ...\n");
    train();

    // 释放内存
    labels.destroy(&labels);
    images.destroy(&images);
    labels_test.destroy(&labels_test);
    images_test.destroy(&images_test);
    return 0;
}