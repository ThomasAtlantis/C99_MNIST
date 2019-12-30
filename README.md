#### 使用说明
下面演示在`linux-Ubuntu`上使用本项目的方法。首先下载并解压，注意将`CMakeLists.txt`与`CMakeLists.bak`交换，其中前者是`Windows10-CLion`上的配置文件。进入`build/`文件夹make一下，可执行文件将生成在`bin/`目录下。
```shell script
wget https://github.com/ThomasAtlantis/C99_MNIST/archive/master.zip
unzip -q master.zip
cd C99_MNIST-master/build/
mv ../CMakeLists.txt ../CMakeLists.tmp
mv ../CMakeLists.bak ../CMakeLists.txt
cmake .. && make
```
结果如下：
```
-- The C compiler identification is GNU 7.4.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: /root/workspace/CWork/C99_MNIST-master/build
Scanning dependencies of target C99_MNIST
[ 25%] Building C object CMakeFiles/C99_MNIST.dir/src/main.c.o
[ 50%] Linking C executable ../bin/C99_MNIST
[ 50%] Built target C99_MNIST
Scanning dependencies of target C99_MNIST_Test
[ 75%] Building C object CMakeFiles/C99_MNIST_Test.dir/src/test.c.o
[100%] Linking C executable ../bin/C99_MNIST_Test
[100%] Built target C99_MNIST_Test
```
使用10000条训练集，1000条测试集，训练10轮：
```shell script
../bin/C99_MNIST --train_num 10000 --test_num 1000 --epoch 10
```
结果如下：
```
Begin Training ...
step:   0 loss: 59.77995 prec: 0.54600
step:   1 loss: 30.09183 prec: 0.49800
step:   2 loss: 23.59912 prec: 0.47800
step:   3 loss: 20.61523 prec: 0.72800
step:   4 loss: 18.55815 prec: 0.78800
step:   5 loss: 17.74338 prec: 0.76200
step:   6 loss: 16.59249 prec: 0.75600
step:   7 loss: 15.75097 prec: 0.76500
step:   8 loss: 15.16759 prec: 0.72200
step:   9 loss: 14.73609 prec: 0.77800
Best Score: 0.788000
model saved to model.sav!
```
使用10条测试集测试一下，使用`--show`参数可以开启字符画显示测试过程。
```shell script
../bin/C99_MNIST_Test --show --num 10
```
结果如下：
```
Index:     9
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . : > n w Z { : . . . . . . . .
. . . . . . . . . . . I z # % % % % % % p v . . . . . .
. . . . . . . . . I z W % % h Y r % w % h h | . . . . .
. . . . . . . . z # % % 0 n ` . . 0 I Z W b % \ . . . .
. . . . . . ` f % % # r ` . . . . p I + % % % \ . . . .
. . . . . . ~ W % h ! . . . . . : / h W % @ Y . . . . .
. . . . . . ~ W % b | n L L J p % h % % % # : . . . . .
. . . . . . ` J W % @ % % % % % % % % % J " . . . . . .
. . . . . . . . \ n Z L v z w % % % % v \ . . . . . . .
. . . . . . . . . . . . . . Y % % W ) . . . . . . . . .
. . . . . . . . . . . . . / % % % { . . . . . . . . . .
. . . . . . . . . . . . ) W % % p . . . . . . . . . . .
. . . . . . . . . . . ! W % % W > . . . . . . . . . . .
. . . . . . . . . . . r % % W + . . . . . . . . . . . .
. . . . . . . . . . ` % % % z . . . . . . . . . . . . .
. . . . . . . . . ` h % % p ` . . . . . . . . . . . . .
. . . . . . . . . ~ @ % % + . . . . . . . . . . . . . .
. . . . . . . . . p % % n . . . . . . . . . . . . . . .
. . . . . . . . . h @ # " . . . . . . . . . . . . . . .
. . . . . . . . . / @ | . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
Ground Truth:  9, Result Predicted:  9: CORRECT
Total:    10, Correct:     9, Precision: 0.900000
```
#### 工程结构
```shell script
C99_MNIST-master/
├── bin
│   ├── C99_MNIST # 训练程序
│   └── C99_MNIST_Test # 测试程序
├── build # 存储编译链接中间文件
├── CMakeLists.bak # Linux Ubuntu下的配置文件
├── CMakeLists.txt # Windows CLion下的配置文件
├── dataset
│   ├── t10k-images.idx3-ubyte  # 一万条测试集图片
│   ├── t10k-labels.idx1-ubyte  # 一万条测试集标签
│   ├── train-images.idx3-ubyte # 十万条训练集图片
│   └── train-labels.idx1-ubyte # 十万条训练集标签
├── include
│   ├── dataio.h  # 读取数据集
│   ├── memtool.h # 内存管理工具
│   ├── model.h   # 模型定义和传播过程
│   ├── mytype.h  # 模型使用的数据类型
│   ├── network.h # 提供神经网络常用结构和函数
│   └── vector.h  # 矩阵数据结构
├── model.sav  # 固化的模型文件
├── README.md  # 文档
└── src
    ├── main.c # 训练主函数
    └── test.c # 测试主函数
```
#### 原始模型
这个工程主要参考了博客[CNN实现MNIST手写数字识别（C++）](https://blog.csdn.net/qq_37141382/article/details/88088781) 中给出的基于C++和STL的代码。经过改进后的本文对应的各个版本在我的[GitHub仓库](https://github.com/ThomasAtlantis/C99_MNIST)。本文设计的模型用于处理经典的手写数字识别问题，使用很工整的MNIST数据集，只是为了理论验证。之前我在学习CNN的过程中也使用PyTorch搭建过Python版本的，在[这里](https://github.com/ThomasAtlantis/NER/tree/master/learningResources/Case/HandwrittenDigitRecognition)。

原始模型结构非常简单，只使用了一个卷积层和一个全连接层。有图有真相：
【占坑】
#### 研究目的
由于作者目前关注神经网络模型的边缘化部署，本工程的目的是为将来在FPGA等更底层的边缘设备上实现CNN做铺垫。当然CNN的训练过程在服务器上进行，推断过程在边缘端进行，那么我们的目的就是开发一个同时支持服务器训练和FPGA推断的项目，或者给出一个通用的研究方法。

原博主是在学习CNN阶段为了加深理解编写的这份代码，使用了最简单的模型结构，也难免有所疏漏。本文将详细探讨偏底层语言编写CNN的过程，并附有传播公式的详细推导过程。希望我的工作能对其代码进行如下几方面的研究和改进（按照顺序依次进行）：
+ 代码一次降级：使用C99标准，不使用STL库
+ 沿着程序思路探究每个函数的功用，推导前向传播和反向传播的计算公式
+ 调试代码中存在的Bug，回归测试是否能够提升准确率
+ 优化代码的可读性、可维护性和可扩展性
+ 优化代码的运行效率，进行精确的性能测试
+ 将代码转移到Linux服务器上运行
+ 修改神经网络结构为经典结构，重新计算公式和复用函数，对其性能和准确率进行评估
+ 将参数矩阵按照某种标准格式导出，并编写解析参数的函数
+ 对模型的前向传播过程分别进行针对性和普遍性的二次降级，使之成为适用于FPGA上HLS的代码
+ 对性能进行二次优化（硬件优化），对卷积等计算密集功能进行模块化测试，最后将模块之间贯穿控制逻辑

#### 前向计算
第一次降级过程我就不说了，对于稍微有代码经验的人都能做到，只要参考仓库的最早的提交就可以。原博主给出了传播过程的公式，却既没有给推导过程，也没有与代码对应起来。推导过程参考了很多资料，主要参考了这篇[宝贝文章](https://blog.csdn.net/qq_16137569/article/details/81449209)，讲的很细致了。

【占坑】
#### 反向传播
反向传播是指【占坑】
##### 误差敏感项的传播
误差敏感项是指【占坑】

**#1 全连接层的输出层误差敏感项**
神经网络的损失函数$C$为交叉熵损失函数，参考[这篇博客](https://blog.csdn.net/chao_shine/article/details/89925762)。设全连接层的第$i$个节点的直接线性输出为$y_i$，经过`softmax`处理的结果是$y_i'$，`ground truth`在类别$i$处的概率是$t_i$，那么输出层的敏感项$\delta_{i}$的计算过程如下：
$$\begin{aligned}
\because C &=-\Sigma_kt_klny_k',y_i'=\frac{e^{y_i}}{\Sigma_k e^{y_k}} \\
\therefore \delta_{i} &= \frac{\partial C}{\partial y_i}
= -t_i\frac{1}{y_i}\frac{\partial y_i'}{\partial y_i} \\
&= -t_i\frac{1}{y_i'}\frac{e^{y_i}(\Sigma_k e^{y_k})-e^{y_i}e^{y_i}}{(\Sigma_ke^{y_k})^2} \\
&=-t_i(1-\frac{e^{y_i}}{\Sigma_ke^{y_k}}) \\
&=y_i'-t_i
\end{aligned}$$
其实由于手写数字识别是一个单目标的多分类问题，所以$t_i$的值，对于$i$若与标签相同，$t_i$概率值为1，否则为0。

**#2 输出池化层/全连接输入层的误差敏感项（未经激活）**
设$z_j$为全连接层输入的第$j$项（未经激活），计算过程：
$$\begin{aligned}
\delta_j&=\frac{\partial C}{\partial z_j}=\sum_{i \in DS(j)}\frac{\partial C}{\partial y_i}\frac{\partial y_i}{\partial z_j} \\
&=\sum_{i \in DS(j)}\delta_i \frac{\partial(\Sigma_kw_{ik}\sigma(z_k)+b_i)}{\partial z_j} \\
&=\sum_{i \in DS(j)}\delta_i \frac{d(\sigma(z_j))}{dz_j}w_{ij}
\end{aligned}$$
其中$DS(j)$的意思是$Downstream(j)$，这个词在一些描述神经网络的文章中也很常见，意思是与节点$j$相连的所有下一层节点组成的集合。这里激活函数使用的是`Sigmoid`函数：$\sigma(x)=1/(1+e^{-x})$，设其倒数为$s(x)$，则其倒数计算过程如下：
$$\begin{aligned}
\because \frac{ds(x)}{dx} &=-e^{-x}=1-s(x) \\
\therefore \frac{d\sigma(x)}{dx} &= -\frac{s'(x)}{s^2(x)}=\frac{s(x)-1}{s^2(x)} \\
&= (\frac{1}{\sigma(x)}-1)\sigma^2(x) \\
&= (1-\sigma(x))\sigma (x)
\end{aligned}$$
设$a_j$为全连接层输入的第j个节点的值（经过激活之后）。将激活函数的导数公式代入敏感项公式中，得到：
$$\begin{aligned}
\delta_j = a_j(1-a_j)\sum_{i \in DS(j)}\delta_iw_{ij}
\end{aligned}$$
**#3 输入池化层的误差敏感项（经过ReLU）**
池化层使用的是`MaxPooling`，所以下一层的敏感项的值会原封不动的传递到上一层最大值所对应的神经元，而其他神经元的敏感项的值都是0，即不会在反向传播的过程中进行更新。设$\delta_k$为池化层输入的第$k$个节点$a_k$的敏感项，$\delta_j$是池化层输出的第$j$个节点$z_j$的敏感项。这里为了简化表示，池化层下标使用了一维表示。
$$ \delta_k^{in}=\left\{
\begin{aligned}
\delta_j^{out}&, a_k^{in}=z_j^{out}\\
0&, otherwise
\end{aligned}
\right.
$$
**#4 （卷积层的）输入层的误差敏感项**
在原始模型的反向传播中实际用不到本层的敏感项，神经网络某一层的敏感项实际是用于上一层的权重和偏置的更新。但为修改神经网络结构做铺垫，还是推导一下。卷积层的推导是CNN中的重点难点，参考博客[卷积神经网络(CNN)反向传播算法](https://www.cnblogs.com/pinard/p/6494810.html)。
$$\begin{aligned}
\delta &= \frac{\partial C}{\partial a^{in}}=(\frac{\partial z^{out}}{\partial a^{in}})^T\frac{\partial C}{\partial z^{out}} \\
&=(\frac{\partial z^{out}}{\partial a^{in}})^T\frac{\partial C}{\partial ReLU(z^{out})}ReLU'(z^{out})\\
&=(\frac{\partial z^{out}}{\partial a^{in}})^T\delta'ReLU'(z^{out})
\end{aligned}$$
只给出这个式子还看不太清晰，举一个简单的例子分析一下。假设我们卷积层的输出$a^{in}$是一个3x3的矩阵，卷积核$W$是一个2x2矩阵，卷积的步长为1，则输出$z^{out}$是一个2x2的矩阵。为了简化假设$b$都为0，则有：
$$\left(\begin{array}{ccc}
    a_{11} & a_{12} & a_{13}\\
    a_{21} & a_{22} & a_{23}\\
    a_{31} & a_{32} & a_{33}\\
\end{array}\right)*
\left(\begin{array}{cc}
    w_{11} & w_{12}\\
    w_{21} & w_{22}\\
\end{array}\right)=
\left(\begin{array}{cc}
    z_{11} & z_{12}\\
    z_{21} & z_{22}\\
\end{array}\right)$$展开之后的形式：$$
z_{11}=a_{11}w_{11}+a_{12}w_{12}+a_{21}w_{21}+a_{22}w_{22} \\
z_{12}=a_{12}w_{11}+a_{13}w_{12}+a_{22}w_{21}+a_{23}w_{22} \\
z_{21}=a_{21}w_{11}+a_{22}w_{12}+a_{31}w_{21}+a_{32}w_{22} \\
z_{22}=a_{22}w_{11}+a_{23}w_{12}+a_{32}w_{21}+a_{33}w_{22}
$$分别计算各项的偏导，这里$\nabla a_{ij}=\delta^{in}_{ij}$，结果如下：
$$\begin{aligned}
\nabla a_{11}&=\delta_{11}w_{11} \\
\nabla a_{12}&=\delta_{11}w_{12}+\delta_{12}w_{11} \\
\nabla a_{13}&=\delta_{12}w_{12} \\
\nabla a_{21}&=\delta_{11}w_{21}+\delta_{21}w_{11} \\
\nabla a_{22}&=\delta_{11}w_{22}+\delta_{12}w_{21}+\delta_{21}w_{12}+\delta_{22}w_{11} \\
\nabla a_{23}&=\delta_{12}w_{22}+\delta_{22}w_{12} \\
\nabla a_{31}&=\delta_{21}w_{21} \\
\nabla a_{32}&=\delta_{21}w_{22}+\delta_{22}w_{21} \\
\nabla a_{33}&=\delta_{22}w_{22}
\end{aligned}$$
上面的式子其实可以用一个矩阵卷积的形式表示，即：
$$\left(\begin{array}{ccc}
	0 & 0 & 0 & 0 \\
    0 & \delta{11} & \delta{12} & 0\\
    0 & \delta{21} & \delta{22} & 0\\
    0 & 0 & 0 & 0 \\
\end{array}\right)*
\left(\begin{array}{cc}
    w_{22} & w_{21}\\
    w_{12} & w_{11}\\
\end{array}\right)=
\left(\begin{array}{ccc}
    \nabla a_{11} & \nabla a_{12} & \nabla a_{13}\\
    \nabla a_{21} & \nabla a_{22} & \nabla a_{23}\\
    \nabla a_{31} & \nabla a_{32} & \nabla a_{33}\\
\end{array}\right)
$$我们可以观察和总结出卷积层输入层的敏感项公式实际为（原博客对于ReLU的导数描述是错的）：
$$
\delta^{in}= pad_1(\delta^{out})*rot_{180}(W)\odot ReLU'(z^{out}) \\
ReLU'(z^{out})=\left\{
\begin{aligned}
1&,z^{out}>0\\
0&, otherwise
\end{aligned}
\right.
$$
##### 权重与偏置的更新
对于神经网络的参数更新，模型使用的是最简单的梯度下降法，其中$\eta$被称作学习率，控制每次参数更新的幅度，也反映了神经网络收敛的速度：
$$\left\{
\begin{aligned}
w&=w-\eta \frac{\partial C}{\partial w} \\
b&=b-\eta \frac{\partial C}{\partial b} \\
\end{aligned}
\right.$$
**#1 全连接层权重的更新**
$$\begin{aligned}
&\because \frac{\partial C}{\partial w_{ji}}=\frac{\partial C}{\partial y_j}\frac{\partial y_j}{\partial w_{ji}}=\delta_jx_{i}\\
&\therefore w_{ji}=w_{ji}-\eta \delta_jx_{i}
\end{aligned}$$
**#2 全连接层偏置的更新**
$$\begin{aligned}
&\because \frac{\partial C}{\partial b_j}=\frac{\partial C}{\partial y_j}\frac{\partial y_j}{\partial b_j}=\delta_j\\
&\therefore b_j=b_j-\eta \delta_j
\end{aligned}$$
**#3 卷积核权重的更新**
假设我们输入$a$是4x4的矩阵，卷积核$W$是3x3的矩阵，输出$z$是2x2的矩阵，那么反向传播的$z$的敏感项$\delta$也是2x2的矩阵。逐项计算可以得到以下四个式子：
$$
\frac{\partial C}{w_{11}}=a_{11}\delta_{11}+a_{12}\delta_{12}+a_{21}\delta_{21}+a_{22}\delta_{22}\\
\frac{\partial C}{w_{12}}=a_{12}\delta_{11}+a_{13}\delta_{12}+a_{22}\delta_{21}+a_{23}\delta_{22}\\
\frac{\partial C}{w_{13}}=a_{13}\delta_{11}+a_{14}\delta_{12}+a_{23}\delta_{21}+a_{24}\delta_{22}\\
\frac{\partial C}{w_{21}}=a_{21}\delta_{11}+a_{22}\delta_{12}+a_{31}\delta_{21}+a_{32}\delta_{22}\\
$$总结其中规律可以发现：$$
\frac{\partial C}{\partial w_{pq}}=\sum^{out.L}_{i=0}\sum^{out.W}_{j=0}\delta^{out}_{ij}x^{in}_{i+p,j+q}
$$其实上式就是卷积的公式，可以写成矩阵卷积的形式：$$
\frac{\partial C}{\partial W}=a^{in}*\delta^{out}
$$

**#4 卷积核偏置的更新**
需要注意的是卷积层的偏置是对于整个卷积核而言的，如下面这个动图（卷积层演示，来自[网站](http://cs231n.github.io/assets/conv-demo/index.html)）所显示的，有几个卷积核，就有几个偏置项，所以卷积层的偏置是一个长度为卷积核数的一维向量。
![卷积层演示](https://img-blog.csdnimg.cn/20191227002511777.gif#pic_center =500x400)
对于第$k$个卷积核，有下式：$$
\frac{\partial C}{\partial b_k}=\sum^{out.L}_{i=0}\sum^{out.W}_{j=0}\frac{\partial C}{\partial z_{ij}}\frac{\partial z_{ij}}{\partial b_k}=\sum^{out.L}_{i=0}\sum^{out.W}_{j=0}\delta^{out}_{ij}
$$