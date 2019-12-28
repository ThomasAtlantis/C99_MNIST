//
// Created by MAC on 2019/12/28.
//

#ifndef CNN_MODEL_H
#define CNN_MODEL_H

const int filter_num = 5;

typedef struct { //定义CNN
    CVLayer * input_layer;
    CVLayer * conv_layer;
    CVLayer * pool_layer;
    CVLayer * filter[filter_num];
    FCLayer * fc_input;
    FCLayer * fc_weight;
    FCLayer * fc_output;
} Network;

Network * Network_() {
    Network * CNN = (Network *)malloc(sizeof(Network));
    CNN->input_layer = Convol2D_(1, 28, 28);
    CNN->conv_layer = Convol2D_(5, 24, 24);
    CNN->pool_layer = Convol2D_(5, 12, 12);
    for (int i = 0; i < filter_num; ++ i)
        CNN->filter[i] = Filter2D_(1, 5, 5);
    CNN->fc_input = FCLayer_(720);
    CNN->fc_weight = FCWeight_(720, 10);
    CNN->fc_output = FCLayer_(10);
    return CNN;
}

#endif //CNN_MODEL_H
