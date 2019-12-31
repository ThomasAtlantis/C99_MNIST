//
// Created by MAC on 2019/12/31.
//

#include <stdio.h>
#include "../include/dataio.h"
#include "../include/vector.h"

int main() {
    int index = 0, label;
    Vector1D labels_test = read_Mnist_Label("../dataset/t10k-labels.idx1-ubyte");
    Vector2D images_test = read_Mnist_Images("../dataset/t10k-images.idx3-ubyte");
    FILE * fp = fopen("../image.mem", "wb");
    label = (int)labels_test.data[index];
    fwrite((char *)images_test.data[index], sizeof(_type), 28 * 28, fp);
    fwrite((char *)&label, sizeof(int), 1, fp);
    fclose(fp);
    return 0;
}