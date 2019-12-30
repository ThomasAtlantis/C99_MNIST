//
// Created by MAC on 2019/12/28.
//

#ifndef CNN_MEMTOOL_H
#define CNN_MEMTOOL_H

#include "../include/mytype.h"

#define delete_p(x) do{(x)->destroy(x);free_p(x);}while(0)
#define delete(x) (x).destroy(&(x))
#define free_p(x) do{free(x);(x)=NULL;}while(0)

void free1D(_type * sth) {
    free_p(sth);
}

void free2D(_type ** sth, int h) {
    if (!sth) return;
    for (int k = 0; k < h; ++ k) {
        free_p(sth[k]);
    }
    free_p(sth);
}

void free3D(_type *** sth, int h, int l) {
    if (!sth) return;
    for (int k = 0; k < h; ++ k) {
        for (int i = 0; i < l; ++ i) {
            free_p(sth[k][i]);
        }
        free_p(sth[k]);
    }
    free_p(sth);
}

#endif //CNN_MEMTOOL_H
