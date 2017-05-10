#ifndef __POINTARRAY_HPP
#define __POINTARRAY_HPP

#include <stdlib.h>

struct PointArray {
    int width;
    int dim;
    float* elements;
};


// static helper methods

static void mallocPointArray(PointArray& m) {

    m.elements = (float*)malloc(m.width * m.dim * sizeof(float));

}

static void generatePointArray(PointArray& m, int width, int dim)
{

    m.dim = dim;
    m.width = width;
    m.elements = (float*)malloc(m.width * m.dim * sizeof(float) );

}

static void fillPointArrayWithSequence(PointArray& m) {

    for(int i=0;i<m.width*m.dim;i++)
    {
        m.elements[i] = i;
    }

}

static void copyVectorInterval(PointArray& in, int start, int end, PointArray& out) {

    for(int i=0; i < (end-start); i++)
    {
        out.elements[i] = in.elements[i + start];
    }
}

static void copyDimensionToPointArray(PointArray& in, int dim, PointArray& out) {

    for(int i = 0; i<out.width; i++)
    {
        out.elements[i] = in.elements[i * in.dim + dim];
    }
}

static void splitPointArray(PointArray& I, PointArray& I_L, PointArray& I_R) {

    int i=0;
    for(; i < I_L.width * I_L.dim; i++){
        I_L.elements[i] = I.elements[i];
    }
    int j=0;
    for(; i<I.width*I.dim && j<I_R.width*I_R.dim; i++, j++){
        I_R.elements[j] = I.elements[i];
    }

}

static void splitPointArrayWithValue(PointArray& V, PointArray& I, PointArray& I_L, PointArray& I_R, int current_dim, float value) {

    int i_l = 0;
    int i_r = 0;

    for(int i=0; i<I.width; i++)
    {
        float current_value = V.elements[static_cast<int>(I.elements[i] + 0.5) * V.dim + current_dim ];
        //~ printf("curr val: %f\n", current_value);
        if(current_value <= value && I_L.width > i_l ){
            //~ printf("add to left: %f with value %f\n", I.elements[i], current_value);
            I_L.elements[i_l++] = I.elements[i];
        }else if(current_value >= value && I_R.width > i_r){
            //~ printf("add to right: %f with value %f\n", I.elements[i], current_value);
            I_R.elements[i_r++] = I.elements[i];
        }else {
            if(i_r<I_R.width){
                I_R.elements[i_r++] = I.elements[i];
            }else if(i_l<I_L.width){
                I_L.elements[i_l++] = I.elements[i];
            }
        }
    }

}

#endif // !__POINTARRAY_HPP
