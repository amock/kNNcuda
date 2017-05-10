#include "LBKdTree.hpp"
#include <stdio.h>


/// Public

LBKdTree::LBKdTree(){

}

LBKdTree::LBKdTree( PointArray& vertices) {
    this->generateKdTree(vertices);
}

void LBKdTree::generateKdTree(PointArray &vertices) {
    struct PointArray indices_sorted[ vertices.dim ];
    struct PointArray values_sorted[ vertices.dim ];

    for(int i=0; i < vertices.dim; i++)
    {
        generatePointArray(indices_sorted[i], vertices.width, 1);

        generatePointArray(values_sorted[i], vertices.width, 1);

        fillPointArrayWithSequence(indices_sorted[i]);

        this->sortByDim( vertices, i, indices_sorted[i] , values_sorted[i]);
    }

    this->generateKdTreeArray(vertices, indices_sorted, vertices.dim, this->kd_tree);

    for(int i=0; i<vertices.dim;i++)
    {
        free(indices_sorted[i].elements);
        free(values_sorted[i].elements);
    }
}

PointArray LBKdTree::getKdTreeArray() {
    return this->kd_tree;
}

/// Private


void LBKdTree::generateKdTreeArray(PointArray& V, PointArray* sorted_indices, int max_dim, PointArray& kd_tree) {

    int size;
    int max_tree_depth;

    max_tree_depth = static_cast<int>( log2f(V.width - 1 ) + 2.0 ) ;

    if (V.width == 1)
    {
        max_tree_depth = 1;
    }

    size = V.width * 2 - 1;

    generatePointArray(kd_tree, size, 1);

    //start real generate
    generateKdTreeRecursive(V, sorted_indices, 0, max_dim, kd_tree, size, max_tree_depth, 0);

}

void LBKdTree::generateKdTreeRecursive(PointArray& V, PointArray* sorted_indices, int current_dim, int max_dim, PointArray& kd_tree, int size, int max_tree_depth, int position) {

    int left = position*2+1;
    int right = position*2+2;

    if( right > size-1 || left > size-1 )
    {

        kd_tree.elements[position] = sorted_indices[current_dim].elements[0];

    } else {
        /// split sorted_indices
        int indices_size = sorted_indices[current_dim].width;

        int v = pow( 2, static_cast<int>(log2f(indices_size-1) ) );
        int left_size = indices_size - v/2;

        if( left_size > v )
        {
            left_size = v;
        }
        int right_size = indices_size - left_size;

        float split_value = ( V.elements[current_dim+static_cast<int>(sorted_indices[current_dim].elements[left_size-1])*V.dim ] + V.elements[current_dim+static_cast<int>(sorted_indices[current_dim].elements[left_size] ) * V.dim] ) /2.0;

        kd_tree.elements[ position ] = split_value;

        struct PointArray sorted_indices_left[max_dim];
        struct PointArray sorted_indices_right[max_dim];

        // alloc new memory
        for( int i=0; i<max_dim; i++ )
        {
            // memory corruption when malloc

            sorted_indices_left[i].width = left_size;
            sorted_indices_left[i].dim = 1;
            sorted_indices_left[i].elements = (float*)malloc( (left_size+1) *sizeof(float) );

            sorted_indices_right[i].width = right_size;
            sorted_indices_right[i].dim = 1;
            sorted_indices_right[i].elements = (float*)malloc( (right_size+1) * sizeof(float) );

            if( i == current_dim ){
                splitPointArray( sorted_indices[i], sorted_indices_left[i], sorted_indices_right[i]);
            }else{
                splitPointArrayWithValue(V, sorted_indices[i], sorted_indices_left[i], sorted_indices_right[i], current_dim, split_value);
            }

        }

        generateKdTreeRecursive(V, sorted_indices_left, (current_dim+1)%max_dim, max_dim, kd_tree, size, max_tree_depth, left);
        generateKdTreeRecursive(V, sorted_indices_right, (current_dim+1)%max_dim, max_dim, kd_tree, size, max_tree_depth, right);


        // alloc new memory
        for(int i=0; i<max_dim; i++)
        {
            free( sorted_indices_left[i].elements );
            free( sorted_indices_right[i].elements );
        }
    }


}


void LBKdTree::sortByDim(PointArray& V, int dim, PointArray& indices, PointArray& values) {

    naturalMergeSort(V, dim, indices, values);

}

void LBKdTree::naturalMergeSort(PointArray& in, int dim, PointArray& indices, PointArray& m, int limit) {

    copyDimensionToPointArray(in, dim, m);

    int m_elements = m.width * m.dim;

    int slide_buffer_size = int(m_elements-0.5);
    int* slide_buffer = (int*) malloc(slide_buffer_size * sizeof(int));


    //create RUNS
    int num_slides = 1;
    slide_buffer[0] = 0;
    for(int i=1; i < slide_buffer_size+1; i++)
    {
        if(m.elements[i] < m.elements[i-1])
        {
            slide_buffer[num_slides] = i;
            num_slides++;
        }

    }
    slide_buffer[num_slides] = m_elements;
    slide_buffer_size = num_slides+1;


    //sort
    int count = 0;
    int current_limit = -1;
    while(num_slides > 1)
    {
        if(num_slides > 2){
            current_limit = limit;
        }

        int i;

        for(i=2;i<int(num_slides+1);i+=2)
        {

            mergeHostWithIndices(m.elements, indices.elements , slide_buffer[i-2], slide_buffer[i-1]-1, slide_buffer[i-1], slide_buffer[i]-1, current_limit);


            slide_buffer[i/2]= slide_buffer[i];
        }

        if(num_slides%2 == 1){
            slide_buffer[(num_slides+1)/2] = slide_buffer[num_slides];
        }

        count ++;
        num_slides = int(num_slides/2.0+0.5);

    }

    free(slide_buffer);
}

void LBKdTree::mergeHostWithIndices(float* a, float* b, int i1, int j1, int i2, int j2, int limit) {

    int limit_end = limit;

    float* temp = (float*) malloc((j2-i1+1) * sizeof(float));  //array used for merging
    int* temp_indices = (int*) malloc((j2-i1+1) * sizeof(int));  //array used for merging

    int i,j,k;
    i=i1;    //beginning of the first list
    j=i2;    //beginning of the second list
    k=0;

    int counter = 0;

    while( i<=j1 && j<=j2 && limit!=0 )    //while elements in both lists
    {
        counter ++;
        limit--;
        if(a[i]<a[j]){
            temp_indices[k] = b[i];
            temp[k++]=a[i++];

        }else{
            temp_indices[k] = b[j];
            temp[k++]=a[j++];
        }
    }

    while(i <= j1 && limit != 0) //copy remaining elements of the first list
    {
        temp_indices[k] = b[i];
        temp[k++]=a[i++];
    }

    while(j <= j2 && limit!=0 ) {   //copy remaining elements of the second list
        temp_indices[k] = b[j];
        temp[k++]=a[j++];
    }

    //Transfer elements from temp[] back to a[]
    for(i=i1,j=0;i<=j2 && limit_end!=0 ;i++,j++,limit_end--)
    {
        b[i] = temp_indices[j];
        if(b[i] < 0){
            printf("THERE IS SOMETHING WRONG\n");
        }
        a[i] = temp[j];
    }

    free(temp_indices);
    free(temp);
}

