#ifndef __LBKDTREE_HPP
#define __LBKDTREE_HPP

#include "PointArray.hpp"
#include "../ext/CTPL/ctpl.h"

#include <stdlib.h>
#include <math.h>


/**
 * @brief The LBKdTree class implements a left-balanced array-based index kd-tree.
 *          Left-Balanced: minimum memory
 *          Array-Based: Good for GPU - Usage
 */
class LBKdTree {
public:

    LBKdTree( PointArray& vertices , int num_threads=8);

    void generateKdTree( PointArray& vertices );

    PointArray getKdTreeArray();

    

private:

    
    void generateKdTreeArray(PointArray& V, PointArray* sorted_indices, int max_dim, PointArray& kd_tree);

    //void generateAndSort(int id, PointArray& vertices, PointArray* indices_sorted, PointArray* values_sorted, int dim);

    void sortByDim(PointArray& V, int dim, PointArray& indices, PointArray& values);

    void naturalMergeSort(PointArray& in, int dim, PointArray& indices,  PointArray& m, int limit=-1);

    void mergeHostWithIndices(float* a, float* b, int i1, int j1, int i2, int j2, int limit=-1);

    PointArray kd_tree;
    

    // Static member

    static int st_num_threads;
    static int st_depth_threads;

    static ctpl::thread_pool *pool;

    static void generateKdTreeRecursive(int id, PointArray& V, PointArray* sorted_indices, int current_dim, int max_dim, PointArray& kd_tree, int size, int max_tree_depth, int position, int current_depth);

    static void test(int id, PointArray* sorted_indices);
    

};




#endif // !__LBKDTREE_HPP
