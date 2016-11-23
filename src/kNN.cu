#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <algorithm>

struct point {
    float x;
    float y;
//    float z;
};

struct pairSort {
    bool operator()(std::pair<int,float> &left, std::pair<int,float> &right){
        return left.second > right.second;
    }
};

float euclideanDistance(point p1, point p2) {
    return (p1.x - p2.x)*(p1.x-p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
}

std::vector<std::pair<int,float> > kNearestNeighbor(int index, std::vector<point> p, int k=1 ) {
    point p1 = p[index];
    std::vector<std::pair<int,float> > bestIndexDist;
    bestIndexDist.resize(k);

    for(int i=0; i<k; i++) {
        bestIndexDist[i].first = -1;
        bestIndexDist[i].second = FLT_MAX;
    }

    for(int i=0; i<p.size(); i++) {
        if(i != index) {
            float currentDist = euclideanDistance(p1, p[i]);

            for(int j=0; j<k; j++) {
                if(currentDist < bestIndexDist[j].second) {
                    bestIndexDist[j].first = i;
                    bestIndexDist[j].second = currentDist;
                    break;
                }
            }
            std::sort(bestIndexDist.begin(),bestIndexDist.end(), pairSort());
        }
    }
    return bestIndexDist;
}

int nearestNeighbor(int index, point* p, int length) {
    point p1 = p[index];
    float bestDist = FLT_MAX;
    int bestIndex = -1;

    for(int i=0; i<length; i++) {
        if(i != index) {
            float currentDist = euclideanDistance(p1, p[i]);
            if(currentDist < bestDist) {
                bestIndex = i;
                bestDist = currentDist;
            }
        }
    }

    return bestIndex;
}

int main(int argc, char** argv) {

    std::vector<point> randoms;
    randoms.resize(100);
    std::cout << randoms.size() << std::endl;
    int k = 20;
    std::vector<std::pair<int, float> > NNs;


    srand(time(NULL));
    for(int i=0; i<100; i++) {

        randoms[i].x = (float)rand() / RAND_MAX * 100.0;
        randoms[i].y = (float)rand() / RAND_MAX * 100.0;
//        randoms[i].z = (float)rand() / RAND_MAX * 100.0;
    }
    std::cout << randoms.size() << std::endl;
    NNs = kNearestNeighbor(0, randoms, k);
    std::cout << NNs.size() << std::endl;
    for(int i= 0; i<k;i++){
        printf("Nearest Neighbor of (%f, %f): (%f, %f)\n", randoms[0].x, randoms[0].y, randoms[NNs[i].first].x, randoms[NNs[i].first].y);
    }
    return 0;
}
