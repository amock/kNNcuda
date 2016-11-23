#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdexcept>

#define POINT_SIZE 10000
#define K_NEAR 20

struct point {
    float x;
    float y;
    float z;
    float operator[](int index) {
        switch(index){
			case 0: return x;
			case 1: return y;
			case 2: return z;
			default: throw std::invalid_argument("Index out of bounds."); 
		}
    }
};

struct pairSort {
    bool operator()(std::pair<int,float> &left, std::pair<int,float> &right){
        return left.second < right.second;
    }
};

float euclideanDistance(point p1, point p2) {
	float dist = 0.0;
	for(int i = 0; i < 3; i++){
		dist += (p1[i] - p2[i])*(p1[i]-p2[i]);
	}
    return dist;
}

std::vector<std::pair<int,float> > kNearestNeighbor(int index, std::vector<point> p, int k=1 ) {
    point p1 = p[index];
    std::vector<std::pair<int,float> > bestIndexDist;
    bestIndexDist.resize(p.size()-1);

    for(int i=0; i<k; i++) {
        bestIndexDist[i].first = -1;
        bestIndexDist[i].second = FLT_MAX;
    }
	int realindex = 0;
    for(int i=0; i<p.size(); i++) {
        if(i != index) {
            float currentDist = euclideanDistance(p1, p[i]);
			bestIndexDist[realindex].first = i;
			bestIndexDist[realindex].second = currentDist;
            realindex++;
        }
    }
    
    std::sort(bestIndexDist.begin(),bestIndexDist.end(), pairSort());
    
    return std::vector<std::pair<int,float> > (bestIndexDist.begin(),bestIndexDist.begin()+k);
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
	clock_t prgstart, prgend;
	
	int seed = time(NULL);
	std::cout << "seed: " << seed << std::endl;
	std::cout << "Start program with values:  " << std::endl;
	std::cout << "  POINT_SIZE = " << POINT_SIZE << std::endl;
	std::cout << "  K          = " << K_NEAR << std::endl; 
	srand(seed);
	prgstart = clock();
	
    std::vector<point> randoms;
    randoms.resize(POINT_SIZE);
    
    
    for(int i=0; i<randoms.size(); i++) {
        randoms[i].x = (float)rand() / RAND_MAX * 100.0;
        randoms[i].y = (float)rand() / RAND_MAX * 100.0;
        randoms[i].z = (float)rand() / RAND_MAX * 100.0;
    }
    
    
    for(int i=0; i<randoms.size(); i++){
		std::vector<std::pair<int, float> > NNs = kNearestNeighbor(i, randoms, K_NEAR);
	}
	
    //for(int i= 0; i<NNs.size();i++){
    //    printf("%d Nearest Neighbor of (%f, %f, %f): (%f, %f, %f) , %f\n",i+1 ,randoms[0][0], randoms[0][1], randoms[0][2], randoms[NNs[i].first][0], randoms[NNs[i].first][1], randoms[NNs[i].first][2], NNs[i].second);
    //}
    
    prgend=clock();
	printf("Laufzeit insgesamt %f seconds\n",(float)(prgend-prgstart) / CLOCKS_PER_SEC);
	
    return 0;
}
