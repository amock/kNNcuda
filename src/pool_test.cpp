#include <iostream>
#include "../ext/CTPL/ctpl.h"

ctpl::thread_pool* p;

void printArray(float* a, size_t size)
{
    for(int i=0; i<size; i++)
    {
        std::cout << *(a+i) << " ";
    }
    std::cout << std::endl;
}

void fillArray(float* a, size_t size){
    for(int i=0; i<size; i++)
    {
        *(a+i) = i*0.5;
    }
}

void addFloatPointer(int id, float* a, float* b, float* c)
{
    for(int i=0; i<30000; i++)
    {
        
    }
    *c = *a + *b;
    
}

void addFloatPointerRec(int id, float* a, float* b, float* c, int index, int limit)
{
    for(int i=0; i<30000; i++)
    {
        
    }
    *c = *a + *b;
    if(index < limit){
        p->push(addFloatPointerRec, a+1, b+1, c+1, index+1, limit);
    }
}

int main(int argc, char** argv){
	
    std::cout << "Hello World" << std::endl;

    size_t vec_size = 100000;
    std::shared_ptr<float> vec_a(new float[vec_size]);
    fillArray(vec_a.get(), vec_size);
    std::shared_ptr<float> vec_b(new float[vec_size]);
    fillArray(vec_b.get(), vec_size);

    //sequentially
    std::cout << "Start Sequentially" << std::endl;
    std::shared_ptr<float> vec_res_seq(new float[vec_size]);
    
    for(int i=0; i<vec_size; i++)
    {
         addFloatPointer(0, vec_a.get()+i, vec_b.get()+i, vec_res_seq.get()+i); 
    }

    //printArray(vec_res_seq.get(), vec_size);
    std::cout << "Finished Sequentially" << std::endl;

    // threadpool
    std::cout << "Start Thread Pool" << std::endl;
    std::shared_ptr<float> vec_res_thr(new float[vec_size]);

    p = new ctpl::thread_pool(8);

    p->push(addFloatPointerRec, vec_a.get(), vec_b.get(), vec_res_thr.get(), 0, vec_size);
/*
    for(int i=0; i<vec_size; i++)
    {
        //push functions
        p->push(addFloatPointer, vec_a.get()+i, vec_b.get()+i, vec_res_thr.get()+i);
    }*/

    // waiting for all threads to finish
    p->stop(true);

    //printArray(vec_res_thr.get(), vec_size);
    std::cout << "Finished Thread Pool" << std::endl;

    std::cout << "Check results..." << std::endl;
    bool success = true;
    int i;
    std::vector<int> false_indices;
    for(i=0; i<vec_size; i++)
    {
        if( *(vec_res_seq.get()+i) != *(vec_res_thr.get()+i) )
        {
            success = false;
            false_indices.push_back(i);
            std::cout << i << " " << *(vec_res_seq.get()+i) << " " << *(vec_res_thr.get()+i) << std::endl;
        }
    }

    if(success){
        std::cout << "Check successful" << std::endl;
    } else {
        std::cout << "Check not successful." << std::endl;
        for(int i = 0; i<false_indices.size(); i++)
        {
            /*std::cout << "Elem: "<<false_indices[i]
            <<" Seq: "<< *(vec_res_seq.get()+false_indices[i] )
            <<" Thr: "<< *(vec_res_thr.get()+false_indices[i] ) << std::endl;
            */
        }
    }

	return 1;
}
