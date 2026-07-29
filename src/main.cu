#include <iostream>
#include <string>


#include "graph.h"
#include "pagerank.h"


AllocatorType parseAllocator(
    std::string name
)
{

    if(name=="cudaMalloc")
        return AllocatorType::CUDA_MALLOC;


    if(name=="cudaMallocAsync")
        return AllocatorType::CUDA_ASYNC;


    if(name=="unified_memory")
        return AllocatorType::UNIFIED;


    return AllocatorType::MEMORY_POOL;

}



int main(int argc,char** argv)
{


    std::string allocator="cudaMalloc";


    std::string file=
        "../data/cit-HepTh.txt";



    for(int i=0;i<argc;i++)
    {

        if(std::string(argv[i])=="--allocator")
            allocator=argv[i+1];

    }



    CSRGraph graph;


    graph.loadFromEdgeList(
        file
    );



    float runtime =
        runPageRank(
            graph,
            100,
            parseAllocator(
                allocator
            )
        );



    std::cout
    << "\nAllocator: "
    << allocator
    << "\nRuntime: "
    << runtime
    << " ms\n";


}
