#include <iostream>
#include <string>


#include "graph.h"
#include "pagerank.h"
#include "benchmark.h"



AllocatorType parseAllocator(
    const std::string& name
)
{


    if(name == "cudaMalloc")
    {

        return AllocatorType::CUDA_MALLOC;

    }



    if(name == "cudaMallocAsync")
    {

        return AllocatorType::CUDA_ASYNC;

    }



    if(name == "memory_pool")
    {

        return AllocatorType::MEMORY_POOL;

    }



    if(name == "unified_memory")
    {

        return AllocatorType::UNIFIED;

    }



    throw std::runtime_error(
        "Unknown allocator: "
        + name
    );

}



int main(int argc, char** argv)
{


    std::cout
    << "====================================\n"
    << " GPU Graph Memory Optimization Lab\n"
    << "====================================\n\n";



    Benchmark::printGPUInfo();



    std::string allocator_name =
        "cudaMalloc";



    std::string graph_file =
        "../data/cit-HepTh.txt";



    for(int i=1;i<argc;i++)
    {


        std::string arg =
            argv[i];


        if(arg=="--allocator")
        {

            allocator_name =
                argv[++i];

        }


        else if(arg=="--graph")
        {

            graph_file =
                argv[++i];

        }

    }



    std::cout
    << "\nSelected allocator: "
    << allocator_name
    << "\n";



    CSRGraph graph;



    graph.loadFromEdgeList(
        graph_file
    );



    int iterations = 100;



    size_t memory_before =
        Benchmark::getFreeMemory();



    float runtime =
        runPageRank(

            graph,

            iterations,

            parseAllocator(
                allocator_name
            )

        );



    size_t memory_after =
        Benchmark::getFreeMemory();



    std::cout
    << "\n========== RESULTS ==========\n";



    std::cout
    << "Allocator: "
    << allocator_name
    << "\n";



    std::cout
    << "Iterations: "
    << iterations
    << "\n";



    std::cout
    << "Total Runtime: "
    << runtime
    << " ms\n";



    std::cout
    << "Average Iteration: "
    << runtime / iterations
    << " ms\n";



    std::cout
    << "Approx GPU Memory Used: "
    << (memory_before - memory_after)
       /
       1024.0
       /
       1024.0
    << " MB\n";



    std::cout
    << "=============================\n";


    return 0;

}
