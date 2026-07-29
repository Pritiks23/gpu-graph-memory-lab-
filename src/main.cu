#include <iostream>
#include <string>


#include "graph.h"
#include "benchmark.h"
#include "memory_pool.h"



float runPageRankCudaMalloc(
    CSRGraph& graph,
    int iterations
);



int main(int argc,char** argv)
{


    Benchmark::printGPUInfo();



    std::string allocator =
        "cudaMalloc";


    std::string graph_file =
        "../data/cit-HepTh.txt";



    for(int i=1;i<argc;i++)
    {

        std::string arg = argv[i];


        if(arg=="--allocator")
        {

            allocator =
                argv[++i];

        }


        if(arg=="--graph")
        {

            graph_file =
                argv[++i];

        }

    }



    std::cout
    << "\nAllocator: "
    << allocator
    << "\n";



    CSRGraph graph;


    graph.loadFromEdgeList(
        graph_file
    );



    int iterations = 100;



    size_t before =
        Benchmark::getFreeMemory();



    float runtime;



    if(allocator=="cudaMalloc")
    {

        runtime =
        runPageRankCudaMalloc(
            graph,
            iterations
        );

    }


    else
    {

        std::cout
        <<
        "Allocator not implemented yet\n";


        return 0;

    }



    size_t after =
        Benchmark::getFreeMemory();



    std::cout
    << "\n========== RESULTS ==========\n";


    std::cout
    << "Runtime: "
    << runtime
    << " ms\n";


    std::cout
    << "Avg iteration: "
    << runtime/iterations
    << " ms\n";


    std::cout
    << "Memory used: "
    << (before-after)
       /
       1024.0
       /
       1024.0
    << " MB\n";


    return 0;

}
