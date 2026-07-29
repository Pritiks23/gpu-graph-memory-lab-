#include <iostream>
#include <string>

#include "graph.h"
#include "benchmark.h"


float runPageRankCudaMalloc(
    CSRGraph& graph,
    int iterations
);



int main(int argc, char** argv)
{


    std::cout
        << "====================================\n"
        << "GPU Graph Memory Optimization Lab\n"
        << "====================================\n\n";



    std::string graph_file =
        "data/cit-HepTh.txt";



    if(argc > 1)
    {
        graph_file = argv[1];
    }



    CSRGraph graph;


    graph.loadFromEdgeList(
        graph_file
    );



    int iterations = 100;



    std::cout
        << "\nRunning PageRank\n"
        << "Iterations: "
        << iterations
        << "\n\n";



    size_t free_before =
        Benchmark::getFreeMemory();



    float runtime =
        runPageRankCudaMalloc(
            graph,
            iterations
        );



    size_t free_after =
        Benchmark::getFreeMemory();



    std::cout
        << "====================================\n"
        << "Results\n"
        << "====================================\n";


    std::cout
        << "Allocator: cudaMalloc\n";


    std::cout
        << "Runtime: "
        << runtime
        << " ms\n";



    std::cout
        << "Average iteration: "
        << runtime / iterations
        << " ms\n";



    std::cout
        << "GPU memory change: "
        << (free_before-free_after)
        /1024.0/1024.0
        << " MB\n";



    return 0;

}
