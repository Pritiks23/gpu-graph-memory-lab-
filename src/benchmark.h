#pragma once

#include <cuda_runtime.h>

#include <iostream>



class Benchmark
{

public:


    static size_t getFreeMemory()
    {

        size_t free_memory;

        size_t total_memory;



        cudaMemGetInfo(
            &free_memory,
            &total_memory
        );


        return free_memory;

    }



    static void printGPUInfo()
    {

        cudaDeviceProp prop;


        cudaGetDeviceProperties(
            &prop,
            0
        );


        std::cout
            << "GPU: "
            << prop.name
            << "\n";


        std::cout
            << "Memory: "
            << prop.totalGlobalMem
              /
              (1024*1024)
            << " MB\n";

    }

};
