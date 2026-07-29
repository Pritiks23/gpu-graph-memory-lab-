#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <chrono>


struct BenchmarkResult
{

    double total_ms;

    double avg_iteration_ms;


    size_t start_free_memory;

    size_t end_free_memory;

    size_t peak_memory_used;



    int allocation_count;

    size_t allocated_bytes;

};



class Benchmark
{

private:

    cudaEvent_t start;

    cudaEvent_t stop;



public:


    Benchmark()
    {

        cudaEventCreate(&start);

        cudaEventCreate(&stop);

    }



    ~Benchmark()
    {

        cudaEventDestroy(start);

        cudaEventDestroy(stop);

    }



    void begin()
    {

        cudaEventRecord(start);

    }



    float end()
    {

        cudaEventRecord(stop);

        cudaEventSynchronize(stop);


        float ms;


        cudaEventElapsedTime(
            &ms,
            start,
            stop
        );


        return ms;

    }



    static size_t getFreeMemory()
    {

        size_t free_mem;

        size_t total_mem;


        cudaMemGetInfo(
            &free_mem,
            &total_mem
        );


        return free_mem;

    }



};
