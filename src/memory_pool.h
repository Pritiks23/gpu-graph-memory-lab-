#pragma once

#include <cuda_runtime.h>

#include <vector>
#include <iostream>


#define CUDA_CHECK_POOL(call)                         \
do                                                    \
{                                                     \
    cudaError_t err = call;                           \
    if(err != cudaSuccess)                            \
    {                                                 \
        std::cerr                                     \
        << "CUDA Error: "                            \
        << cudaGetErrorString(err)                    \
        << std::endl;                                 \
        exit(EXIT_FAILURE);                           \
    }                                                 \
} while(0)



class GPUMemoryPool
{

private:

    char* memory_block_;

    size_t total_size_;

    size_t current_offset_;



public:


    GPUMemoryPool(size_t bytes)
        :
        total_size_(bytes),
        current_offset_(0)
    {

        CUDA_CHECK_POOL(
            cudaMalloc(
                &memory_block_,
                bytes
            )
        );


        std::cout
        << "Memory pool created: "
        << bytes/(1024*1024)
        << " MB\n";

    }



    ~GPUMemoryPool()
    {

        if(memory_block_)
        {
            cudaFree(memory_block_);
        }

    }



    void* allocate(size_t bytes)
    {


        if(current_offset_ + bytes > total_size_)
        {

            throw std::runtime_error(
                "GPU memory pool exhausted"
            );

        }



        void* ptr =
            memory_block_
            +
            current_offset_;



        current_offset_ += bytes;



        return ptr;

    }



    void reset()
    {

        current_offset_ = 0;

    }



    size_t usedMemory()
    {

        return current_offset_;

    }

};
