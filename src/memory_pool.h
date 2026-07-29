#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>


class GPUMemoryPool
{


private:

    char* buffer;

    size_t capacity;

    size_t offset;



public:


    GPUMemoryPool(size_t bytes)
    {

        capacity = bytes;

        offset = 0;


        cudaError_t err =
            cudaMalloc(
                &buffer,
                bytes
            );


        if(err != cudaSuccess)
        {

            throw std::runtime_error(
                "Could not allocate GPU pool"
            );

        }


        std::cout
        << "GPU Memory Pool initialized: "
        << bytes/(1024*1024)
        << " MB\n";

    }



    ~GPUMemoryPool()
    {

        cudaFree(buffer);

    }



    void* allocate(size_t bytes)
    {


        if(offset + bytes > capacity)
        {

            throw std::runtime_error(
                "GPU memory pool exhausted"
            );

        }



        void* ptr =
            buffer + offset;



        offset += bytes;



        return ptr;

    }



    void reset()
    {

        offset=0;

    }



    size_t used()
    {

        return offset;

    }


};
