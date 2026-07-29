#pragma once


#include <cuda_runtime.h>


class GPUMemoryPool
{


private:

    char* memory;

    size_t capacity;

    size_t offset;



public:


    GPUMemoryPool(size_t bytes)
        :
        capacity(bytes),
        offset(0)
    {

        cudaMalloc(
            &memory,
            bytes
        );

    }



    ~GPUMemoryPool()
    {

        cudaFree(memory);

    }



    void* allocate(size_t bytes)
    {


        if(offset + bytes > capacity)
        {

            throw std::runtime_error(
                "Pool exhausted"
            );

        }


        void* ptr =
            memory + offset;


        offset += bytes;


        return ptr;

    }



    void reset()
    {

        offset=0;

    }

};
