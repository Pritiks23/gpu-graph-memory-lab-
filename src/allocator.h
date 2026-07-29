#pragma once

#include <cuda_runtime.h>
#include <iostream>


class GPUAllocator
{

public:

    virtual void* allocate(size_t bytes)=0;

    virtual void deallocate(void* ptr)=0;

    virtual ~GPUAllocator(){}

};



// ==================================================
// cudaMalloc allocator
// ==================================================

class CudaMallocAllocator : public GPUAllocator
{

public:

    void* allocate(size_t bytes) override
    {

        void* ptr=nullptr;


        cudaError_t err =
            cudaMalloc(
                &ptr,
                bytes
            );


        if(err != cudaSuccess)
        {
            std::cerr
            << "cudaMalloc failed: "
            << cudaGetErrorString(err)
            << std::endl;

            exit(1);
        }


        return ptr;

    }



    void deallocate(void* ptr) override
    {

        cudaFree(ptr);

    }

};



// ==================================================
// cudaMallocAsync allocator
// ==================================================

class CudaAsyncAllocator : public GPUAllocator
{

private:

    cudaStream_t stream;


public:


    CudaAsyncAllocator(cudaStream_t s)
        :
        stream(s)
    {}



    void* allocate(size_t bytes) override
    {

        void* ptr=nullptr;


        cudaMallocAsync(
            &ptr,
            bytes,
            stream
        );


        return ptr;

    }



    void deallocate(void* ptr) override
    {

        cudaFreeAsync(
            ptr,
            stream
        );

    }

};



// ==================================================
// Unified Memory allocator
// ==================================================

class UnifiedAllocator : public GPUAllocator
{

public:


    void* allocate(size_t bytes) override
    {

        void* ptr=nullptr;


        cudaMallocManaged(
            &ptr,
            bytes
        );


        return ptr;

    }



    void deallocate(void* ptr) override
    {

        cudaFree(ptr);

    }

};
