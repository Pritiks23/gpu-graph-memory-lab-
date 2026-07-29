#include "pagerank.h"

#include "allocator.h"
#include "memory_pool.h"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <algorithm>


#define CUDA_CHECK(call)                                      \
do                                                           \
{                                                            \
    cudaError_t err = call;                                  \
    if(err != cudaSuccess)                                   \
    {                                                        \
        std::cerr                                             \
        << "CUDA ERROR: "                                     \
        << cudaGetErrorString(err)                            \
        << std::endl;                                         \
        exit(EXIT_FAILURE);                                   \
    }                                                        \
} while(0)



// ============================================================
// CUDA PageRank Kernel
//
// SAME kernel for every memory strategy.
//
// Only allocation changes.
//
// ============================================================

__global__
void pagerank_kernel(

    const int* row_offsets,

    const int* columns,

    const float* rank,

    float* new_rank,

    int nodes

)
{


    int node =
        blockIdx.x * blockDim.x
        +
        threadIdx.x;



    if(node >= nodes)
        return;



    int start =
        row_offsets[node];


    int end =
        row_offsets[node + 1];



    float contribution = 0.0f;



    for(int edge=start; edge<end; edge++)
    {

        int neighbor =
            columns[edge];


        contribution +=
            rank[neighbor];

    }



    new_rank[node] =
        0.85f * contribution
        +
        0.15f;

}




// ============================================================
// Main Benchmark Function
//
// Four allocation strategies:
//
// 1. cudaMalloc
// 2. cudaMallocAsync
// 3. Memory Pool
// 4. Unified Memory
//
// ============================================================


float runPageRank(

    CSRGraph& graph,

    int iterations,

    AllocatorType allocator_type

)
{


    cudaStream_t stream;


    CUDA_CHECK(
        cudaStreamCreate(
            &stream
        )
    );



    int nodes =
        graph.nodes;


    int edges =
        graph.edges;



    GPUAllocator* allocator = nullptr;


    GPUMemoryPool* pool = nullptr;



    // ---------------------------------------------------------
    // Select allocator
    // ---------------------------------------------------------

    if(
        allocator_type ==
        AllocatorType::CUDA_MALLOC
    )
    {

        allocator =
            new CudaMallocAllocator();

    }


    else if(
        allocator_type ==
        AllocatorType::CUDA_ASYNC
    )
    {

        allocator =
            new CudaAsyncAllocator(
                stream
            );

    }


    else if(
        allocator_type ==
        AllocatorType::UNIFIED
    )
    {

        allocator =
            new UnifiedAllocator();

    }


    else if(
        allocator_type ==
        AllocatorType::MEMORY_POOL
    )
    {

        pool =
            new GPUMemoryPool(
                256 * 1024 * 1024
            );

    }



    int* d_rows;

    int* d_columns;

    float* d_rank;

    float* d_next;



    size_t rows_bytes =
        (nodes + 1)
        *
        sizeof(int);



    size_t columns_bytes =
        edges
        *
        sizeof(int);



    size_t rank_bytes =
        nodes
        *
        sizeof(float);



    // ---------------------------------------------------------
    // Allocate GPU memory
    // ---------------------------------------------------------

    if(pool)
    {


        d_rows =
        static_cast<int*>(
            pool->allocate(
                rows_bytes
            )
        );


        d_columns =
        static_cast<int*>(
            pool->allocate(
                columns_bytes
            )
        );


        d_rank =
        static_cast<float*>(
            pool->allocate(
                rank_bytes
            )
        );


        d_next =
        static_cast<float*>(
            pool->allocate(
                rank_bytes
            )
        );

    }

    else

    {


        d_rows =
        static_cast<int*>(
            allocator->allocate(
                rows_bytes
            )
        );


        d_columns =
        static_cast<int*>(
            allocator->allocate(
                columns_bytes
            )
        );


        d_rank =
        static_cast<float*>(
            allocator->allocate(
                rank_bytes
            )
        );


        d_next =
        static_cast<float*>(
            allocator->allocate(
                rank_bytes
            )
        );

    }



    // ---------------------------------------------------------
    // Copy graph data
    // ---------------------------------------------------------


    CUDA_CHECK(
        cudaMemcpy(
            d_rows,
            graph.row_offsets.data(),
            rows_bytes,
            cudaMemcpyHostToDevice
        )
    );



    CUDA_CHECK(
        cudaMemcpy(
            d_columns,
            graph.column_indices.data(),
            columns_bytes,
            cudaMemcpyHostToDevice
        )
    );



    std::vector<float> initial_rank(

        nodes,

        1.0f / nodes

    );



    CUDA_CHECK(
        cudaMemcpy(
            d_rank,
            initial_rank.data(),
            rank_bytes,
            cudaMemcpyHostToDevice
        )
    );



    CUDA_CHECK(
        cudaMemset(
            d_next,
            0,
            rank_bytes
        )
    );



    // ---------------------------------------------------------
    // Launch benchmark
    // ---------------------------------------------------------


    cudaEvent_t start;

    cudaEvent_t stop;



    CUDA_CHECK(
        cudaEventCreate(&start)
    );


    CUDA_CHECK(
        cudaEventCreate(&stop)
    );



    CUDA_CHECK(
        cudaEventRecord(start)
    );



    int threads = 256;


    int blocks =
        (nodes + threads - 1)
        /
        threads;



    for(int i=0;i<iterations;i++)
    {


        CUDA_CHECK(
            cudaMemset(
                d_next,
                0,
                rank_bytes
            )
        );



        pagerank_kernel<<<
            blocks,
            threads,
            0,
            stream
        >>>(

            d_rows,

            d_columns,

            d_rank,

            d_next,

            nodes

        );



        CUDA_CHECK(
            cudaGetLastError()
        );



        std::swap(
            d_rank,
            d_next
        );


    }



    CUDA_CHECK(
        cudaEventRecord(stop)
    );


    CUDA_CHECK(
        cudaEventSynchronize(stop)
    );



    float milliseconds;



    CUDA_CHECK(
        cudaEventElapsedTime(
            &milliseconds,
            start,
            stop
        )
    );



    // ---------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------


    if(pool)
    {

        delete pool;

    }

    else

    {

        allocator->deallocate(
            d_rows
        );


        allocator->deallocate(
            d_columns
        );


        allocator->deallocate(
            d_rank
        );


        allocator->deallocate(
            d_next
        );


        delete allocator;

    }



    cudaStreamDestroy(
        stream
    );



    return milliseconds;

}
