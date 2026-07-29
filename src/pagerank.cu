#include "pagerank.h"

#include "memory_pool.h"


#include <cuda_runtime.h>
#include <iostream>



__global__
void pagerank_kernel(

    int* row_offsets,

    int* columns,

    float* rank,

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
        row_offsets[node+1];



    float value=0;



    for(int i=start;i<end;i++)
    {

        value +=
            rank[columns[i]];

    }



    new_rank[node]=
        0.85f*value
        +
        0.15f;

}




float runPageRank(

    CSRGraph& graph,

    int iterations,

    AllocatorType type

)
{


    cudaStream_t stream;


    cudaStreamCreate(
        &stream
    );



    GPUAllocator* allocator=nullptr;



    if(type==AllocatorType::CUDA_MALLOC)
    {

        allocator =
        new CudaMallocAllocator();

    }


    else if(type==AllocatorType::CUDA_ASYNC)
    {

        allocator =
        new CudaAsyncAllocator(
            stream
        );

    }


    else if(type==AllocatorType::UNIFIED)
    {

        allocator =
        new UnifiedAllocator();

    }



    int nodes =
        graph.nodes;


    int edges =
        graph.edges;



    int* d_rows =
    (int*)allocator->allocate(
        (nodes+1)*sizeof(int)
    );


    int* d_cols =
    (int*)allocator->allocate(
        edges*sizeof(int)
    );


    float* rank =
    (float*)allocator->allocate(
        nodes*sizeof(float)
    );


    float* next =
    (float*)allocator->allocate(
        nodes*sizeof(float)
    );



    cudaMemcpy(
        d_rows,
        graph.row_offsets.data(),
        (nodes+1)*sizeof(int),
        cudaMemcpyHostToDevice
    );


    cudaMemcpy(
        d_cols,
        graph.column_indices.data(),
        edges*sizeof(int),
        cudaMemcpyHostToDevice
    );



    int threads=256;


    int blocks=
        (nodes+threads-1)
        /
        threads;



    cudaEvent_t start,stop;


    cudaEventCreate(&start);

    cudaEventCreate(&stop);



    cudaEventRecord(start);



    for(int i=0;i<iterations;i++)
    {


        pagerank_kernel<<<blocks,threads>>>(
            d_rows,
            d_cols,
            rank,
            next,
            nodes
        );


        cudaDeviceSynchronize();



        std::swap(
            rank,
            next
        );


    }



    cudaEventRecord(stop);


    cudaEventSynchronize(stop);



    float ms;


    cudaEventElapsedTime(
        &ms,
        start,
        stop
    );



    allocator->deallocate(d_rows);

    allocator->deallocate(d_cols);

    allocator->deallocate(rank);

    allocator->deallocate(next);



    delete allocator;


    cudaStreamDestroy(stream);



    return ms;

}
