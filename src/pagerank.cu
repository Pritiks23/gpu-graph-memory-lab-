#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cmath>

#include "graph.h"


#define CUDA_CHECK(call)                                  \
do                                                       \
{                                                        \
    cudaError_t err = call;                              \
    if(err != cudaSuccess)                               \
    {                                                    \
        std::cerr << "CUDA Error: "                       \
                  << cudaGetErrorString(err)              \
                  << std::endl;                           \
        exit(EXIT_FAILURE);                              \
    }                                                    \
} while(0)



// ======================================================
// CUDA PageRank Kernel
//
// One GPU thread processes one graph node.
//
// CSR format:
//
// row_offsets[node]
//        |
//        v
// neighbors in column_indices
//
// ======================================================

__global__
void pagerank_kernel(
    const int* row_offsets,
    const int* column_indices,

    const float* rank,
    float* new_rank,

    int nodes,
    float damping
)
{

    int node = blockIdx.x * blockDim.x + threadIdx.x;


    if(node >= nodes)
        return;



    float contribution = 0.0f;



    int start =
        row_offsets[node];


    int end =
        row_offsets[node + 1];



    int degree =
        end - start;



    if(degree > 0)
    {

        float outgoing =
            rank[node] / degree;



        for(int i=start; i<end; i++)
        {

            int neighbor =
                column_indices[i];


            atomicAdd(
                &new_rank[neighbor],
                outgoing
            );

        }

    }



    new_rank[node] =
        damping * new_rank[node]
        +
        (1.0f - damping) / nodes;

}



// ======================================================
// Run PageRank using explicit CUDA memory
//
// This is our baseline allocator.
// Later versions replace only this section.
//
// ======================================================

float runPageRankCudaMalloc(
    CSRGraph& graph,
    int iterations
)
{

    int nodes =
        graph.nodes;


    int edges =
        graph.edges;



    int* d_row_offsets;

    int* d_columns;


    float* d_rank;

    float* d_new_rank;



    CUDA_CHECK(
        cudaMalloc(
            &d_row_offsets,
            (nodes + 1) * sizeof(int)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_columns,
            edges * sizeof(int)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_rank,
            nodes * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_new_rank,
            nodes * sizeof(float)
        )
    );



    CUDA_CHECK(
        cudaMemcpy(
            d_row_offsets,
            graph.row_offsets.data(),
            (nodes + 1) * sizeof(int),
            cudaMemcpyHostToDevice
        )
    );



    CUDA_CHECK(
        cudaMemcpy(
            d_columns,
            graph.column_indices.data(),
            edges * sizeof(int),
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
            nodes*sizeof(float),
            cudaMemcpyHostToDevice
        )
    );



    cudaEvent_t start, stop;


    cudaEventCreate(&start);
    cudaEventCreate(&stop);



    cudaEventRecord(start);



    int threads = 256;


    int blocks =
        (nodes + threads - 1)
        / threads;



    for(int i=0;i<iterations;i++)
    {

        CUDA_CHECK(
            cudaMemset(
                d_new_rank,
                0,
                nodes*sizeof(float)
            )
        );


        pagerank_kernel<<<blocks,threads>>>(
            d_row_offsets,
            d_columns,
            d_rank,
            d_new_rank,
            nodes,
            0.85f
        );


        CUDA_CHECK(
            cudaDeviceSynchronize()
        );



        std::swap(
            d_rank,
            d_new_rank
        );

    }



    cudaEventRecord(stop);

    cudaEventSynchronize(stop);



    float milliseconds;


    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );



    cudaFree(d_row_offsets);
    cudaFree(d_columns);
    cudaFree(d_rank);
    cudaFree(d_new_rank);



    return milliseconds;

}
