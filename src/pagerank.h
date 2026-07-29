#pragma once


#include "graph.h"


enum class AllocatorType
{

    CUDA_MALLOC,

    CUDA_ASYNC,

    MEMORY_POOL,

    UNIFIED

};



float runPageRank(
    CSRGraph& graph,
    int iterations,
    AllocatorType type
);
