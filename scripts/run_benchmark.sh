#!/bin/bash

set -e


echo "================================"
echo "GPU Graph Memory Benchmark"
echo "================================"



ALLOCATORS=(

cudaMalloc

cudaMallocAsync

memory_pool

unified_memory

)



for allocator in "${ALLOCATORS[@]}"
do

    echo ""

    echo "Running:"
    echo $allocator


    ./build/graph_benchmark \
    --allocator $allocator


done



echo ""

echo "Benchmark complete"
