#!/bin/bash


set -e


ALLOCATORS=(
cudaMalloc
cudaMallocAsync
unified_memory
)


for allocator in "${ALLOCATORS[@]}"
do

echo "Running $allocator"


./build/graph_benchmark \
--allocator $allocator


done
