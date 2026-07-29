#!/bin/bash


set -e



echo "Building project..."



mkdir -p build

cd build


cmake ..

make -j$(nproc)



cd ..



echo "Running cudaMalloc benchmark"

./build/graph_benchmark \
--allocator cudaMalloc \
--graph data/cit-HepTh.txt



echo "Benchmark complete"
