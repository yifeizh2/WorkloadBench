#!/bin/bash
set -e
set -x

WORK_DIR=$PWD

# Step 0: Create conda environment & install related packages
conda create -n benchdnn_graph python=3.8 -y
conda activate benchdnn_graph
conda install gperftools -c conda-forge -y

# Step 1: Build benchdnn graph
git clone --branch dev-graph-beta-3-paper https://github.com/oneapi-src/oneDNN.git
cd oneDNN
git submodule sync
git submodule update --init --recursive
mkdir build
cd build
cmake -DDNNL_GRAPH_BUILD_TESTS=1 -DDNNL_GRAPH_BUILD_COMPILER_BACKEND=1 -DDNNL_GRAPH_LLVM_CONFIG="/llvm-project/install/bin/llvm-config" ..
make -j benchdnn
