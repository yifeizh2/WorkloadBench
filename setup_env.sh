#!/bin/bash
set -e
set -x

WORK_DIR=$PWD

# Prerequsite: conda, python 3.8 or above; cmake; LLVM-13

# Step 0.0: Create conda environment & install related packages
which conda
if [ $?==1 ]; then
  wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.1-0-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b -p ~/miniconda
    ./miniconda/bin/conda create -yn ${CONDA_ENV_NAME}
    source ./miniconda/bin/activate ${CONDA_ENV_NAME}
else
    conda create -n ${CONDA_ENV_NAME} python=3.9 -y
    conda activate ${CONDA_ENV_NAME}
fi
conda install cmake ninja mkl mkl-include -y
conda install gperftools -c conda-forge -y

# Step 0.1: Build LLVM-13
git clone --depth 1 --branch llvmorg-13.0.0  https://github.com/llvm/llvm-project
cd llvm-project
mkdir build
if [ -z "${LLVM_INSTALL_PATH}" ]; then
  echo "Explicitly set LLVM_INSTALL_PATH as ${WORK_DIR}/llvm-project/install" 
  mkdir install
  LLVM_INSTALL_PATH=${WORK_DIR}/llvm-project/install
fi
cd build
cmake ../llvm -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PATH} -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=X86
make install -j
export PATH=${LLVM_INSTALL_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${LLVM_INSTALL_PATH}/lib:$LD_LIBRARY_PATH
cd ../..

# Step 1: Build benchdnn graph
git clone --branch dev-graph-beta-3-paper https://github.com/oneapi-src/oneDNN.git
cd oneDNN
git submodule sync
git submodule update --init --recursive
mkdir build
cd build
cmake -DDNNL_GRAPH_BUILD_TESTS=1 -DDNNL_GRAPH_BUILD_COMPILER_BACKEND=1 ..
make -j benchdnn
