#!/bin/bash
set -e
set -x

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

if [ -z "${JSON_PATH}" ]; then
  echo "JSON_PATH unset, by default set to ${PWD}/workloads" 
  JSON_PATH=${PWD}/workloads
fi

if [ -z "${ONEDNN_BUILD_PATH}" ]; then
  echo "ONEDNN_BUILD_PATH unset, by default set to ${PWD}/oneDNN/build" 
  ONEDNN_BUILD_PATH=${PWD}/oneDNN/build
fi

if [ -z "${WORKLOAD_BENCH_ROOT_PATH}" ]; then
  echo "WORKLOAD_BENCH_ROOT_PATH unset, by default set to ${PWD}" 
  WORKLOAD_BENCH_ROOT_PATH=${PWD}
fi

export SC_COARSE_GRAIN_FUSION=0

# FP32 MLP
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_1_B32_F32.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:32x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128_fp32.json
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_1_B64_F32.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:64x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128_fp32.json
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_1_B128_F32.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:128x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128_fp32.json
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_1_B256_F32.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:256x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128_fp32.json
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_1_B512_F32.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:512x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128_fp32.json
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_2_B32_F32.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:32x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1_fp32.json
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_2_B64_F32.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:64x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1_fp32.json
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_2_B128_F32.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:128x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1_fp32.json
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_2_B256_F32.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:256x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1_fp32.json
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_2_B512_F32.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:512x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1_fp32.json

# INT8 MLP
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_1_B32_INT8.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:32x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128.json
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_1_B64_INT8.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:64x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128.json
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_1_B128_INT8.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:128x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128.json
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_1_B256_INT8.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:256x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128.json
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_1_B512_INT8.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:512x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128.json
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_2_B32_INT8.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:32x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1.json
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_2_B64_INT8.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:64x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1.json
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_2_B128_INT8.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:128x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1.json
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_2_B256_INT8.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:256x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1.json
SC_TRACE="${WORKLOAD_BENCH_ROOT_PATH}/MLP_2_B512_INT8.json" KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:512x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1.json

# aggregate data
python ${WORKLOAD_BENCH_ROOT_PATH}/scripts/aggregated_trace_parser.py --workload_bench_dir ${WORKLOAD_BENCH_ROOT_PATH}
echo "Performance data dumped to ${WORKLOAD_BENCH_ROOT_PATH}/compiler_single_matmul.csv"
