#!/bin/bash
set -e
set -x

if [ -z "${WORKLOAD_BENCH_ROOT_PATH}" ]; then
  echo "WORKLOAD_BENCH_ROOT_PATH unset, by default set to ${PWD}" 
  WORKLOAD_BENCH_ROOT_PATH=${PWD}
fi

# collecting oneDNN primitive performance
_DNNL_GRAPH_DISABLE_COMPILER_BACKEND=1 ONEDNN_GRAPH_VERBOSE=1 bash ${WORKLOAD_BENCH_ROOT_PATH}/run_benchdnn_graph.sh MLP > ${WORKLOAD_BENCH_ROOT_PATH}/primitive.log

# aggregate data
python ${WORKLOAD_BENCH_ROOT_PATH}/scripts/onednn_verbose_parser.py --log_dir ${WORKLOAD_BENCH_ROOT_PATH}
echo "Performance data dumped to ${WORKLOAD_BENCH_ROOT_PATH}/primitive_single_matmul.csv"
