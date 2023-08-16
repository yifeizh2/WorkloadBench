#!/bin/bash
set -e
set -x

if [ -z "${WORKLOAD_BENCH_ROOT_PATH}" ]; then
  echo "WORKLOAD_BENCH_ROOT_PATH unset, by default set to ${PWD}" 
  WORKLOAD_BENCH_ROOT_PATH=${PWD}
fi

# collecting oneDNN primitive performance
_DNNL_GRAPH_DISABLE_COMPILER_BACKEND=1 bash ${WORKLOAD_BENCH_ROOT_PATH}/run_benchdnn_graph.sh > ${WORKLOAD_BENCH_ROOT_PATH}/onednn_primitive_workload_perf.log

# collecting compiler coarse grain fusion disabled performance
SC_COARSE_GRAIN_FUSION=0 bash ${WORKLOAD_BENCH_ROOT_PATH}/run_benchdnn_graph.sh > ${WORKLOAD_BENCH_ROOT_PATH}/compiler_without_coarse_grain_workload_perf.log

# collecting compiler default performance
bash ${WORKLOAD_BENCH_ROOT_PATH}/run_benchdnn_graph.sh > ${WORKLOAD_BENCH_ROOT_PATH}/compiler_workload_perf.log

# aggregate data
python ${WORKLOAD_BENCH_ROOT_PATH}/scripts/subgraph_perf_parser.py --log_dir ${WORKLOAD_BENCH_ROOT_PATH}
echo "Performance data dumped to ${WORKLOAD_BENCH_ROOT_PATH}/subgraph_perf.csv"
