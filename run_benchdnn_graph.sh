export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

if [ -z "${JSON_PATH}" ]; then
  echo "JSON_PATH unset, by default set to ${PWD}/workloads" 
  JSON_PATH=${PWD}/workloads
fi

if [ -z "${ONEDNN_BUILD_PATH}" ]; then
  echo "ONEDNN_BUILD_PATH unset, by default set to ${PWD}/oneDNN/build" 
  ONEDNN_BUILD_PATH=${PWD}/oneDNN/build
fi

# FP32 MLP
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:32x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:64x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:128x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:256x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:512x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:32x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:64x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:128x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:256x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=0:512x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1_fp32.json

# INT8 MLP
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:32x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:64x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:128x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:256x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:512x13 --case=${JSON_PATH}/mlp_relu_13x512x256x128.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:32x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:64x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:128x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:256x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --in-shapes=1:512x479 --case=${JSON_PATH}/mlp_relu_479x1024x1024x512x256x1.json

# FP32 MHA
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS32_SL128_H768_A8_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS64_SL128_H768_A8_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS128_SL128_H768_A8_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS32_SL128_H768_A12_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS64_SL128_H768_A12_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS128_SL128_H768_A12_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS32_SL384_H1024_A8_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS64_SL384_H1024_A8_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS128_SL384_H1024_A8_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS32_SL512_H1024_A16_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS64_SL512_H1024_A16_fp32.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS128_SL512_H1024_A16_fp32.json


# INT8 MHA
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS32_SL128_H768_A8.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS64_SL128_H768_A8.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS128_SL128_H768_A8.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS32_SL128_H768_A12.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS64_SL128_H768_A12.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS128_SL128_H768_A12.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS32_SL384_H1024_A8.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS64_SL384_H1024_A8.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS128_SL384_H1024_A8.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS32_SL512_H1024_A16.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS64_SL512_H1024_A16.json
KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=32 numactl --physcpubind=0-31 --membind=0 ${ONEDNN_BUILD_PATH}/tests/benchdnn/benchdnn --graph --mode=P --case=${JSON_PATH}/MHA_BS128_SL512_H1024_A16.json
