# WorkloadBench

This repo provides scripts for benchmarking oneDNN Primitive & oneDNN Graph Compiler's performance with benchdnn graph.

[Benchdnn graph](https://github.com/oneapi-src/oneDNN/blob/master/tests/benchdnn/doc/driver_graph.md) is a benchdnn extension for performance benchmarking on workloads defined in the format of json serialization graphs.

The current repo includes json serialization graphs for the following listed workloads.

| test name  | data type     | input batch size         | sequence length | hidden size              | head numbers
|------------|---------------|--------------------------|-----------------|--------------------------|--------------
| MLP-1      | FP32, Int8    | 32, 64, 128, 256, 512    | N/A             | 13x512x256x128           | N/A
| MLP-2      | FP32, Int8    | 32, 64, 128, 256, 512    | N/A             | 479x1024x1024x512x256x1  | N/A
| MHA-1      | FP32, Int8    | 32, 64, 128              | 128             | 768                      | 8
| MHA-2      | FP32, Int8    | 32, 64, 128              | 128             | 768                      | 12
| MHA-3      | FP32, Int8    | 32, 64, 128              | 384             | 1024                     | 8
| MHA-4      | FP32, Int8    | 32, 64, 128              | 512             | 1024                     | 16

We also provide corresponding scripts for convenient performance collection and aggregation. 

## Environment setup

We recommend to setup the benchmarking environment within the provided [podman image](https://drive.google.com/file/d/1PjQtYhNYF6nzgLkSI9lrcCkMX9duxo-9/view?usp=drive_link).

```bash
git clone https://github.com/yifeizh2/WorkloadBench.git
cd WorkloadBench
source setup_env.sh
```

The environment setup script will install oneDNN's `benchdnn` and `tcmalloc`.

## Benching oneDNN Primitive individual matmul performance

For this test, we run entire MLP subgraph, collect the performance breakdown of each layer, and aggregate the performance number.
We provide a script for performing performance collection automatically.

```
bash bench_primitive_single_matmul.sh
```

The final result will be written to `primitive_single_matmul.csv`.

## Benching oneDNN Graph Compiler individual matmul performance

For this test, we run entire MLP subgraph with Graph Compiler's coarse grain fusion disabled while invoking Graph Compiler's tracing
utility. The collected trace will further be aggregated and parsed into per-layer data.

```
bash bench_compiler_matmul.sh
```

The final result will be written to `compiler_single_matmul.csv`.

## Benching MLP and MHA subgraph performance

To bench entire subgraph's performance, run the following script.

```
bash bench_subgraph_workloads.sh
```

The performance will be written to `subgraph_perf.csv`.
