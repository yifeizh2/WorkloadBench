import argparse
import re
import csv
import subprocess

def main():
    args_parser = argparse.ArgumentParser(description='trace parser')
    args_parser.add_argument('--workload_bench_dir', help='path to trace files')
    args = args_parser.parse_args()

    workload_names = []
    workload_shapes = []
    mlp_shapes = [[13, 512, 256, 128], [479, 1024, 1024, 512, 256, 1]]
    batch_sizes = [32, 64, 128, 256, 512]
    for dtype in ["F32", "INT8"]:
        for mlp_set in [1, 2]:
            for bs in batch_sizes:
                workload_names.append("MLP_" + str(mlp_set) + "_B" + str(bs) + "_" + dtype)
                workload_shapes.append([bs, mlp_shapes[mlp_set - 1]])

    matmul_shapes = []
    for mlp_shape in mlp_shapes:
        for i in range(len(mlp_shape) - 1):
            for bs in batch_sizes:
                matmul_shapes.append(", ".join([str(bs), str(mlp_shape[i]), str(mlp_shape[i + 1])]))

    execution_time_dict = {"f32": {}, "int8": {}}
    for i in range(len(workload_names)):
        trace_path = args.workload_bench_dir + "/" + workload_names[i] + ".json"
        cmd = ["python", args.workload_bench_dir + "/scripts/trace_analyzer.py", "--file", trace_path, "--out", "csv"]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
            partition_time = []
            for out_line in p.stdout:
                if "managed_matmul_core" in out_line:
                    seg = out_line.split(",")
                    ticks = float(seg[1])
                    calls = float(seg[-1])
                    partition_time.append([seg[0], ticks / calls / 1e3])
            def rule(element):
                return element[0].split("_")[-1].split("@")[0]
            partition_time.sort(key=rule)
            print(partition_time)
            print(workload_shapes[i])
            for layer in range(len(partition_time)):
                matmul_shape = ", ".join([str(workload_shapes[i][0]), str(workload_shapes[i][1][layer]),
                                         str(workload_shapes[i][1][layer + 1])])
                print(matmul_shape)
                dtype = "int8" if "INT8" in workload_names[i] else "f32"
                if matmul_shape in execution_time_dict[dtype]:
                    continue
                execution_time_dict[dtype][matmul_shape] = partition_time[layer][1]

    
    with open(args.workload_bench_dir + "/compiler_single_matmul.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["data type", "m, k, n", "time (ms)"])
        for dtype in ["f32", "int8"]:
            for shape in matmul_shapes:
                writer.writerow([dtype, shape, execution_time_dict[dtype][shape]])


if __name__ == "__main__":
    main()
