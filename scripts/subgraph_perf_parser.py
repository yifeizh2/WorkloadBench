import argparse
import csv

def get_perf(filename):
    perf = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("total perf: "):
                perf.append(float(line.split(":")[-1]))
    return perf

def main():
    args_parser = argparse.ArgumentParser(description='benchdnn graph parser')
    args_parser.add_argument('--log_dir', help='path to input files')
    args = args_parser.parse_args()

    workload_names = []
    for dtype in ["F32", "INT8"]:
        for mlp_set in [1, 2]:
            for bs in [32, 64, 128, 256, 512]:
                workload_names.append("MLP_" + str(mlp_set) + "_B" + str(bs) + "_" + dtype)
    for dtype in ["F32", "INT8"]:
        for mha_set in [1, 2, 3, 4]:
            for bs in [32, 64, 128]:
                workload_names.append("MHA_" + str(mha_set) + "_B" + str(bs) + "_" + dtype)
    primitive_perf = get_perf(args.log_dir + "/onednn_primitive_workload_perf.log")
    compiler_perf_wo_coarse = get_perf(args.log_dir + "/compiler_without_coarse_grain_workload_perf.log")
    compiler_perf = get_perf(args.log_dir + "/compiler_workload_perf.log")
    with open(args.log_dir + "/subgraph_perf.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["workload", "oneDNN Primitives + postops",
                         "Anonymous compiler without coarse-grain fusion",
                         "Anonymous compiler"])
        for pack in zip(workload_names, primitive_perf, compiler_perf_wo_coarse, compiler_perf):
            print(pack)
            writer.writerow(list(pack))



if __name__ == "__main__":
    main()
