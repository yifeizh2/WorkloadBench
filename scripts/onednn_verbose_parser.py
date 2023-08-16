import argparse
import re
import csv

def main():
    args_parser = argparse.ArgumentParser(description='onednn verbose parser')
    args_parser.add_argument('--log_dir', help='path to input files')
    args = args_parser.parse_args()

    # as order of "M, K, N"
    mlp_shapes = [[13, 512, 256, 128], [479, 1024, 1024, 512, 256, 1]]
    batch_sizes = [32, 64, 128, 256, 512]
    matmul_shapes = []
    for mlp_shape in mlp_shapes:
        for i in range(len(mlp_shape) - 1):
            for bs in batch_sizes:
                matmul_shapes.append(", ".join([str(bs), str(mlp_shape[i]), str(mlp_shape[i + 1])]))

    execution_time_dict = {"f32": {}, "int8": {}}
    with open(args.log_dir + "/primitive.log", "r") as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("onednn_graph_verbose,exec,cpu"):
                verbose_seg = line.split(",")
                time = float(verbose_seg[-1])
                shapes = verbose_seg[7]
                result = re.findall(r"([0-9]+)x([0-9]+)", shapes)
                dtype = "int8" if "u8" in shapes else "f32"
                matmul_shape = ", ".join([str(result[0][0]), str(result[0][1]), str(result[1][1])])
                if matmul_shape in execution_time_dict[dtype]:
                    execution_time_dict[dtype][matmul_shape].append(time)
                else:
                    execution_time_dict[dtype][matmul_shape] = [time]
    
    with open(args.log_dir + "/primitive_single_matmul.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["data type", "m, k, n", "time (ms)"])
        for dtype in ["f32", "int8"]:
            for shape in matmul_shapes:
                length = len(execution_time_dict[dtype][shape]) // 10 * 9
                avg_time = sum(execution_time_dict[dtype][shape][-length:]) / length
                writer.writerow([dtype, shape, avg_time])


if __name__ == "__main__":
    main()
