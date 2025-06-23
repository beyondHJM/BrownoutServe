from datetime import datetime
import subprocess

# 启动外部程序（例如，启动 Python 程序）
rps_list = [1,2,3,4]
workload_list = [(8, 1),(2, 0), (4, 0.2), (8, 0.4)]

# rps_list = [1,2,4,5,7,8,10,11,13,14]
for i ,workload in enumerate(workload_list):
    for j,rps in enumerate(rps_list):
        way, threshold = workload[0], workload[1]
        result = subprocess.run(
            [
                "python",
                "/root/hujianmin/qwen2_moe_i/online.py",
                "--way",
                f"{way}",
                "--max_batch_size",
                "64",
                "--threshold",
                f"{threshold}",
                "--rps",
                f"{rps}",
                "--duration",
                "30",
                "--output_length",
                "2048",
                # "--file_path",
                # "/root/hujianmin/qwen2_moe_i/result/unfused_throughput_alpaca.json"
            ]
        )

for i ,workload in enumerate(workload_list):
    for j,rps in enumerate(rps_list):
        way, threshold = workload[0], workload[1]
        result = subprocess.run(
            [
                "python",
                "/root/hujianmin/qwen2_moe_i/online.py",
                "--way",
                f"{way}",
                "--max_batch_size",
                "64",
                "--threshold",
                f"{threshold}",
                "--rps",
                f"{rps}",
                "--duration",
                "100",
                "--output_length",
                "2048",
                # "--file_path",
                # "/root/hujianmin/qwen2_moe_i/result/unfused_throughput_alpaca.json"
            ]
        )

# rps_list = [55,60]
# workload_list = [(8, 1),(2, 0), (4, 0.2), (8, 0.4)]

# rps_list = [9,12]
# for i ,workload in enumerate(workload_list):
#     for j,rps in enumerate(rps_list):
#         way, threshold = workload[0], workload[1]
#         result = subprocess.run(
#             [
#                 "python",
#                 "/root/hujianmin/qwen2_moe_i/online.py",
#                 "--way",
#                 f"{way}",
#                 "--max_batch_size",
#                 "64",
#                 "--threshold",
#                 f"{threshold}",
#                 "--rps",
#                 f"{rps}",
#                 "--duration",
#                 "100",
#                 "--output_length",
#                 "2048",
#                 "--use_fused_moe",
#                 # "--file_path",
#                 # "/root/hujianmin/qwen2_moe_i/result/fused_throughput_alpaca.json"
#             ]
#         )
# for workload in workload_list:
#     way,threshold = workload[0],workload[1]
#     for rps in rps_list:
#         result = subprocess.run(
#             [
#                 "python",
#                 "/root/hujianmin/qwen2_moe_i/online.py",
#                 "--way",
#                 f"{way}",
#                 "--threshold",
#                 f"{threshold}",
#                 "--rps",
#                 f"{rps}",
#                 "--duration",
#                 "100",
#                 "--use_fused_moe",
#                 "--output_length",
#                 "128",
#             ]
#         )
