import subprocess

# 启动外部程序（例如，启动 Python 程序）
workload_list = [(16, 128)]
way_list = [2, 4, 8,-1]

for way in way_list:
    for workload in workload_list:
        result = subprocess.run(
            [
                "python",
                "/root/hujianmin/qwen2_moe_i/example.py",
                "--way",
                f"{way}",
                "--batch_size",
                f"{workload[0]}",
                "--input_length",
                f"{workload[1]}",
                '--write',
                '--fused',
                'True',
                '--epoch',
                '30'
            ]
        )
