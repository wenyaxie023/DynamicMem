# /// script
# dependencies = [
#   "vllm>=0.6.0",
# ]
# ///

import os
import subprocess

# 定义模型参数
command = [
    "vllm", "serve", "moonshotai/Kimi-Linear-48B-A3B-Instruct",
    "--port", "8000",
    "--tensor-parallel-size", "4",
    "--max-model-len", "1048576",
    "--trust-remote-code"
]

# 运行服务
subprocess.run(command)