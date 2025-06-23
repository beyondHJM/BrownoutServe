# BrownoutServe: SLO-Aware Serving Approach with Brownout for MoE-based LLMs

[![GitHub](https://img.shields.io/github/stars/beyondHJM/BrownoutServe?style=social)](https://github.com/beyondHJM/BrownoutServe)
[![GitHub license](https://img.shields.io/github/license/beyondHJM/BrownoutServe)](https://github.com/beyondHJM/BrownoutServe/blob/main/LICENSE.txt)

## Introduction

`BrownoutServe` is an innovative serving framework designed for Mixture-of-Experts (MoE) based Large Language Models (LLMs). It optimizes inference efficiency and maintains service reliability under dynamic computational demands and workload conditions by introducing United Experts and a dynamic Brownout mechanism. Additionally, it integrates state-of-the-art optimization techniques such as PagedAttention, FlashAttention, and ContinuousBatching to further enhance performance.

## Features

### Core Features

- **United Experts**: Integrates knowledge from multiple expert models into a single "united expert" to reduce the number of expert accesses during inference, improving GPU parallelism and reducing latency.
- **Brownout Mechanism**: Dynamically routes a subset of tokens to united experts under resource constraints or burst workloads, reducing expert access overhead and helping maintain SLO attainment.
- **SLO-Aware Latency Control Algorithm**: Real-time adjusts the Brownout configuration to minimize accuracy loss while ensuring inference latency meets SLO, even under dynamic request environments and sudden load spikes.

### Integrated Optimizations

- **PagedAttention(proposed in [SOSP'23 vLLM](https://arxiv.org/abs/2309.06180))**: Optimizes KV cache management by using a paged approach to reduce GPU memory fragmentation. This technique is adapted from vLLM and further optimized in `BrownoutServe` by moving the block table to the GPU.
- **FlashAttention(proposed in [NeurIPS'22 FlashAttention]( https://arxiv.org/abs/2205.14135))**: Accelerates attention computation by leveraging efficient memory access patterns and reducing computational overhead. This technique is integrated into the inference engine to improve overall performance.
- **Continuous Batching(proposed in [OSDI'22 Orca](https://www.usenix.org/conference/osdi22/presentation/yu))**: Enhances throughput by allowing dynamic insertion and removal of requests in the batch, enabling more efficient use of computational resources. This technique is adapted from Orca and integrated into `BrownoutServe` to improve scheduler performance.

### Performance Highlights

- Achieves up to 2.07× higher throughput compared to vLLM.
- Reduces SLO violation periods by 90.28% under bursty traffic conditions.
- Maintains acceptable inference accuracy with minimal loss (around 5%).

## Current Limitations

- **Model Support**: Currently, `BrownoutServe` only supports the `Qwen1.5-MoE-A2.7B-Chat` model. This limitation is primarily due to time constraints and the high human resource costs associated with adapting the system to additional models. We plan to address this limitation in future releases by expanding support to other models.
- **Code and Documentation**: The codebase of `BrownoutServe` is still under active development.The current code may lack enough comments, and some functionalities may not be fully implemented. Additionally, the documentation is not yet comprehensive. We are working to improve the code quality, add more comments, and complete the documentation in upcoming updates.
## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.40+
- CUDA 11.0+
- NVIDIA GPU (A100-40GB recommended)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/beyondHJM/BrownoutServe.git
   cd BrownoutServe
2. Install dependencies and environments:
    ```bash
    pip install -r requirements.txt
    pip install e .
3. Run an example:
    ```bash
    cd examples
    python chatbot.py