from dataclasses import dataclass
import json


import torch

@dataclass
class GPUManagerConfig:
    gpu_mem_utilization: int 
    max_seqs_in_table: int
    block_size: int
    max_seq_len: int
    max_batch_size:int
    dtype: torch.dtype
    
        
        
        
            

