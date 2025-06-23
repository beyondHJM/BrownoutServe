import dataclasses
import torch

from brownoutserve.brownout_config import BrownoutConfig

@dataclasses.dataclass
class Qwen2MoEInferState:
    batch_size: int
    num_tokens: int

    prefill_seq_ids: torch.Tensor   # [batch_size]
    softmax_scale: float    # Equal to 1/sqrt(head_dim)

    num_prefill_seqs: int
    num_prefill_tokens: int
    cum_prefill_seqs_len: torch.Tensor # [batch_size]
    prefill_seq_start_locs: torch.Tensor # [batch_size]
    prefill_seq_start_locs_with_end: torch.Tensor # [batch_size+1], = prefill_seq_start_locs + [num_prefill_tokens]
    prefill_seq_lens: torch.Tensor # [batch_size]
    max_prefill_len: int
    num_block_per_seq:torch.Tensor
    cum_block_num: torch.Tensor
    num_decoding_seqs: int
    decoding_seq_ids:torch.Tensor
    decoding_seq_lens_before:torch.Tensor # [batch_size]
    decoding_seq_lens: torch.Tensor # [batch_size] decoding_seq_lens = decoding_seq_lens_before + 1
    free_physical_blocks_indices: torch.Tensor
    
    # max_decoding_len: int
    # seq_block_size: int
    # num_seq_blocks: int
    physical_block_ids: torch.Tensor
    position_cos: torch.Tensor	# [num_tokens, hidden_size]
    position_sin: torch.Tensor	# [num_tokens, hidden_size]
    use_kv_cache: bool
    is_warm_up: bool
    residual_buf: torch.Tensor = None
    total_time_usage: float = 0
    moe_time_usage: float = 0
    mode:str = None
    layer_id  :int= -1
    brownout_config: BrownoutConfig = None
    # up: torch.Tensor = None
    # gate : torch.Tensor = None
    # ffnout : torch.Tensor=None


    # ignore_kvcache: bool    # Skip storing the key/value cache, useful when profiling the number of kv blocks
