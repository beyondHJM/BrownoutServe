import torch
import triton
import triton.language as tl

# copy from swiftLLM (https://github.com/interestingLSY/swiftLLM)
@triton.jit
def _fwd_silu_and_mul_inplace(
    x: torch.Tensor,	# [num_tokens, 2*ffn_inter_dim]. Result will be stored at input[:, :ffn_inter_dim]
    ffn_inter_dim: tl.constexpr,
    block_size: tl.constexpr
):
    # grid shape: [num_tokens, ffn_inter_dim / block_size]
    # require ffn_inter_dim % block_size == 0
    my_token_id = tl.program_id(0).to(tl.int64)
    my_block_id = tl.program_id(1)
    offsets = my_block_id*block_size + tl.arange(0, block_size)
    mask = offsets<ffn_inter_dim
    offs = my_token_id*(2*ffn_inter_dim) + offsets
    gate = tl.load(x + (offs+ffn_inter_dim),mask=mask)
    gate = gate.to(tl.float32)
    gate = gate / (1 + tl.exp(-gate))
    gate = gate.to(tl.float16)
    up = tl.load(x + offs,mask=mask)
    down = up *gate
    tl.store(x + offs, down,mask=mask)

@triton.jit
def _fwd_silu_and_mul(
    x: torch.Tensor,	# [num_tokens, 2*ffn_inter_dim]
    out:torch.Tensor,# [num_tokens, ffn_inter_dim]
    ffn_inter_dim: tl.constexpr,
    block_size: tl.constexpr,
    
):
    # grid shape: [num_tokens, ffn_inter_dim / block_size]
    # require ffn_inter_dim % block_size == 0
    my_token_id = tl.program_id(0).to(tl.int64)
    my_block_id = tl.program_id(1)

    offsets = my_block_id*block_size + tl.arange(0, block_size)
    mask = offsets<ffn_inter_dim
    offs_inp = my_token_id*(2*ffn_inter_dim) + offsets
    offs_out = my_token_id*ffn_inter_dim + offsets
    gate = tl.load(x + (offs_inp+ffn_inter_dim),mask=mask)
    gate = gate.to(tl.float32)
    gate = gate / (1 + tl.exp(-gate))
    gate = gate.to(tl.float16)
    up = tl.load(x + offs_inp,mask=mask)
    down = up *gate
    tl.store(out + offs_out, down,mask=mask)

def silu_and_mul_inplace(
    x: torch.Tensor # [num_tokens, 2*ffn_inter_dim]
):
    assert x.is_contiguous()
    num_tokens = x.shape[0]
    ffn_inter_dim = x.shape[1] // 2

    block_size = 8
    assert ffn_inter_dim % block_size == 0
    _fwd_silu_and_mul_inplace[(num_tokens, ffn_inter_dim//block_size)](x, ffn_inter_dim, block_size)

def silu_and_mul(
        x:torch.Tensor, # [num_tokens, 2*ffn_inter_dim]
        

):
    assert x.is_contiguous()
    num_tokens = x.shape[0]
    ffn_inter_dim = x.shape[1] // 2
    out = torch.zeros(num_tokens,ffn_inter_dim,device=x.device,dtype=x.dtype)
    block_size = 256
    # assert ffn_inter_dim % block_size == 0
    _fwd_silu_and_mul[(num_tokens, (ffn_inter_dim+block_size-1)//block_size)](x,out, ffn_inter_dim, block_size)
    return out