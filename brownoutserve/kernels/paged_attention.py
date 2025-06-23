import time
import torch
import triton
from triton import language as tl


@triton.jit
def _attention_decoding(
    q_ptr: torch.Tensor, # [batch_size, 1, dim] or [batch_size, dim],[batch_size, 1, n_q_heads,head_dim]
    output: torch.Tensor,  # [batch_size, 1, dim] or [batch_size, dim]
    k_cache: torch.Tensor,  # [num_blocks, block_size,dim]
    v_cache: torch.Tensor,  # [num_blocks, block_size,dim]
    logical_slots_indices: torch.Tensor,  # [batch_size,]
    logical_kv_cache: torch.Tensor,  # [max_batch_size,max_num_block_per_seq]
    seqs_len: torch.Tensor,  # [batch_size]
    scaling: float,
    block_size: tl.constexpr,
    n_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    n_rep: tl.constexpr,
    head_dim: tl.constexpr,
    max_num_blocks_per_seq: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    q_offset = seq_id * n_heads * head_dim + head_id * head_dim + tl.arange(0, head_dim)
    q = tl.load(q_ptr + q_offset) * scaling  # [head_dim,]
    cur_seq_len = tl.load(seqs_len + seq_id)
    num_blocks = (cur_seq_len - 1 + block_size) // block_size

    logical_slot = tl.load(logical_slots_indices + seq_id)
    logical_slot_start_offset = logical_slot * max_num_blocks_per_seq
    m = tl.full((), value=-float("inf"), dtype=tl.float32)
    z = tl.full((), value=0, dtype=tl.float32)
    acc = tl.full((head_dim,), value=0, dtype=tl.float32)
    
    kv_head_id = head_id // n_rep
    for start_pos in range(0, num_blocks):
        physical_block_id = tl.load(
            logical_kv_cache + logical_slot_start_offset + start_pos
        )

        physical_k_block = tl.load(
            k_cache
            + physical_block_id * block_size * n_kv_heads * head_dim
            + tl.arange(0, block_size)[:, None] * n_kv_heads * head_dim
            + kv_head_id * head_dim
            + tl.arange(0, head_dim)[None, :]
        )  # [block_size,head_dim]
        physical_v_block = tl.load(
            v_cache
            + physical_block_id * block_size * n_kv_heads * head_dim
            + tl.arange(0, block_size)[:, None] * n_kv_heads * head_dim
            + kv_head_id * head_dim
            + tl.arange(0, head_dim)[None, :]
        )  # [block_size,head_dim]

        qk = tl.sum(
            q[None, :] * physical_k_block, axis=1
        )  # q[None,:].shape is [1,head_dim]    q[None,:]*physical_k_block results'shape is [block_size, head_dim], qk shape is [block_size,]
        mask = (start_pos * block_size + tl.arange(0, block_size)) < cur_seq_len
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m, tl.max(qk, axis=0))
        exp = tl.exp(qk - m_new)  # [block_size,]
        
        acc = acc * tl.exp(m - m_new) + tl.sum(
            exp[:, None] * physical_v_block, axis=0
        )  # [head_dim]
        z = z * tl.exp(m - m_new) + tl.sum(exp, axis=0)
        m = m_new

    tl.store(
        output
        + seq_id * n_heads * head_dim
        + head_id * head_dim
        + tl.arange(0, head_dim),
        acc / z,
    )  # 0 <= head_id <= n_heads -1


def attention_decoding(
    q: torch.Tensor,  # [batch_size, 1, n_head, head_dim] or [batch_size, n_head, head_dim]
    output: torch.Tensor,  # [batch_size, 1, dim] or [batch_size, dim]
    k_cache: torch.Tensor,  # [num_blocks, block_size,dim]
    v_cache: torch.Tensor,  # [num_blocks, block_size,dim]
    logical_slots_indices: torch.Tensor,  # [batch_size,]
    logical_kv_cache: torch.Tensor,  # [max_batch_size,max_num_block_per_seq]
    seqs_len: torch.Tensor,  # [max_batch_size]
    block_size: tl.constexpr,
    n_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    max_nums_block_per_seq: tl.constexpr,
):
    bsz = q.shape[0]
    assert q.is_contiguous()
    # assert output.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert logical_slots_indices.is_contiguous()
    assert logical_kv_cache.is_contiguous()
    assert seqs_len.is_contiguous()
    # torch.cuda.synchronize()
    # t1=time.perf_counter()
    _attention_decoding[(bsz, n_heads)](
        q,
        output,
        k_cache,
        v_cache,
        logical_slots_indices,
        logical_kv_cache,
        seqs_len,
        head_dim**-0.5,
        block_size,
        n_heads,
        n_kv_heads,
        n_heads // n_kv_heads,
        head_dim,
        max_nums_block_per_seq,
        num_warps = 1,
        num_stages = 8
    )
    # torch.cuda.synchronize()
    # t2=time.perf_counter()
    # print(torch.max(seqs_len),t2-t1)





@triton.jit
def _attention_decoding_for_long_seq_compute(
    q_ptr: torch.Tensor,  # [batch_size, 1, dim] or [batch_size, dim],[batch_size, 1, n_q_heads,head_dim]
    # output: torch.Tensor,  # [batch_size, 1, dim] or [batch_size, dim]
    mid_o: torch.Tensor,  # [batch_size, n_heads, max_num_region_per_seq, head_dim] max_num_region_per_seq = (max_seq_len + region_size - 1 // region_size)
    mid_o_logexpsum: torch.Tensor,  # [batch_size, n_heads, max_num_region_per_seq]
    k_cache: torch.Tensor,  # [num_blocks, block_size,dim]
    v_cache: torch.Tensor,  # [num_blocks, block_size,dim]
    logical_slots_indices: torch.Tensor,  # [batch_size,]
    logical_kv_cache: torch.Tensor,  # [max_batch_size,max_num_block_per_seq]
    seqs_len: torch.Tensor,  # [batch_size]
    max_num_region_per_seq: int,
    scaling: float,
    block_size: tl.constexpr,
    region_size: tl.constexpr,
    n_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    n_rep: tl.constexpr,
    head_dim: tl.constexpr,
    max_num_block_per_seq: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    region_id = tl.program_id(2)
    q_offset = seq_id * n_heads * head_dim + head_id * head_dim + tl.arange(0, head_dim)
    q = tl.load(q_ptr + q_offset) * scaling  # [head_dim,]
    cur_seq_len = tl.load(seqs_len + seq_id)
    logical_slot = tl.load(logical_slots_indices + seq_id)
    logical_slot_start_offset = logical_slot * max_num_block_per_seq
    num_blocks_per_region = region_size // block_size
    # region_start = logical_slot_start_offset + region_id * num_blocks_per_region

    m = tl.full((), value=-float("inf"), dtype=tl.float32)
    z = tl.full((), value=0, dtype=tl.float32)
    acc = tl.full((head_dim,), value=0, dtype=tl.float32)
    kv_head_id = head_id // n_rep

    region_start =  region_id * num_blocks_per_region
    num_tokens_before = region_start*block_size
    logical_block_base_addr = logical_kv_cache + logical_slot_start_offset + region_start
    k_base_addr = k_cache+tl.arange(0, block_size)[:, None] * n_kv_heads * head_dim+ kv_head_id * head_dim + tl.arange(0, head_dim)[None, :]
    v_base_addr = v_cache+tl.arange(0, block_size)[:, None] * n_kv_heads * head_dim+ kv_head_id * head_dim + tl.arange(0, head_dim)[None, :]
    block_dim = block_size * n_kv_heads * head_dim
    
    mask_arange = tl.arange(0, block_size)
    for start_pos in range(0,num_blocks_per_region):
        
        physical_block_id = tl.load(
            logical_block_base_addr + start_pos
        )

        physical_k_block = tl.load(
            k_base_addr
            + physical_block_id * block_dim
        )  # [block_size,head_dim]
        physical_v_block = tl.load(
            v_base_addr
            + physical_block_id * block_dim
        )  # [block_size,head_dim]

        qk = tl.sum(
            q[None, :] * physical_k_block, axis=1
        )  # q[None,:].shape is [1,head_dim]    q[None,:]*physical_k_block results'shape is [block_size, head_dim], qk shape is [block_size,]
        
        mask = ( num_tokens_before+start_pos * block_size + mask_arange) < cur_seq_len

        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m, tl.max(qk, axis=0))
        exp = tl.exp(qk - m_new)  # [block_size,]
        
        acc = acc * tl.exp(m - m_new) + tl.sum(
            exp[:, None] * physical_v_block, axis=0
        )  # [head_dim]
        z = z * tl.exp(m - m_new) + tl.sum(exp, axis=0)
        m = m_new
  
        # mid_o: torch.Tensor, [batch_size, n_heads, max_num_region_per_seq, head_dim] max_num_region_per_seq = (max_seq_len + region_size - 1 // region_size)
    mid_o_offsets = (
            seq_id * n_heads * max_num_region_per_seq * head_dim
            + head_id * max_num_region_per_seq * head_dim
            + region_id * head_dim
            + tl.arange(0, head_dim)
        )
    
    tl.store(mid_o + mid_o_offsets, acc/z)
    # mid_o_logexpsum: torch.Tensor, #[batch_size, n_heads, max_num_region_per_seq]
    mid_o_logexpsum_offsets = (
        seq_id * n_heads * max_num_region_per_seq
        + head_id * max_num_region_per_seq
        + region_id
    )
    # test = tl.full((),value=-1*seq_id,dtype=tl.float32)
    # print("aa",tl.log(z) + m)
    tl.store(mid_o_logexpsum + mid_o_logexpsum_offsets, tl.log(z) + m)


@triton.jit
def _attention_decoding_for_long_seq_reduce(
    mid_o: torch.Tensor,  # [batch_size, n_heads, max_num_region_per_seq, head_dim], contiguous
    mid_o_logexpsum: torch.Tensor,  # [batch_size, n_heads, max_num_region_per_seq], contiguous
    output: torch.Tensor,  # [batch_size, n_heads, head_dim], contiguous
    seqs_len: torch.Tensor,  # [batch_size], contiguous
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    max_num_region_per_seq: tl.constexpr,
    region_size: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    cur_seq_len = tl.load(seqs_len + seq_id)
    num_region = tl.cdiv(cur_seq_len, region_size)
    m = tl.full((), value=-float("inf"), dtype=tl.float32)
    z = tl.full((), value=0, dtype=tl.float32)
    acc = tl.full((head_dim,), value=0, dtype=tl.float32)
    for region_id in range(0, num_region):
        mid_o_logexpsum_offset = (
            seq_id * n_heads * max_num_region_per_seq
            + head_id * max_num_region_per_seq
            + region_id
        )
        cur_logexpsum = tl.load(mid_o_logexpsum + mid_o_logexpsum_offset)
        mid_o_offsets = (
            seq_id * n_heads * max_num_region_per_seq * head_dim
            + head_id * max_num_region_per_seq * head_dim
            + region_id * head_dim
            + tl.arange(0, head_dim)
        )
        cur_o = tl.load(mid_o + mid_o_offsets)
        m_new = tl.maximum(m, cur_logexpsum)
        scale1 = tl.exp(m - m_new)
        scale2 = tl.exp(cur_logexpsum - m_new)
        acc = acc * scale1 + cur_o * scale2
        z = z * scale1 + scale2
        m = m_new

    tl.store(
        output
        + seq_id * n_heads * head_dim
        + head_id * head_dim
        + tl.arange(0, head_dim),
        acc / z,
    )  # 0 <= head_id <= n_heads -1



def attention_decoding_for_long_seq(
    q: torch.Tensor,  # [batch_size, 1, dim] or [batch_size, dim]
    output: torch.Tensor,  # [batch_size, 1, dim] or [batch_size, dim]
    k_cache: torch.Tensor,  # [num_blocks, block_size,dim]
    v_cache: torch.Tensor,  # [num_blocks, block_size,dim]
    logical_slots_indices: torch.Tensor,  # [batch_size,]
    logical_kv_cache: torch.Tensor,  # [max_batch_size,max_num_block_per_seq]
    seqs_len: torch.Tensor,  # [max_batch_size]
    block_size: tl.constexpr,
    region_size: tl.constexpr,
    n_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    max_num_block_per_seq: tl.constexpr,
    max_num_region_per_seq: tl.constexpr,  # Here seq refers to max_seq_len instead of cur_seq_len
):
    bsz = q.shape[0]

    # mid_o: torch.Tensor,  # [batch_size, n_heads, num_region_per_seq, head_dim] num_region_per_seq = (max_seq_len + region_size - 1 // region_size)
    # mid_o_logexpsum: torch.Tensor,  # [batch_size, n_heads, num_region_per_seq]
    mid_o = torch.zeros(
        bsz, n_heads, max_num_region_per_seq, head_dim, device=q.device, dtype=torch.float32
    )
    mid_o_logexpsum = torch.zeros(
        bsz, n_heads, max_num_region_per_seq, device=q.device, dtype=torch.float32
    )
    grid = (bsz, n_heads, max_num_region_per_seq)
    # print('grid is',grid )
    # print('mid_o_logexpsum',mid_o_logexpsum.tolist())
    # print('mid_o',mid_o.tolist())
    _attention_decoding_for_long_seq_compute[grid](
        q,
        mid_o,
        mid_o_logexpsum,
        k_cache,
        v_cache,
        logical_slots_indices,
        logical_kv_cache,
        seqs_len,
        max_num_region_per_seq,
        head_dim**-0.5,
        block_size,
        region_size,
        n_heads,
        n_kv_heads,
        n_heads // n_kv_heads,
        head_dim,
        max_num_block_per_seq,
        # num_warps = 1,
        # num_stages = 4
    )
    # print('mid_o_logexpsum',mid_o_logexpsum.tolist())
    # print('mid_o',mid_o.tolist())
    grid = (bsz, n_heads,)   
    _attention_decoding_for_long_seq_reduce[grid](
        mid_o,
        mid_o_logexpsum,
        output,
        seqs_len,
        n_heads,
        head_dim,
        max_num_region_per_seq,
        region_size,
    )
