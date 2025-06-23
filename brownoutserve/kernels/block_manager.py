import torch
import triton
import triton.language as tl


@triton.jit
def _allocate_blocks_for_prefilling(
    logical_slots_indices: torch.Tensor,  # [num_seqs,]
    logical_kv_cache: torch.Tensor,  # [num_seqs,max_num_block_per_seq]
    num_block_per_seq: torch.Tensor,  # [num_seqs]
    cum_block_num: torch.Tensor,  # [num_seqs]
    physical_block_indices: torch.Tensor,  # [num_seqs,]
    k: torch.Tensor,  # [num_tokens,dim]
    v: torch.Tensor,  # [num_tokens,dim]
    physical_k_cache: torch.Tensor,
    physical_v_cache: torch.Tensor,  # [max_blocks_in_all_seq, block_size, dim]
    seqs_len: torch.Tensor,
    cum_seqs_len: torch.Tensor,
    block_size: tl.constexpr,
    max_num_blocks_per_seq: tl.constexpr,
    dim: tl.constexpr,
):
    seq_id = tl.program_id(0)
    block_id = tl.program_id(1)

    cur_seq_block_num = tl.load(num_block_per_seq + seq_id)
    cur_cum_block_num = tl.load(cum_block_num + seq_id)
    if block_id >= cur_seq_block_num:
        return
    cur_logical_slots_index = tl.load(logical_slots_indices + seq_id)

    logical_block_offset = cur_logical_slots_index * max_num_blocks_per_seq + block_id
    cur_physical_block_offset = cur_cum_block_num - cur_seq_block_num + block_id

    physical_kv_cache_block_id = tl.load(physical_block_indices +
                                         cur_physical_block_offset)
    physical_kv_cache_block_offset = (
        physical_kv_cache_block_id * block_size * dim +
        (tl.arange(0, block_size) * dim)[:, None] + tl.arange(0, dim)[None, :])
    # set block mapping
    tl.store(
        logical_kv_cache + logical_block_offset,
        physical_kv_cache_block_id,
    )

    cur_seq_len = tl.load(seqs_len + seq_id)
    cur_cum_seqs_len = tl.load(cum_seqs_len + seq_id)

    tokens_arange = tl.arange(0, block_size)
    mask = (block_id * block_size + tokens_arange < cur_seq_len)[:, None]

    kv_offset = ((cur_cum_seqs_len - cur_seq_len) * dim +
                 block_id * block_size * dim +
                 (tl.arange(0, block_size) * dim)[:, None] +
                 tl.arange(0, dim)[None, :])

    # store k in physical k cache
    tl.store(
        physical_k_cache + physical_kv_cache_block_offset,
        tl.load(k + kv_offset, mask=mask),
        mask=mask,
    )
    # store v in physical v cache
    tl.store(
        physical_v_cache + physical_kv_cache_block_offset,
        tl.load(v + kv_offset, mask=mask),
        mask=mask,
    )


def allocate_blocks_for_prefilling(
    logical_slots_indices: torch.Tensor,  # [num_seqs,]
    logical_k_cache: torch.Tensor,  # [num_seqs,max_num_block_per_seq]
    num_block_per_seq: torch.Tensor,  # [num_seqs]
    cum_block_num: torch.Tensor,  # [num_seqs]
    physical_block_indices: torch.Tensor,  # [num_seqs,]
    k: torch.Tensor,  # [num_tokens,n_kv_heads, head_dim] or [num_tokens, dim]
    v: torch.Tensor,  # [num_tokens,n_kv_heads, head_dim] or [num_tokens, dim]
    physical_k_cache: torch.Tensor,  # [max_blocks_in_all_seq, block_size, n_kv_heads, head_dim] or [max_blocks_in_all_seq, block_size, dim]
    physical_v_cache: torch.Tensor,  # [max_blocks_in_all_seq, block_size, n_kv_heads, head_dim] or [max_blocks_in_all_seq, block_size, dim]
    seqs_len: torch.Tensor,
    cum_seqs_len: torch.Tensor,
    block_size: tl.constexpr,
    max_num_blocks_per_seq: tl.constexpr,
    dim: tl.constexpr, #n_kv_heads* head_dim
):
    assert logical_slots_indices.is_contiguous()
    assert num_block_per_seq.is_contiguous()
    assert cum_block_num.is_contiguous()
    assert logical_k_cache.is_contiguous()
    assert physical_block_indices.is_contiguous()

    grid = (logical_slots_indices.shape[0], max_num_blocks_per_seq)
    # print("logical_slots_indices", logical_slots_indices.tolist())
    # print("num_block_per_seq", num_block_per_seq.tolist())
    # print("cum_block_num", cum_block_num.tolist())
    # print("physical_block_indices", physical_block_indices.tolist())
    # print("seqs_len", seqs_len.tolist())
    # print("cum_seqs_len", cum_seqs_len.tolist())
    # print("max_num_block_per_seq", max_num_blocks_per_seq)
    _allocate_blocks_for_prefilling[grid](
        logical_slots_indices,
        logical_k_cache,
        num_block_per_seq,
        cum_block_num,
        physical_block_indices,
        k,
        v,
        physical_k_cache,
        physical_v_cache,
        seqs_len,
        cum_seqs_len,
        block_size,
        max_num_blocks_per_seq,
        dim,
    )





@triton.jit
def _allocate_blocks_for_decoding(
    k: torch.Tensor,  # [batch_size, dim]
    v: torch.Tensor,  # [batch_size, dim]
    logical_slots_indices: torch.Tensor,  # [batch_size,]
    logical_kv_cache: torch.Tensor,  # [max_batch_size, max_num_blocks_per_seq]
    k_cache: torch.Tensor,  # [max_blocks_in_all_seq, block_size, dim]
    v_cache: torch.Tensor,  # [max_blocks_in_all_seq, block_size, dim]
    new_physical_block_ids: torch.Tensor,  # [batch_size,]
    seqs_len: torch.Tensor,
    dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks_per_seq: tl.constexpr,
):
    seq_id = tl.program_id(0)
    cur_logical_slots_index = tl.load(logical_slots_indices + seq_id)
    logical_slots_start = (
        logical_kv_cache + cur_logical_slots_index * max_num_blocks_per_seq
    )
    cur_seq_len = tl.load(seqs_len + seq_id)
    kv_offset = seq_id * dim + tl.arange(0, dim)
    key = tl.load(k + kv_offset)  # [dim,]
    value = tl.load(v + kv_offset)  # [dim,]
    remaining_tokens_in_block = (cur_seq_len % block_size)
    # print("tt",seq_id)
    # If remaining_tokens_in_block == 0, it means each block is full, and the KV needs to be stored in a new block.
    if remaining_tokens_in_block == 0:
        
        new_logical_block_idx = cur_seq_len // block_size
        physical_block_id = tl.load(new_physical_block_ids + seq_id).to(tl.int64)
        #   set mapping
        tl.store(logical_slots_start + new_logical_block_idx, physical_block_id)
        kv_cache_block_offset = physical_block_id * block_size * dim + tl.arange(0, dim)

    else:
       
        last_logical_block_id = (cur_seq_len - 1) // block_size
        physical_block_id = tl.load(logical_slots_start + last_logical_block_id).to(tl.int64)
        kv_cache_block_offset = (
            physical_block_id * block_size * dim
            + remaining_tokens_in_block * dim
            + tl.arange(0, dim)
        )

    # store k in physical k cache
    tl.store(
        k_cache + kv_cache_block_offset,
        key,
    )
    # store v in physical v cache
    tl.store(
        v_cache + kv_cache_block_offset,
        value,
    )


def allocate_blocks_for_decoding(
    k: torch.Tensor,
    v: torch.Tensor,
    logical_slots_indices: torch.Tensor,
    logical_kv_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    new_physical_block_ids: torch.Tensor,
    seqs_len: torch.Tensor,
    dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks_per_seq: tl.constexpr,
):
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert logical_slots_indices.is_contiguous()
    assert logical_kv_cache.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert new_physical_block_ids.is_contiguous()
    assert seqs_len.is_contiguous()
  
    grid = (len(seqs_len),)
    _allocate_blocks_for_decoding[grid](
        k,
        v,
        logical_slots_indices,
        logical_kv_cache,
        k_cache,
        v_cache,
        new_physical_block_ids,
        seqs_len,
        dim,
        block_size,
        max_num_blocks_per_seq,
    )

@triton.jit
def _free_physical_blocks(
    logical_slots_indices:torch.Tensor,
    num_block_per_seq: torch.Tensor,
    allocated_physical_blocks_indices:torch.Tensor,
    logical_kv_cache: torch.Tensor,
    max_num_blocks_per_seq: tl.constexpr

):
    seq_id = tl.program_id(0)
    block_id = tl.program_id(1)
    cur_logical_slots_index = tl.load(logical_slots_indices + seq_id)
    num_block = tl.load(num_block_per_seq+cur_logical_slots_index )
    if block_id < num_block:
        physical_block_id = tl.load(logical_kv_cache+cur_logical_slots_index*max_num_blocks_per_seq+block_id)
        tl.store(allocated_physical_blocks_indices+physical_block_id,0)


def free_physical_blocks(
        seq_ids: torch.Tensor,
        num_block_per_seq: torch.Tensor,
        allocated_physical_blocks_indices:torch.Tensor,
        logical_kv_cache: torch.Tensor,
        max_num_blocks_per_seq:int
):
    assert seq_ids.is_contiguous()
    assert num_block_per_seq.is_contiguous()
    assert allocated_physical_blocks_indices.is_contiguous()
    assert logical_kv_cache.is_contiguous() 
    max_num_block = int(torch.max(num_block_per_seq).item())
    # print('max_num_block',max_num_block)
    grid=(seq_ids.shape[0],max_num_block )
    # print('grid',grid)
    _free_physical_blocks[grid](seq_ids,num_block_per_seq,allocated_physical_blocks_indices,logical_kv_cache,max_num_blocks_per_seq)