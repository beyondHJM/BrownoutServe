import time
import torch
import triton
import triton.language as tl
from brownoutserve.infer_state import Qwen2MoEInferState
from brownoutserve.brownout_config import BrownoutConfig
from brownoutserve.brownout_mask_map import BrownoutMaskMap

silu = torch.nn.SiLU()


class Weight:
    def __init__(self):
        self.weights = torch.load("/root/hujianmin/fused_moe_test/experts_weight.pt")

    def get_up(self, expert_id: int) -> torch.Tensor:
        k = f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight"
        return self.weights[k]

    def get_gate(self, expert_id: int) -> torch.Tensor:
        k = f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
        return self.weights[k]

    def get_down(self, expert_id: int) -> torch.Tensor:
        k = f"model.layers.0.mlp.experts.{expert_id}.down_proj.weight"
        return self.weights[k]

    def concat_up(self):
        up_list = []
        for i in range(60):
            up_list.append(self.get_up(i))

        return torch.concat(up_list, dim=0)  # [num_experts*intermediate, dim]

    def concat_gate(self):
        gate_list = []
        for i in range(60):
            gate_list.append(self.get_gate(i))

        return torch.concat(gate_list, dim=0)  # [num_experts*intermediate, dim]

    def concat_down(self):
        down_list = []
        for i in range(60):
            down_list.append(self.get_down(i))

        return torch.concat(down_list, dim=0)  # [num_experts*intermediate, dim]


@triton.jit
def calculate_token_offsets_kernel(
    topk_ids: torch.Tensor,
    topk_ids_off: torch.Tensor,
    token_num,
    BLOCK_SIZE: tl.constexpr,
):
    # grid = [triton.cdiv(token_num,BLOCK_SIZE)]
    cur_block_id = tl.program_id(0)
    token_ids_offsets = tl.arange(0, BLOCK_SIZE)
    cur_token_ids_offsets = cur_block_id * BLOCK_SIZE + token_ids_offsets
    token_mask = cur_token_ids_offsets < token_num
    cur_token_ids = tl.load(topk_ids + cur_token_ids_offsets, mask=token_mask)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for block_id in range(0, cur_block_id):
        token_ids_in_range = tl.load(
            topk_ids + BLOCK_SIZE * block_id + token_ids_offsets
        )  # [BLOCK_SIZE]
        acc += tl.sum(
            tl.where(token_ids_in_range[None, :] == cur_token_ids[:, None], 1, 0),
            axis=1,
        )

    condition = (cur_token_ids[None, :] == cur_token_ids[:, None]) & (
        token_ids_offsets[:, None] > token_ids_offsets[None, :]
    )
    acc += tl.sum(tl.where(condition, 1, 0), axis=1)
    tl.store(topk_ids_off + cur_token_ids_offsets, acc, mask=token_mask)


def calculate_token_offsets(topk_ids: torch.Tensor, topk_ids_off: torch.Tensor):
    BLOCK_SIZE = 128
    grid = (triton.cdiv(topk_ids.shape[0], BLOCK_SIZE),)
    calculate_token_offsets_kernel[grid](
        topk_ids, topk_ids_off, topk_ids.shape[0], BLOCK_SIZE
    )


@triton.jit
def _moe_align_block_size(
    topk_ids: torch.Tensor,  # [num_tokens]
    topk_ids_padded: torch.Tensor,  # [max_num_tokens_padded],output
    num_blocks_per_expert_offset: torch.Tensor,  # [num_experts+1]
    topk_ids_off_list: torch.Tensor,  # [num_tokens]
    expert_ids: torch.Tensor,  # [max_num_tokens_padded // block_size]
    BLOCK_SIZE: tl.constexpr,
):
    token_id = tl.program_id(0)
    expert_id = tl.load(topk_ids + token_id)
    expert_block_start = tl.load(num_blocks_per_expert_offset + expert_id)
    offset_in_block_range = tl.load(topk_ids_off_list + token_id)

    tl.store(
        topk_ids_padded + expert_block_start * BLOCK_SIZE + offset_in_block_range,
        token_id,
    )
    tl.store(
        expert_ids + expert_block_start + (offset_in_block_range // BLOCK_SIZE),
        expert_id,
    )


@triton.jit
def __fused_matrix_multiple(
    hidden_states: torch.Tensor,  # [num_tokens,dim]
    outputs: torch.Tensor,  # [num_tokens,intermediate]
    sorted_ids: torch.Tensor,
    w: torch.Tensor,  # [num_experts*intermediate,dim]
    expert_ids: torch.Tensor,  # [num_experts+1]
    # expert_ids: torch.Tensor,  # [num_experts],
    dim,
    intermediate_size,
    PADDING_ID: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 512,
):

    group_id = tl.program_id(0)
    block_id = tl.program_id(1)
    expert_id = tl.load(expert_ids + group_id)
    token_ids = tl.load(
        sorted_ids + group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    )  # [GROUP_SIZE]
    token_mask = token_ids != PADDING_ID  # [GROUP_SIZE]
    # inp = tl.load(hidden_states+token_ids,mask=token_mask) #[GROUP_SIZE,dim]
    accumulator = tl.zeros((GROUP_SIZE, BLOCK_SIZE_M), dtype=tl.float32)
    inp_offsets = token_ids[:, None] * dim + tl.arange(0, BLOCK_SIZE_K)[None, :]

    w_offsets = (
        expert_id * intermediate_size * dim
        + block_id * BLOCK_SIZE_M * dim
        + tl.arange(0, BLOCK_SIZE_M)[None, :] * dim
        + tl.arange(0, BLOCK_SIZE_K)[:, None],
    )

    weight_mask = (
        block_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    ) < intermediate_size
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    for k in range(0, tl.cdiv(dim, BLOCK_SIZE_K)):

        inp = tl.load(
            hidden_states + inp_offsets,
            mask=token_mask[:, None] & (k_offsets[None, :] < dim - k * BLOCK_SIZE_K),
            other=0.0,
            cache_modifier=".cg",
        )  # [GROUP_SIZE,BLOCK_SIZE_K]

        w_partial_T = tl.load(
            w + w_offsets,
            mask=weight_mask[None, :] & (k_offsets[:, None] < dim - k * BLOCK_SIZE_K),
            other=0.0,
            cache_modifier=".cg",
        )
        o = tl.dot(
            inp, w_partial_T
        )  # [GROUP_SIZE,BLOCK_SIZE_M], [GROUP_SIZE,dim] * [dim,BLOCK_SIZE_M] = [GROUP_SIZE,BLOCK_SIZE_M]
        accumulator += o
        inp_offsets += BLOCK_SIZE_K
        w_offsets += BLOCK_SIZE_K

    o_offsets = (
        token_ids[:, None] * intermediate_size
        + block_id * BLOCK_SIZE_M
        + tl.arange(0, BLOCK_SIZE_M)[None, :]
    )

    tl.store(
        outputs + o_offsets,
        accumulator,
        mask=token_mask[:, None] & weight_mask[None, :],
    )


@triton.jit
def div_add_kernel(
    input_ptr,  # 输入张量的指针
    n_elements, # 张量的大小
    N,          # 整除的参数
    M:tl.constexpr,# 加的参数
    BLOCK_SIZE: tl.constexpr,  # 每个程序应该处理的元素数量
):
    pid = tl.program_id(axis=0)  # 并行处理的一维启动
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 从DRAM加载数据
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # 计算: (x // N) + M
    output = (x // N) + M
    
    # 将结果写回DRAM
    tl.store(input_ptr + offsets, output, mask=mask)

def div_add(input_tensor: torch.Tensor, N: int, M: int) -> torch.Tensor:
    # 预分配输出


    BLOCK_SIZE=512
    grid = (triton.cdiv(input_tensor.shape[0], BLOCK_SIZE)), 
    
    # 启动内核
    div_add_kernel[grid](
        input_tensor, input_tensor.shape[0],N, M, 
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
@triton.jit
def div_add_v2_kernel(
    input_ptr,  # 输入张量的指针
    update_indices,
    n_elements, # 张量的大小
    N,          # 整除的参数
    M:tl.constexpr,# 加的参数
    BLOCK_SIZE: tl.constexpr,  # 每个程序应该处理的元素数量
):
    pid = tl.program_id(axis=0)  # 并行处理的一维启动
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 从DRAM加载数据
    x = tl.load(input_ptr + offsets, mask=mask) #[BLOCK_SIZE]    
    update_mask = tl.load(update_indices+x,mask=mask)==True

    output = (x // N) + M
    
    # 将结果写回DRAM
    tl.store(input_ptr + offsets, output, mask=mask & update_mask)

def div_add_v2(input_tensor: torch.Tensor,update_indices:torch.Tensor, N: int, M: int) -> torch.Tensor:
    # 预分配输出


    BLOCK_SIZE=512
    grid = (triton.cdiv(input_tensor.shape[0], BLOCK_SIZE)), 
    
    # 启动内核
    div_add_v2_kernel[grid](
        input_tensor, update_indices,input_tensor.shape[0],N, M, 
        BLOCK_SIZE=BLOCK_SIZE,
    )

def top_p(topk_ids:torch.Tensor,expert_count:torch.Tensor,top_p:float,brownout_config: BrownoutConfig):
     expert_count_sort, expert_count_idx = torch.sort(expert_count, dim=-1, descending=True)
     expert_count_sum = torch.cumsum(expert_count_sort, dim=-1)
     mask = expert_count_sum<=top_p*topk_ids.shape[0]
     expert_selected = expert_count_idx[mask]
     mask = torch.isin(topk_ids, expert_selected)
     div_add(topk_ids[mask],brownout_config.way,60)

def top_p_v2(topk_ids:torch.Tensor,brownout_config: BrownoutConfig,brownout_mask_map:BrownoutMaskMap):
    # pass
    mask = brownout_mask_map.get_mask_by_threshold(brownout_config.top_p,device=str(topk_ids.device))
    div_add_v2(topk_ids,mask,brownout_config.way,60)
def align_token_ids(topk_ids: torch.Tensor, num_experts: int, brownout_config: BrownoutConfig, brownout_mask_map:BrownoutMaskMap,infer_state:Qwen2MoEInferState,block_size: int):
    topk_ids = topk_ids.flatten()
    # topk_ids= topk_ids//4
    # expert_count = torch.bincount(topk_ids, minlength=num_experts)
    if not brownout_config.full_brownout_mode and brownout_config.top_p < 1:

        # top_p(topk_ids,expert_count,brownout_config.top_p,brownout_config)
        top_p_v2(topk_ids,brownout_config,brownout_mask_map)
        # threshold_indices = int(brownout_config.top_p*topk_ids.shape[0])
        # div_add(topk_ids[threshold_indices:],brownout_config.way,60)
       

    expert_count = torch.bincount(topk_ids, minlength=num_experts)
    topk_ids_off = torch.empty_like(topk_ids)
    num_blocks_per_expert = (expert_count - 1 + block_size) // block_size
    num_blocks_per_expert_offset = torch.cat(
        (
            torch.tensor([0], device=num_blocks_per_expert.device),
            num_blocks_per_expert.cumsum(0),
        ),
        dim=0,
    )

    calculate_token_offsets(topk_ids, topk_ids_off)

    max_num_m_blocks = torch.sum(num_blocks_per_expert, dim=0)
    max_num_tokens_padded = max_num_m_blocks * block_size

    # 这里可以优化
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(-1)
    # print("padding id is", topk_ids.numel())
    # print(f'valid_tokens_number is {topk_ids.shape[0]}, {sorted_ids.shape[0]},rate is {topk_ids.shape[0]/sorted_ids.shape[0]:.3}')
    _moe_align_block_size[(topk_ids.shape[0],)](
        topk_ids,
        sorted_ids,
        num_blocks_per_expert_offset,
        topk_ids_off,
        expert_ids,
        block_size,
    )

    return sorted_ids, expert_ids


def fused_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    up_proj_packaged: torch.Tensor,
    gate_proj_packaged: torch.Tensor,
    down_proj_packaged: torch.Tensor,
    infer_state: Qwen2MoEInferState,
    brownout_config: BrownoutConfig,
    brownout_mask_map:BrownoutMaskMap,
    topk: int = 4,
    num_experts: int = 60,
    dim: int = 2048,
    intermediate_size: int = 1408,
):

    GROUP_SIZE = 16
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_K = 512
    inp = hidden_states.repeat_interleave(topk, dim=0)
    if inp.shape[0] > 256:
        GROUP_SIZE = 32
    else:
        GROUP_SIZE = 16
    # topk_ids = topk_ids
    sorted_ids, expert_ids = align_token_ids(topk_ids, num_experts, brownout_config,brownout_mask_map, infer_state,GROUP_SIZE)
    # print('rate is',inp.shape[0]/sorted_ids.shape[0])
    # o = torch.empty(
    #     inp.shape[0], intermediate_size, device=inp.device, dtype=torch.float16
    # )
    up = torch.empty(
        inp.shape[0], intermediate_size, device=inp.device, dtype=torch.float16
    )


    __fused_matrix_multiple[
        (expert_ids.shape[0], (intermediate_size - 1 + BLOCK_SIZE_M) // BLOCK_SIZE_M)
    ](
        inp,
        up,
        sorted_ids,
        up_proj_packaged,
        expert_ids,
        dim,
        intermediate_size,
        -1,
        GROUP_SIZE,
        # BLOCK_SIZE_M,
        # BLOCK_SIZE_K,
        num_warps=4,
        num_stages=2,
    )

    gate = torch.empty(
        inp.shape[0], intermediate_size, device=inp.device, dtype=torch.float16
    )

    __fused_matrix_multiple[
        (expert_ids.shape[0], (intermediate_size - 1 + BLOCK_SIZE_M) // BLOCK_SIZE_M)
    ](
        inp,
        gate,
        sorted_ids,
        gate_proj_packaged,
        expert_ids,
        dim,
        intermediate_size,
        -1,
        GROUP_SIZE,
        # BLOCK_SIZE_M,
        # BLOCK_SIZE_K,
        num_warps=4,
        num_stages=2,
    )

   

    o = up * silu(gate)

    ffnout = torch.empty(inp.shape[0], dim, device=inp.device, dtype=torch.float16)

    __fused_matrix_multiple[
        (expert_ids.shape[0], (dim - 1 + BLOCK_SIZE_M) // BLOCK_SIZE_M)
    ](
        o,
        ffnout,
        sorted_ids,
        down_proj_packaged,
        expert_ids,
        intermediate_size,
        dim,
        -1,
        GROUP_SIZE,
        # BLOCK_SIZE_M,
        # BLOCK_SIZE_K,
        num_warps=4,
        num_stages=2,
    )
    return ffnout


if __name__ == "__main__":
    GROUP_SIZE = 32
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_K = 32
    topk_ids = torch.load(
        "/root/hujianmin/fused_moe_test/expert_indices_prefilling_32.pt"
    ).to("cuda:0")
    inp = torch.load("/root/hujianmin/fused_moe_test/inp_prefilling_32.pt").to("cuda:0")

    inp = inp.repeat_interleave(4, dim=0)
    num_experts = 60
    dim = 2048
    intermediate_size = 1408
    sorted_ids, expert_ids = align_token_ids(topk_ids, num_experts, GROUP_SIZE)

    print("sorted_ids", sorted_ids)
    print("expert_ids", expert_ids)

    w = Weight()

    up = torch.zeros(
        inp.shape[0], intermediate_size, device="cuda:0", dtype=torch.float16
    )

    __fused_matrix_multiple[
        (expert_ids.shape[0], (intermediate_size - 1 + BLOCK_SIZE_M) // BLOCK_SIZE_M)
    ](
        inp,
        up,
        sorted_ids,
        w.concat_up(),
        expert_ids,
        dim,
        intermediate_size,
        -1,
        GROUP_SIZE,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
    )

    gate = torch.zeros(
        inp.shape[0], intermediate_size, device="cuda:0", dtype=torch.float16
    )

    __fused_matrix_multiple[
        (expert_ids.shape[0], (intermediate_size - 1 + BLOCK_SIZE_M) // BLOCK_SIZE_M)
    ](
        inp,
        gate,
        sorted_ids,
        w.concat_gate(),
        expert_ids,
        dim,
        intermediate_size,
        -1,
        GROUP_SIZE,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
    )
    gate = silu(gate)
    o = up * gate

    ffnout = torch.zeros(inp.shape[0], dim, device="cuda:0", dtype=torch.float16)

    print(o, o.shape)
    __fused_matrix_multiple[
        (expert_ids.shape[0], (dim - 1 + BLOCK_SIZE_M) // BLOCK_SIZE_M)
    ](
        o,
        ffnout,
        sorted_ids,
        w.concat_down(),
        expert_ids,
        intermediate_size,
        dim,
        -1,
        GROUP_SIZE,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
    )

    print(ffnout, ffnout.shape)
