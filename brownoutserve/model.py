import itertools
import json
import random
import time
from collections import Counter
from typing import List
import torch
# import vllm_flash_attn
from brownoutserve.infer_state import Qwen2MoEInferState
from brownoutserve.gpu_manager_config import GPUManagerConfig
from brownoutserve.brownout_config import BrownoutConfig
from brownoutserve.weight import Weight
from brownoutserve.model_config import Qwen2MoEModelConfig
from brownoutserve.kernels.linear import linear
from brownoutserve.kernels.fused import fused_moe
from brownoutserve.kernels.rmsnorm import fused_add_rmsnorm_inplace, rmsnorm_inplace
from brownoutserve.kernels.rotary_emb import rotary_embedding_inplace
from brownoutserve.kernels.silu_and_mul import silu_and_mul, silu_and_mul_inplace
from brownoutserve.kernels.paged_attention import attention_decoding
from brownoutserve.kernels.fused_moe import swiglu_forward
from brownoutserve.kernels.prefill_attention import prefill_attention
from brownoutserve.gpu_manager import GPUManager
import torch.nn.functional as F
from brownoutserve.utils import generate_random_int_list
from brownoutserve.brownout_mask_map import BrownoutMaskMap
moe_time_usage=[]
class Expert:
    def __init__(
            self,
            # gate_proj: torch.Tensor,
            # up_proj: torch.Tensor,
            down_proj: torch.Tensor,
            up_gate_proj:torch.Tensor,
            model_config:Qwen2MoEModelConfig,
            layer_id:int,
            idx: int
    ):
        # self.gate_proj = gate_proj
        # self.up_proj = up_proj
        self.expert_id = idx
        self.layer_id = layer_id
        # self.device='cuda:3'
        self.down_proj = down_proj
        self.up_gate_proj = up_gate_proj
        self.model_config = model_config
        self.sliu=torch.nn.SiLU()
    
    def forward(self,o:torch.Tensor,infer_state:Qwen2MoEInferState):

        # origin_device = o.device
        # o=o.to(self.device)
        if o.shape[0]==0:
            return o
        
        up_gate_proj = linear(o, self.up_gate_proj)
        t = self.sliu(up_gate_proj[:,self.model_config.moe_intermediate_size:])*up_gate_proj[:,:self.model_config.moe_intermediate_size]
        ffn_out = torch.nn.functional.linear(t,self.down_proj)
        # ffn_out=ffn_out.to('cuda:0')
        return ffn_out


class TransformerLayer:
    @torch.inference_mode()
    def __init__(
        self,
        model_config: Qwen2MoEModelConfig,
        brownout_config: BrownoutConfig,
        brownout_mask_map: BrownoutMaskMap,
        weight: Weight,
        layers_device: List,
        layer_id: int,
        gpu_manager: GPUManager,
        decoding_piggyback_stream:torch.cuda.Stream,
        support_flash_attn: False,
        device="cuda:0",
    ):
        self.model_config = model_config
        self.brownout_config = brownout_config
        self.brownout_mask_map = brownout_mask_map
        self.layer_id = layer_id
        self.decoding_piggyback_stream = decoding_piggyback_stream
        self.device = device
        self.layers_device = layers_device
        self.gpu_manager = gpu_manager
        self.support_flash_attn = support_flash_attn
        self.input_layernorm = weight.input_layernorm[self.layer_id]
        self.mlp_gate = weight.mlp_gate[self.layer_id]
        self.mlp_shared_expert_down_proj = weight.mlp_shared_expert_down_proj[self.layer_id]
        self.mlp_shared_expert_gate = weight.mlp_shared_expert_gate[self.layer_id]
        self.mlp_shared_expert_up_gate_proj=weight.mlp_shared_expert_up_gate_proj[self.layer_id]
        self.post_attention_layernorm = weight.post_attention_layernorm[self.layer_id]
        self.k_proj_weight = weight.k_proj_weight[self.layer_id]
        self.k_proj_bias = weight.k_proj_bias[self.layer_id]
        self.o_proj = weight.o_proj[self.layer_id]
        self.q_proj_weight = weight.q_proj_weight[self.layer_id]
        self.q_proj_bias = weight.q_proj_bias[self.layer_id]
        self.v_proj_weight = weight.v_proj_weight[self.layer_id]
        self.v_proj_bias = weight.v_proj_bias[self.layer_id]

        if not self.brownout_config.use_fused_moe:
        # if True:
            self.mlp_experts_down_proj = weight.mlp_experts_down_proj[self.layer_id]
            self.mlp_experts_up_gate_proj = weight.mlp_experts_up_gate_proj[self.layer_id]
            self.experts = [Expert(self.mlp_experts_down_proj[idx],self.mlp_experts_up_gate_proj[idx],model_config,layer_id,idx) for idx in range(model_config.num_experts)]
            self.num_united_experts = (self.model_config.num_experts+self.brownout_config.way-1)//self.brownout_config.way
            if self.brownout_config.top_p<1.1 and not self.brownout_config.full_brownout_mode:
                self.united_experts = [Expert(weight.united_experts_weights[f'{layer_id}_{idx}_down_proj.weight'],weight.united_experts_weights[f'{layer_id}_{idx}_up_gate_proj.weight'],model_config,layer_id,idx) for idx in range(self.num_united_experts)]
        self.experts_inps = [None for _ in range(self.model_config.num_experts)]
        if brownout_config.use_fused_moe:
        # if True:
            self.mlp_experts_up_proj_packaged = weight.up_proj_packaged_list[self.layer_id]  
            self.mlp_experts_gate_proj_packaged = weight.gate_proj_packaged_list[self.layer_id]
            self.mlp_experts_down_proj_packaged = weight.down_proj_packaged_list[self.layer_id]
        self.way = self.brownout_config.way

    @torch.inference_mode()
    def forward(
        self,
        input_embds: torch.Tensor,  # [num_tokens, hidden_size]
        infer_state: Qwen2MoEInferState,
    ):
        if self.layer_id ==0 or self.layer_id > 0 and self.layers_device[self.layer_id - 1] != self.layers_device[self.layer_id]:
            cur_device = self.layers_device[self.layer_id]
            torch.cuda.set_device(cur_device)
            input_embds = input_embds.to(cur_device)
            infer_state.position_cos = infer_state.position_cos.to(cur_device)
            infer_state.position_sin = infer_state.position_sin.to(cur_device)
            infer_state.prefill_seq_start_locs_with_end = (
                infer_state.prefill_seq_start_locs_with_end.to(cur_device)
            )
            infer_state.prefill_seq_start_locs = infer_state.prefill_seq_start_locs.to(
                cur_device
            )
            infer_state.prefill_seq_lens = infer_state.prefill_seq_lens.to(cur_device)
            infer_state.decoding_seq_ids = infer_state.decoding_seq_ids.to(cur_device)
            infer_state.decoding_seq_lens = infer_state.decoding_seq_lens.to(cur_device)
            infer_state.residual_buf = infer_state.residual_buf.to(cur_device)
            # if self.brownout_config.use_fused_moe:
            #     infer_state.up = infer_state.up.to(cur_device)
            #     infer_state.gate= infer_state.gate.to(cur_device)
            #     infer_state.ffnout = infer_state.ffnout.to(cur_device)
            if (
                self.layer_id > 0
                and infer_state.use_kv_cache
                and infer_state.num_prefill_tokens > 0
            ):
                infer_state.prefill_seq_ids = infer_state.prefill_seq_ids.to(cur_device)
                infer_state.free_physical_blocks_indices = (
                    infer_state.free_physical_blocks_indices.to(cur_device)
                )
                infer_state.num_block_per_seq = infer_state.num_block_per_seq.to(
                    cur_device
                )
                infer_state.cum_block_num = infer_state.cum_block_num.to(cur_device)
                infer_state.cum_prefill_seqs_len = infer_state.cum_prefill_seqs_len.to(
                    cur_device
                )

            if self.layer_id > 0 and infer_state.num_decoding_seqs > 0:
                infer_state.decoding_seq_ids = infer_state.decoding_seq_ids.to(
                    cur_device
                )
                infer_state.physical_block_ids = infer_state.physical_block_ids.to(
                    cur_device
                )
                infer_state.decoding_seq_lens_before = (
                    infer_state.decoding_seq_lens_before.to(cur_device)
                )
        # print(f"layer_id:{self.layer_id}")
        # if self.layer_id>infer_state.layer_id and infer_state.mode=='train':
        #     return input_embds
        # torch.cuda.synchronize()
        # t0 = time.perf_counter()
        fused_add_rmsnorm_inplace(
            input_embds,
            infer_state.residual_buf,
            self.input_layernorm,
            self.model_config.rms_norm_eps,
        )

        # Calculate QKV
        q = linear(input_embds, self.q_proj_weight,self.q_proj_bias)  # [num_total_tokens, hidden_size]
        k = linear(
            input_embds, self.k_proj_weight,self.k_proj_bias
        )  # [num_total_tokens, num_kv_heads*head_dim]
        v = linear(
            input_embds, self.v_proj_weight,self.v_proj_bias
        )  # [num_total_tokens, num_kv_heads*head_dim]

        q = q.view(
            -1, self.model_config.n_heads, self.model_config.head_dim
        )  # [num_total_tokens, num_q_heads, head_dim]
        k = k.view(
            -1, self.model_config.n_kv_heads, self.model_config.head_dim
        )  # [num_total_tokens, num_kv_heads, head_dim]
        v = v.view(
            -1, self.model_config.n_kv_heads, self.model_config.head_dim
        )  # [num_total_tokens, num_kv_heads, head_dim]

        rotary_embedding_inplace(q, k, infer_state)
        o = input_embds

##########################
        # print('infer_state.num_decoding_seqs',infer_state.num_decoding_seqs,infer_state.num_prefill_tokens)
        if infer_state.num_prefill_tokens > 0:

            if infer_state.use_kv_cache:
                self.gpu_manager.allocate_blocks_for_new_seqs(
                    k, v, infer_state, self.layer_id
                )

                if not self.support_flash_attn:
                    o[: infer_state.num_prefill_tokens, :] = (
                        vllm_flash_attn.flash_attn_varlen_func(
                            q[: infer_state.num_prefill_tokens, :, :],
                            k[: infer_state.num_prefill_tokens, :, :],
                            v[: infer_state.num_prefill_tokens, :, :],
                            infer_state.prefill_seq_start_locs_with_end,
                            infer_state.prefill_seq_start_locs_with_end,
                            infer_state.max_prefill_len,
                            infer_state.max_prefill_len,
                            softmax_scale=infer_state.softmax_scale,
                            causal=True,
                        ).reshape(-1, self.model_config.hidden_size)
                    )
                else:
                    prefill_attention(
                        q[: infer_state.num_prefill_tokens, :, :],
                        k[: infer_state.num_prefill_tokens, :, :],
                        v[: infer_state.num_prefill_tokens, :, :],
                        o[: infer_state.num_prefill_tokens, :],
                        self.model_config,
                        infer_state,
                    )

  
##########################
        
        if infer_state.num_decoding_seqs > 0:
                self.gpu_manager.allocate_blocks_for_decoding(
                    k, v, self.layer_id, infer_state
                )
                attention_decoding(
                    q[infer_state.num_prefill_tokens : , :, :],
                    o[infer_state.num_prefill_tokens :, :],
                    self.gpu_manager.physical_k_cache_blocks[self.layer_id],
                    self.gpu_manager.physical_v_cache_blocks[self.layer_id],
                    infer_state.decoding_seq_ids,
                    self.gpu_manager.block_table,
                    infer_state.decoding_seq_lens,
                    self.gpu_manager.block_size,
                    self.model_config.n_heads,
                    self.model_config.n_kv_heads,
                    self.model_config.head_dim,
                    self.gpu_manager.max_num_blocks_per_seq
                )
        o = linear(o, self.o_proj)

        # print(o.shape)
        # print('o.shape',o.shape)
        fused_add_rmsnorm_inplace(
            o,
            infer_state.residual_buf,
            self.post_attention_layernorm,
            self.model_config.rms_norm_eps,
        )
        q = None 
        k = None
        v = None
        # FFN
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # latency = t1-t0
        # if not infer_state.is_warm_up:
        #     if infer_state.num_prefill_seqs>0:
        #         self.brownout_config.trace['prefilling_attention']+=latency
        #     else:
        #         self.brownout_config.trace['decoding_attention']+=latency
        orig_shape = o.shape
        o = o.view(-1, o.shape[-1])
        scores = linear(o,self.mlp_gate)

        expert_weights = F.softmax(scores, dim=1, dtype=torch.float)
        expert_weights, expert_indices = torch.topk(expert_weights, self.model_config.num_experts_per_tok, dim=-1)

        if self.model_config.norm_topk_prob:
            expert_weights /= expert_weights.sum(dim=-1, keepdim=True)
        expert_weights = expert_weights.to(o.dtype)


        if infer_state.mode == 'super':
                 self.brownout_config.way = self.way
                 y = fused_moe(o,expert_indices,self.mlp_experts_up_proj_packaged,self.mlp_experts_gate_proj_packaged,self.mlp_experts_down_proj_packaged,infer_state,self.brownout_config,self.brownout_mask_map)

           
        # 这个python的代码

        else:
            flat_expert_indices = expert_indices.view(-1)

            inp =o.repeat_interleave(self.model_config.num_experts_per_tok, dim=0)
        
            y = torch.zeros_like(inp)

            flat_expert_indices_list = flat_expert_indices.tolist()

            # print(flat_expert_indices.shape,flat_expert_indices.device)
            expert_required_set = set(flat_expert_indices_list)

            # 常规代码
            access_cnt=0
            if infer_state.mode == 'normal':
                for i, expert in enumerate(self.experts):
                    if i in expert_required_set:
                        y[flat_expert_indices == i] = expert.forward(inp[flat_expert_indices == i] ,infer_state)
            
            # 推理代码
            if infer_state.mode == 'infer' :
                # pass
                num_total_token = len(flat_expert_indices_list)
                count = Counter(flat_expert_indices_list)
                if infer_state.brownout_config.greedy:
                    sorted_count = sorted(count.items(), key=lambda x: x[1], reverse=True)
                else:
                    sorted_count = list(count.items())
                cur_sum=0
                threshold=-1
                if infer_state.brownout_config.top_p==0:
                    threshold=0
                else:
                    for i in range(len(sorted_count)):
                        cur_sum+=sorted_count[i][1]
                        if cur_sum>=num_total_token*infer_state.brownout_config.top_p:
                            threshold=i+1
                            break
            
                original_experts_indices = [x[0] for x in sorted_count[:threshold]]
                
                if self.layer_id>=2 and self.layer_id<=21:
                    for i in original_experts_indices:
                        indices = flat_expert_indices == i
                        y[indices] = self.experts[i].forward(inp[indices] ,infer_state)
                        access_cnt+=1
                    
                    if not self.brownout_config.full_brownout_mode:
                        united_experts_indices = sorted(sorted_count[threshold:], key=lambda x: x[0], reverse=False)
                        united_experts_indices = [x[0] for x in united_experts_indices]
                        united_experts_indices_groups = [[] for _ in range(self.num_united_experts)]
                        for i in united_experts_indices:
                            united_experts_indices_groups[i//self.way].append(i)
                        for i,group in enumerate(united_experts_indices_groups):  
                            if len(group)==0:
                                continue
                            elif len(group)==1:
                                indices = flat_expert_indices == group[0]
                                y[indices] = self.experts[group[0]].forward(inp[indices] ,infer_state)
                                access_cnt+=1
                            else:
                                indices = torch.isin(flat_expert_indices, torch.tensor(group,device=flat_expert_indices.device))
                                y[indices] = self.united_experts[i].forward(inp[indices] ,infer_state)
                                access_cnt+=1
                        # print(len(united_experts_indices_groups))
                else:
                    for i, expert in enumerate(self.experts):
                        if i in expert_required_set:
                            y[flat_expert_indices == i] = expert.forward(inp[flat_expert_indices == i] ,infer_state)
                            # access_cnt+=1
            
            # self.brownout_config.debug_info[self.layer_id].append(access_cnt)
                
        y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)
        up_gate_proj = linear(o, self.mlp_shared_expert_up_gate_proj)
        silu_and_mul_inplace(up_gate_proj)
        shared_expert_output = linear(
            up_gate_proj[:, : self.model_config.shared_expert_intermediate_size],
            self.mlp_shared_expert_down_proj,
        )

        shared_expert_output = F.sigmoid(linear(o,self.mlp_shared_expert_gate)) * shared_expert_output
        o = (y+shared_expert_output).view(*orig_shape)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        infer_state.moe_time_usage+=t2-t1
        return o    


class Qwen2MoE:
    @torch.inference_mode()
    def __init__(
        self,
        model_path: str,
        gpu_manager_config: GPUManagerConfig,
        brownout_config: BrownoutConfig,
        devices=["cuda:0"],
        dtype=torch.float16,
    ):
        self.model_path = model_path
        self.model_config = Qwen2MoEModelConfig.load_from_model_path(self.model_path)
        self.brownout_config=brownout_config
        self.layers_device = [None for _ in range(self.model_config.num_layers)]
        self.device = devices[0]
        total_devices = len(devices)
        num_devices_per_device = (self.model_config.num_layers + total_devices - 1) // total_devices
        for i in range(len(self.layers_device)):
                self.layers_device[i] = devices[i // num_devices_per_device]
        self.dtype = dtype
        self.gpu_manager_config = gpu_manager_config
        self.check_gpu_memory()
        self.weight = Weight(self.model_path, self.model_config,brownout_config,self.layers_device,self.dtype)
        # self.weight = Weight(model_path, self.model_config,device=device ,dtype=dtype)
        self.gpu_manager = None
        self.transformer_layers = None
        self.brownout_mask_map = BrownoutMaskMap(devices,self.model_config.num_experts)
        self.create_gpu_manager_and_transformer_layers()
        torch.cuda.synchronize()


    @torch.inference_mode()
    def _forward(self, input_ids: torch.Tensor, infer_state: Qwen2MoEInferState):
        input_embds = torch.embedding(
            self.weight.embed_tokens, input_ids, padding_idx=-1
        )

        infer_state.residual_buf = torch.zeros_like(input_embds)
        torch.cuda.set_device(self.layers_device[0])

        for layer in self.transformer_layers:

            input_embds = layer.forward(input_embds, infer_state)

        input_embds += infer_state.residual_buf
        last_token_indices = torch.cat(
            (
                infer_state.prefill_seq_start_locs + infer_state.prefill_seq_lens - 1,
                torch.arange(infer_state.num_prefill_tokens, infer_state.num_tokens, device=input_embds.device, dtype=torch.int32)
            ), dim=0
        )
        last_input = torch.empty(
            (infer_state.batch_size, self.model_config.hidden_size),
            device=input_embds.device,
            dtype=input_embds.dtype,
        )
        last_input[:, :] = input_embds[last_token_indices, :]
        rmsnorm_inplace(last_input, self.weight.norm, self.model_config.rms_norm_eps)
        logits = linear(last_input, self.weight.lm_head)
        if not infer_state.is_warm_up and infer_state.num_decoding_seqs>0:
            moe_time_usage.append(infer_state.moe_time_usage)
            # print('moe_time_usage:',infer_state.moe_time_usage)
        return logits

    @torch.inference_mode()
    def forward(
        self,
        input_ids_list: list[list[int]],  # total inputs
        num_decoding_seqs: int,  # the number of decoding sequences
        decoding_seq_ids: torch.Tensor,  # [num_decoding_seqs]
        use_kv_cache: bool = True,
        is_warm_up: bool = False,
        mode: str = 'normal',
        layer_id:int  = -1,
        brownout_config: BrownoutConfig=None

    ):
        # print('aa',len(input_ids_list),num_decoding_seqs)
        num_prefill_seqs = len(input_ids_list) - num_decoding_seqs
        flattened_input_ids = list(itertools.chain(*input_ids_list))
        # print(flattened_input_ids)
        # input_ids = torch.tensor(flattened_input_ids,dtype = torch.int32,device = self.device)
        prefill_seqs_len_list = [
            len(input_ids) for input_ids in input_ids_list[:num_prefill_seqs]
        ]
        prefill_seqs_len = torch.tensor(
            prefill_seqs_len_list, dtype=torch.int32, device=self.device
        )  # [batch_size,]
        cum_prefill_seqs_len = torch.cumsum(
            prefill_seqs_len, dim=0, dtype=torch.int32,
        )  # [batch_size,]
        prefill_start_locs = (
            torch.cumsum(prefill_seqs_len, dim=0, dtype=torch.int32) - prefill_seqs_len
        )  # [batch_size,]
        num_block_per_seq = (
            prefill_seqs_len + self.gpu_manager_config.block_size - 1
        ) // self.gpu_manager_config.block_size
        cum_block_num = torch.cumsum(num_block_per_seq, dim=0)  # [num_seq]
        max_prefill_len = max(prefill_seqs_len_list) if prefill_seqs_len_list else 0
        # print(max_prefill_len)
        # exit(0)
        batch_size = len(input_ids_list)
        num_tokens = len(flattened_input_ids)

        decoding_seq_lens_before = self.gpu_manager.num_tokens_allocated_per_seq[decoding_seq_ids.to(self.device)]
        decoding_seq_lens=decoding_seq_lens_before + 1

        position_indices = torch.cat((
            torch.concat([
                torch.arange(0, prefill_seq_len, device=self.device, dtype=torch.int32)
                for prefill_seq_len in prefill_seqs_len
            ]) if prefill_seqs_len_list else torch.empty(0, device=self.device, dtype=torch.int32),
            decoding_seq_lens_before
        ),dim=0)

        infer_state = Qwen2MoEInferState(
            batch_size=batch_size,
            num_tokens=num_tokens,
            prefill_seq_ids=None,
            physical_block_ids=torch.empty_like(decoding_seq_ids).to(self.device),
            softmax_scale=self.model_config.head_dim**-0.5,
            num_prefill_seqs=batch_size - num_decoding_seqs,
            num_prefill_tokens=num_tokens-num_decoding_seqs,
            cum_prefill_seqs_len=cum_prefill_seqs_len,
            prefill_seq_start_locs=prefill_start_locs,
            prefill_seq_start_locs_with_end=torch.cat(
                [
                    prefill_start_locs,
                    torch.tensor([num_tokens], dtype=torch.int32, device=self.device),
                ]
            ),
            prefill_seq_lens=prefill_seqs_len,
            max_prefill_len=max_prefill_len,
            num_block_per_seq=num_block_per_seq,
            cum_block_num=cum_block_num,
            num_decoding_seqs=num_decoding_seqs,
            decoding_seq_ids=decoding_seq_ids,
            decoding_seq_lens=decoding_seq_lens,
            decoding_seq_lens_before=decoding_seq_lens_before,
            free_physical_blocks_indices = None, #for prefilling
            position_cos=self.weight.cos[position_indices],
            position_sin=self.weight.sin[position_indices],
            use_kv_cache = use_kv_cache,
            is_warm_up=is_warm_up,
            mode= mode,
            layer_id=layer_id,
            brownout_config=brownout_config
    
        )
        # brownout_config.trace['decoding_seq_ids'] = decoding_seq_ids.clone()
        # if brownout_config.use_fused_moe:
        #     infer_state.up= torch.empty(num_tokens*self.model_config.num_experts_per_tok, self.model_config.intermediate_size, device=self.device, dtype=torch.float16)
        #     infer_state.gate= torch.empty(num_tokens*self.model_config.num_experts_per_tok, self.model_config.intermediate_size, device=self.device, dtype=torch.float16)
        #     infer_state.ffnout= torch.empty(num_tokens*self.model_config.num_experts_per_tok, self.model_config.hidden_size, device=self.device, dtype=torch.float16)
        return (
            self._forward(
                torch.tensor(flattened_input_ids, dtype=torch.int32, device=self.device),
                infer_state,
            ),
            infer_state.prefill_seq_ids,
        )
    
    def check_gpu_memory(self):
        MB = 1024**2
        GB = 1024**3
        free_memory, total_memory = torch.cuda.mem_get_info(self.device)
        free_memory = free_memory
        total_memory = total_memory
        required_free_memory = total_memory*self.gpu_manager_config.gpu_mem_utilization
        if free_memory< required_free_memory:
            error_message = (
            f"Insufficient GPU memory: "
            f"Free memory ({free_memory/MB:.2f} MB) is less than the required memory "
            f"({required_free_memory/MB:.2f} MB) based on the configured utilization threshold "
            f"({self.gpu_manager_config.gpu_mem_utilization * 100:.2f}%). "
            f"Total memory: {total_memory/GB:.2f} GB."
        )
            raise RuntimeError(error_message)
        print(f'Free memory is {free_memory/GB:.2f} GB, total memory is {total_memory/GB:.2f}')
        self.total_free_memory = free_memory
    
    def auto_configure_gpu_manager(self):
        KB = 1024
        MB = 1024**2
        GB = 1024**3
     
        # torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        free_memory_before,_ = torch.cuda.mem_get_info(self.device)
        max_memory_allocated = torch.cuda.max_memory_allocated(self.device) 
        print(f'After Loading Model free memory: {free_memory_before/GB:.2f} GB')
        print(f'Model weight memory usage: {max_memory_allocated/GB:.2f} GB')

        # max_seq_len = self.gpu_manager_config.max_seq_len
        input_ids = [[0  for _ in range(self.gpu_manager_config.max_seq_len)] for _ in range(self.gpu_manager_config.max_batch_size)]
        self.forward(
                input_ids_list = input_ids,
                num_decoding_seqs=0,
                decoding_seq_ids = torch.empty(0,device=self.device,dtype=torch.int32),
                use_kv_cache=False,
                mode='super' if self.brownout_config.use_fused_moe else 'normal',
                is_warm_up=True,
                brownout_config=self.brownout_config
        )
        torch.cuda.synchronize()
        free_memory_after,_ = torch.cuda.mem_get_info(self.device)
        print(f"Maximum memory allocated during forward pass: {(free_memory_before-free_memory_after)/MB:.2f} MB")
        free_memory,_ = torch.cuda.mem_get_info(self.device)
        print(f"Free memory available on {self.device} for kv cache: {free_memory/MB:.2f} MB")
        memory_per_block = self.gpu_manager.get_block_memory()
        print(f'Memory_used_per_block: {memory_per_block/KB:.2f} KB' )
        max_num_blocks = (free_memory // memory_per_block)//(2*self.model_config.num_layers)
        print(f"The maximum number of blocks in the KV cache: {max_num_blocks}")
        if max_num_blocks < self.gpu_manager.max_blocks_in_all_seq:
            print(f"WARNING:Actual max block count ({max_num_blocks}) is less than the configured limit ({self.gpu_manager.max_blocks_in_all_seq})")
    
        self.gpu_manager.init_physical_kv_cache(max_num_blocks)

    @torch.inference_mode()
    def warm_up(self):
        epoch = min(10,self.gpu_manager.max_seq_in_table)
        input_len = min(10,self.gpu_manager.max_seq_len)
        # input_ids = [[0 for _ in range(input_len)] for _ in range(epoch)]
        input_ids = generate_random_int_list(epoch,input_len,0,self.model_config.vocab_size)
        logits,seq_ids = self.forward(
                input_ids_list = input_ids,
                num_decoding_seqs=0,
                decoding_seq_ids = torch.empty(0,device=self.device,dtype=torch.int32),
                mode='super' if self.brownout_config.use_fused_moe else 'normal',
                is_warm_up=True,
                brownout_config=self.brownout_config
        )
        self.gpu_manager.reset()
        original_top_p = self.brownout_config.top_p
        self.brownout_config.top_p = 0
        logits,seq_ids = self.forward(
                input_ids_list = input_ids,
                num_decoding_seqs=0,
                decoding_seq_ids = torch.empty(0,device=self.device,dtype=torch.int32),
                mode='super' if self.brownout_config.use_fused_moe else 'normal',
                is_warm_up=True,
                brownout_config=self.brownout_config
        )
        prompt_phase_outputs = torch.argmax(logits,dim=1)
        self.brownout_config.top_p = original_top_p
        _ = self.forward(
        input_ids_list = [[x] for x in prompt_phase_outputs],
        num_decoding_seqs = len(prompt_phase_outputs),
        decoding_seq_ids =seq_ids,
        mode='super' if self.brownout_config.use_fused_moe else 'normal',
        is_warm_up=True,
        brownout_config=self.brownout_config
        )
        self.gpu_manager.reset()

    def create_gpu_manager_and_transformer_layers(self):
        self.gpu_manager = GPUManager(
            max_batch_size=self.gpu_manager_config.max_batch_size,
            max_seq_in_table=self.gpu_manager_config.max_seqs_in_table,
            max_seq_len=self.gpu_manager_config.max_seq_len,
            n_heads=self.model_config.n_heads,
            n_kv_heads=self.model_config.n_kv_heads,
            n_layers=self.model_config.num_layers,
            block_size=self.gpu_manager_config.block_size,
            head_dim=self.model_config.head_dim,
            layers_device=self.layers_device,
            device=self.device
        )
        support_flash_attn = self.check_support_flash_attn()
        if support_flash_attn:
            print("Device supports FlashAttention!")
        else:
            print("Device doesn't support FlashAttention!")

        decoding_piggyback_stream = torch.cuda.Stream()
        self.transformer_layers = [
            TransformerLayer(self.model_config,self.brownout_config,self.brownout_mask_map, self.weight, self.layers_device,layer_id, self.gpu_manager,decoding_piggyback_stream,support_flash_attn=support_flash_attn)
            for layer_id in range(self.model_config.num_layers)
        ]
        self.auto_configure_gpu_manager()
        # self.gpu_manager.init_physical_kv_cache()
        print("GPU Manager init successfully!")

    def check_support_flash_attn(self) -> bool:
        gpu_name = torch.cuda.get_device_name(self.device)
        return "A100" in gpu_name or "3090" in gpu_name or "4090" in gpu_name
    
    def release_finised_seqs_resource(self,fininshed_seq_ids:torch.Tensor):
     
        # fininshed_seq_ids_list = fininshed_seq_ids.tolist()
        # print(f'seqs {fininshed_seq_ids_list} finished')
        fininshed_seq_ids = fininshed_seq_ids.to(self.gpu_manager.device)
        self.gpu_manager.free_physical_blocks(fininshed_seq_ids)