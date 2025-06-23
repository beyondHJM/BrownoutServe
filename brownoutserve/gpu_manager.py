import threading
import json
import time
from typing import List
import torch

from brownoutserve.kernels.block_manager import allocate_blocks_for_prefilling,allocate_blocks_for_decoding,free_physical_blocks

from brownoutserve.infer_state import Qwen2MoEInferState


class GPUManager:

    @torch.inference_mode()
    def __init__(
        self,
        max_batch_size: int,
        max_seq_in_table: int,
        max_seq_len: int,
        n_heads: int,
        n_kv_heads: int,
        n_layers: int,
        block_size: int,
        head_dim: int,
        layers_device: List,
        device="cuda:0",
        dtype=torch.float16,
    ):
        assert (
            n_heads % n_kv_heads == 0
        ), f"Error: 'n_heads' ({n_heads}) must be an integer multiple of 'n_kv_heads' ({n_kv_heads}). "

        assert (
            max_seq_in_table >= max_batch_size
        ), f"Error: max_seq_in_table ({max_seq_in_table}) must be greater than or equal to max_batch_size ({max_batch_size})."

    
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.block_size = block_size
        self.max_seq_in_table = max_seq_in_table
        self.head_dim = head_dim
        self.device = device
        self.layers_device = layers_device
        self.devices_list = list(set(self.layers_device))
        self.dtype = dtype
        self.max_num_blocks_per_seq = (
            self.max_seq_len + self.block_size - 1
        ) // self.block_size

        self.max_blocks_in_all_seq = self.max_num_blocks_per_seq * max_batch_size
        # self.block_tables = {}
        # for device in self.devices_list:
        #     if device != self.device:
        #         self.block_tables[device] = None

        self.block_table = torch.zeros(
            max_batch_size,
            self.max_num_blocks_per_seq,
            dtype=torch.int32,
            device=self.device,
        )
        # block_tables[self.device] = self.block_table
        self.allocated_logical_blocks_indices = torch.zeros(
            max_seq_in_table, device=self.device, dtype=torch.bool
        )

        self.num_tokens_allocated_per_seq = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=self.device
        )

        self.physical_k_cache_blocks = None
        self.physical_v_cache_blocks = None

        self.allocated_physical_kv_cache_blocks_indices = torch.zeros(
            self.max_blocks_in_all_seq, dtype=torch.bool, device=self.device
        )

    @torch.inference_mode()
    def allocate_blocks_for_new_seqs(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        infer_state: Qwen2MoEInferState,
        layer_id: torch.Tensor,
    ):
        # print(f"prefill layer_id = {layer_id}")
        cur_device = self.layers_device[layer_id]
        seqs_len = infer_state.prefill_seq_lens
        assert (
            self.physical_k_cache_blocks is not None
        ), "KV cache not initialized. Please initialize it first."
        assert len(keys.shape) == 3, "keys shape must be (num_tokens,n_kv_heads)."
        assert len(seqs_len.shape) == 1, "seqs_len shape must be (nums_seq,)"
        max_seq_len = torch.max(seqs_len)
        assert (
            max_seq_len <= self.max_seq_len
        ), f"max_seq_len ({max_seq_len}) is greater than the allowed maximum of {self.max_seq_len}"
        bsz = seqs_len.shape[0]
        # with self.lock:
        if layer_id == 0:
            free_slots_indices = self.find_k_free_logical_slots(bsz)  # [batch_size,]
            infer_state.prefill_seq_ids = free_slots_indices
            free_physical_blocks_indices = self.find_k_free_physical_blocks(
                infer_state.cum_block_num[-1]
            )
            infer_state.free_physical_blocks_indices = free_physical_blocks_indices
        else:
            free_slots_indices = infer_state.prefill_seq_ids
            free_physical_blocks_indices = infer_state.free_physical_blocks_indices

        self.allocated_logical_blocks_indices[free_slots_indices] = 1
        # if layer_id == 0:
        #     cur_block_table = self.block_table
        # else:
        #     if cur_device == self.device:
        #         cur_block_table = self.block_table
        #     else:
        #         cur_block_table = self.block_tables[cur_device]
        

        allocate_blocks_for_prefilling(
            free_slots_indices,
            self.block_table,
            infer_state.num_block_per_seq,
            infer_state.cum_block_num,
            free_physical_blocks_indices,
            keys,
            values,
            self.physical_k_cache_blocks[layer_id],
            self.physical_v_cache_blocks[layer_id],
            seqs_len,
            infer_state.cum_prefill_seqs_len,
            self.block_size,
            self.max_num_blocks_per_seq,
            self.n_kv_heads * self.head_dim,
        )

        # with self.lock:
        self.allocated_physical_kv_cache_blocks_indices[
            free_physical_blocks_indices
        ] = True
        # if layer_id == 0:
        #     for device in self.block_tables:
        #         self.block_tables[device] = self.block_table.to(device)

        if layer_id == self.n_layers - 1:
            # print(self.num_tokens_allocated_per_seq.device,free_slots_indices.device,seqs_len.device)
            self.num_tokens_allocated_per_seq[free_slots_indices.to(self.device)] = (
                seqs_len.to(self.device)
            )

        # Return the indices of the sequences in the block table to the user
        # for clearing the sequences' kv cache when finishing generation.
        return free_slots_indices

    @torch.inference_mode()
    def find_k_free_logical_slots(
        self, k: int, to_allocate: bool = True
    ) -> torch.Tensor:
        # with self.lock:
        free_blocks_indices = torch.nonzero(
            self.allocated_logical_blocks_indices == 0
        ).view(-1)
        if free_blocks_indices.shape[0] < k:
            raise RuntimeError(
                f"Block table only has {free_blocks_indices.shape[0]} free sequence slots now,but {k} slots are required."
            )
        free_blocks_indices = free_blocks_indices[:k]
        if to_allocate:
            self.allocated_logical_blocks_indices[free_blocks_indices] = True
        return free_blocks_indices

    @torch.inference_mode()
    def find_k_free_physical_blocks(
        self, k: int, to_allocate: bool = True
    ) -> torch.Tensor:
        # with self.lock:

        free_blocks_indices = torch.nonzero(
            self.allocated_physical_kv_cache_blocks_indices == 0
        ).view(-1)

        if free_blocks_indices.shape[0] < k:
            raise RuntimeError(
                f"Physical blocks only has {free_blocks_indices.shape[0]} free blocks now,but {k} blocks are required."
            )

        free_blocks_indices = free_blocks_indices[:k]
        if to_allocate:
            self.allocated_physical_kv_cache_blocks_indices[free_blocks_indices] = True
        return free_blocks_indices

    @torch.inference_mode()
    def allocate_blocks_for_decoding(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        infer_state: Qwen2MoEInferState,
    ):

        cur_device = self.layers_device[layer_id]
        # with self.lock:
        if layer_id == 0:
            need_new_block_seq_ids_indices = infer_state.decoding_seq_lens_before % self.block_size == 0
            # print(infer_state.decoding_seq_ids.device,need_new_block_seq_ids_indices.device)
            need_new_block_seq_ids = infer_state.decoding_seq_ids[need_new_block_seq_ids_indices]  # [seqs_num,]
            new_physical_block_ids = self.find_k_free_physical_blocks(
                need_new_block_seq_ids.shape[0]
            )  # [seqs_need_block_newlen,]
            # print("new_physical_block_ids",new_physical_block_ids)
            # print("infer_state.physical_block_ids",infer_state.physical_block_ids.device)
            # print("need_new_block_seq_ids",need_new_block_seq_ids.device)
            infer_state.physical_block_ids[need_new_block_seq_ids_indices] = (
                new_physical_block_ids
            )
            infer_state.physical_block_ids = infer_state.physical_block_ids

            

        # if layer_id == 0:
        #     cur_block_table = self.block_table
        # else:
        #     if cur_device == self.device:
        #         cur_block_table = self.block_table
        #     else:
        #         cur_block_table = self.block_tables[cur_device]

        allocate_blocks_for_decoding(
            k,
            v,
            infer_state.decoding_seq_ids,
            # self.block_size,
            self.block_table,
            self.physical_k_cache_blocks[layer_id],
            self.physical_v_cache_blocks[layer_id],
            infer_state.physical_block_ids,
            infer_state.decoding_seq_lens_before,
            self.n_kv_heads * self.head_dim,
            self.block_size,
            self.max_num_blocks_per_seq,
        )

        if layer_id == self.n_layers - 1:
            self.num_tokens_allocated_per_seq[infer_state.decoding_seq_ids.to(self.device)] += 1
        # self.allocated_physical_kv_cache_blocks_indices[need_new_block_seq_ids]=1

    @property
    def num_free_physical_blocks(self):
        return (
            self.max_blocks_in_all_seq
            - self.allocated_physical_kv_cache_blocks_indices.sum().item()
        )

    def init_physical_kv_cache(self, max_num_blocks: int):

        self.physical_v_cache_blocks = [
            torch.empty(
                max_num_blocks,
                self.block_size,
                self.n_kv_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.layers_device[id],
            )
            for id in range(self.n_layers)
        ]

        self.physical_k_cache_blocks = [
            torch.empty(
                max_num_blocks,
                self.block_size,
                self.n_kv_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.layers_device[id],
            )
            for id in range(self.n_layers)
        ]

    @torch.inference_mode()
    def reset(self):
        self.allocated_logical_blocks_indices[:] = 0
        self.num_tokens_allocated_per_seq[:] = 0
        self.allocated_physical_kv_cache_blocks_indices[:] = 0

    def get_block_memory(self):
        temp_tesnor = torch.tensor([0], dtype=self.dtype)
        dtype_size = temp_tesnor.element_size()

        return self.block_size * self.n_kv_heads * self.head_dim * dtype_size

    def print_full_kv(self, layer_id: int):
        print("full k cache:")
        print(self.physical_k_cache_blocks[layer_id].shape)
        print("full v cache:")
        print(self.physical_v_cache_blocks[layer_id].shape)

    def print_seqs_logical_kv(self, seq_ids: torch.Tensor):
        print(f"{seq_ids.tolist()} logical_kv")
        print(self.block_table[seq_ids].tolist())


    @torch.inference_mode()
    def free_physical_blocks(self,seq_ids: torch.Tensor):
        num_blocks_per_seq  = torch.ceil(self.num_tokens_allocated_per_seq / self.block_size)
        self.allocated_logical_blocks_indices[seq_ids]=False
        free_physical_blocks(seq_ids,num_blocks_per_seq,self.allocated_physical_kv_cache_blocks_indices,self.block_table,self.max_num_blocks_per_seq)
        # print(f'total physical blocks:{self.max_blocks_in_all_seq},free physical:{self.num_free_physical_blocks}')