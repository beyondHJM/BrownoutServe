import json
import os
from pathlib import Path
from typing import Dict, List
from safetensors import safe_open
import torch
from tqdm import tqdm
import re
from brownoutserve.model_config import Qwen2MoEModelConfig
from brownoutserve.brownout_config import BrownoutConfig
import gc


class Weight():
    def __init__(self,model_path: str,config:  Qwen2MoEModelConfig, brownout_config:BrownoutConfig, layers_device: List ,dtype: torch.dtype = torch.float16):
        self.layers_device=layers_device
        self.brownout_config = brownout_config
        self.device=self.layers_device[0]
        self.config=config
        self.raw_weight = self.load_safetensor(model_path,dtype,device_map_path=None)
        self.cos,self.sin = self.init_to_get_rotary(config,device=self.device)
        self.embed_tokens = self.raw_weight['embed_tokens.weight']
        self.input_layernorm=[]
        self.mlp_down_proj=[]
        self.mlp_gate=[]

        self.mlp_shared_expert_down_proj=[]
        self.mlp_shared_expert_gate=[]

        self.post_attention_layernorm=[]
        self.k_proj_weight=[]
        self.k_proj_bias=[]
        self.q_proj_weight=[]
        self.q_proj_bias=[]
        self.v_proj_weight=[]
        self.v_proj_bias=[]
        self.o_proj=[]

        self.mlp_experts_down_proj = []
        self.mlp_experts_up_gate_proj=[]
        self.mlp_shared_expert_up_gate_proj=[]
        self.up_proj_packaged_list=[]
        self.gate_proj_packaged_list=[]
        self.down_proj_packaged_list=[]
        for layer_id in range(config.num_layers):

            down_projs=[]
            up_gate_projs=[]
            if brownout_config.use_fused_moe:
            # if True:
                self.up_proj_packaged_list.append(torch.concat([self.raw_weight[f'layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight'] for expert_id in range(config.num_experts)],dim=0))
                self.gate_proj_packaged_list.append(torch.concat([self.raw_weight[f'layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight'] for expert_id in range(config.num_experts)],dim=0))
                self.down_proj_packaged_list.append(torch.concat([self.raw_weight[f'layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight'] for expert_id in range(config.num_experts)],dim=0))
            for expert_id in range(config.num_experts):
                # up_projs.append(self.raw_weight[f'layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight'])
                # gate_projs.append(self.raw_weight[f'layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight']) 
                if not brownout_config.use_fused_moe:
                # if True:
                    down_projs.append(self.raw_weight[f'layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight'])
                    up_gate_projs.append(torch.cat((self.raw_weight[f'layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight'],self.raw_weight[f'layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight']), dim=0).contiguous())
                else:
                    del self.raw_weight[f'layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight']
                del self.raw_weight[f'layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight']
                del self.raw_weight[f'layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight']
            if not brownout_config.use_fused_moe:
            # if True:
                self.mlp_experts_down_proj.append(down_projs)
                self.mlp_experts_up_gate_proj.append(up_gate_projs)
            self.input_layernorm.append(self.raw_weight[f'layers.{layer_id}.input_layernorm.weight'])
            self.mlp_gate.append(self.raw_weight[f'layers.{layer_id}.mlp.gate.weight'])

            self.mlp_shared_expert_down_proj.append(self.raw_weight[f'layers.{layer_id}.mlp.shared_expert.down_proj.weight'])
            self.mlp_shared_expert_up_gate_proj.append(torch.cat((self.raw_weight[f'layers.{layer_id}.mlp.shared_expert.up_proj.weight'], self.raw_weight[f'layers.{layer_id}.mlp.shared_expert.gate_proj.weight']), dim=0).contiguous())
            self.raw_weight[f'layers.{layer_id}.mlp.shared_expert.up_proj.weight'] = None
            self.raw_weight[f'layers.{layer_id}.mlp.shared_expert.gate_proj.weight'] = None
            self.mlp_shared_expert_gate.append(self.raw_weight[f'layers.{layer_id}.mlp.shared_expert_gate.weight'])
            # self.mlp_up_gate_proj.append(torch.cat((self.mlp_up_proj[layer_id], self.mlp_gate_proj[layer_id]), dim=0).contiguous())
            self.post_attention_layernorm.append(self.raw_weight[f'layers.{layer_id}.post_attention_layernorm.weight'])
            self.k_proj_weight.append(self.raw_weight[f'layers.{layer_id}.self_attn.k_proj.weight'])
            self.k_proj_bias.append(self.raw_weight[f'layers.{layer_id}.self_attn.k_proj.bias'])
            self.o_proj.append(self.raw_weight[f'layers.{layer_id}.self_attn.o_proj.weight'])
            self.q_proj_weight.append(self.raw_weight[f'layers.{layer_id}.self_attn.q_proj.weight'])
            self.q_proj_bias.append(self.raw_weight[f'layers.{layer_id}.self_attn.q_proj.bias'])
            self.v_proj_weight.append(self.raw_weight[f'layers.{layer_id}.self_attn.v_proj.weight'])
            self.v_proj_bias.append(self.raw_weight[f'layers.{layer_id}.self_attn.v_proj.bias'])
        
        self.lm_head = self.raw_weight['lm_head.weight']
        self.norm = self.raw_weight['norm.weight']
        if not self.brownout_config.full_brownout_mode:
            self.load_united_experts_weights()
        gc.collect()
        # print(self.raw_weight[f'layers.0.mlp.experts.0.up_proj.weight'])
    def load_safetensor(self,ckpt_dir: str,dtype = torch.float16,device_map_path: str = None) -> Dict:
    
        checkpoints = sorted(Path(ckpt_dir).glob("*.safetensors"))
        if device_map_path is not None:
            with open(device_map_path,'r') as f:
                device_map = json.load(f)
 
        state_dict={}
        pbar = tqdm(total=len(checkpoints), desc="Loading model weight")
        for chpt in checkpoints:
            with safe_open(chpt, framework="pt", device=self.device) as f:
                for k in f.keys():
                    new_k = k.replace("model.", "")
                    state_dict[new_k] = f.get_tensor(k).to(dtype)
                    if device_map_path is not None:
                        state_dict[new_k] = state_dict[new_k].to(device_map[k])
                    else:
                        if new_k == "embed_tokens.weight":
                            device = self.layers_device[0]
                        elif new_k == "lm_head.weight" or new_k == "norm.weight":
                            device = self.layers_device[-1]
                        else:
                            match = re.search(r"\.([0-9]+)\.", new_k)
                            device = self.layers_device[int(match.group(1))]
                        state_dict[new_k] = state_dict[new_k].to(device)
            pbar.update(1) 
        pbar.close()
        print("successfully load ")
        # print(state_dict['layers.1.input_layernorm.weight'])
        return state_dict
 
    
    
    def init_to_get_rotary(self,config:  Qwen2MoEModelConfig,device='cuda',dtype :torch.dtype = torch.float16):
        rope_scaling_factor = config.rope_scaling
        base = config.rope_theta
        max_position_embeddings = config.max_position_embeddings
        max_seq_len = max_position_embeddings * rope_scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, config.head_dim, 2, device=device, dtype=torch.float32) / config.head_dim))
        t = torch.arange(max_seq_len + 128, device=device, dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)
        return cos,sin
    

    def load_united_experts_weights(self):

        weight_dict={}
        pbar = tqdm(total=3, desc="Loading united experts weight")
        for i in range(1,4):
                result = torch.load(f'{self.brownout_config.united_experts_weight_dirctory}/w{i}.pth',map_location=torch.device('cpu'))
                weight_dict.update(result)
                pbar.update(1) 
        pbar.close
        for layer_id in range(24):
            num_united_experts = (self.config.num_experts+self.brownout_config.way-1) // self.brownout_config.way
            if self.brownout_config.use_fused_moe:
                all_proj_packaged_list = [self.up_proj_packaged_list[layer_id]]
                all_proj_packaged_list.extend([weight_dict[f'{layer_id}_{expert_id}_up_proj.weight'].to(self.layers_device[layer_id]) for expert_id in range(num_united_experts)])
                self.up_proj_packaged_list[layer_id] = torch.cat(all_proj_packaged_list,dim=0)


                all_proj_packaged_list = [self.gate_proj_packaged_list[layer_id]]
                all_proj_packaged_list.extend([weight_dict[f'{layer_id}_{expert_id}_gate_proj.weight'].to(self.layers_device[layer_id]) for expert_id in range(num_united_experts)])
                self.gate_proj_packaged_list[layer_id] = torch.cat(all_proj_packaged_list,dim=0)
    
                all_proj_packaged_list = [self.down_proj_packaged_list[layer_id]]
                all_proj_packaged_list.extend([weight_dict[f'{layer_id}_{expert_id}_down_proj.weight'].to(self.layers_device[layer_id]) for expert_id in range(num_united_experts)])
                self.down_proj_packaged_list[layer_id] = torch.cat(all_proj_packaged_list,dim=0)
                del all_proj_packaged_list[0]

            for expert_id in range(num_united_experts):
                if not self.brownout_config.use_fused_moe:
                    weight_dict[f'{layer_id}_{expert_id}_up_gate_proj.weight']=torch.cat((weight_dict[f'{layer_id}_{expert_id}_up_proj.weight'],weight_dict[f'{layer_id}_{expert_id}_gate_proj.weight']),dim=0).contiguous().to(self.layers_device[layer_id])
                    weight_dict[f'{layer_id}_{expert_id}_down_proj.weight'] = weight_dict[f'{layer_id}_{expert_id}_down_proj.weight'].to(self.layers_device[layer_id])
                else:
                    del weight_dict[f'{layer_id}_{expert_id}_down_proj.weight']
                del weight_dict[f'{layer_id}_{expert_id}_up_proj.weight']
                del weight_dict[f'{layer_id}_{expert_id}_gate_proj.weight']
        # gc.collect()
        self.united_experts_weights=weight_dict
        print("united experts weights load successfully!")
    
    def update_united_experts_weights(self,way:int):
        pass
    


if __name__ == '__main__':
    model_path = '/root/llm-resource/Models/Meta-Llama-3-8B-Instruct'
    llama_model_config =  Qwen2MoEModelConfig.load_from_model_path(model_path)
    w = Weight(model_path,llama_model_config,torch.float16)
    print(w.input_layernorm[0])