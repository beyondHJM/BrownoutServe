import json
import os

import torch


class Qwen2MoEModelConfig():
    def __init__(self, model_config: dict):
        assert  model_config['model_type'] == 'qwen2_moe'
        self.num_layers = model_config['num_hidden_layers']
        self.n_heads = model_config["num_attention_heads"]
        self.n_kv_heads = model_config.get("num_key_value_heads", self.n_heads)
        self.hidden_size = model_config["hidden_size"]
        self.head_dim = self.hidden_size // self.n_heads
        self.vocab_size = model_config["vocab_size"]
        self.max_position_embeddings = model_config["max_position_embeddings"]
        self.ffn_inter_dim = model_config["intermediate_size"]
        self.rotary_base = model_config.get("rope_theta", model_config.get("rotary_base", 10000))
        self.rms_norm_eps = model_config["rms_norm_eps"]
        self.rope_scaling = model_config.get("rope_scaling", 1.0)
        self.rope_theta = model_config.get("rope_theta", 10000)
        self.intermediate_size = model_config['intermediate_size']
        self.moe_intermediate_size = model_config['moe_intermediate_size']
        self.shared_expert_intermediate_size = model_config['shared_expert_intermediate_size']
        self.num_experts_per_tok = model_config['num_experts_per_tok']
        self.num_experts = model_config['num_experts']
        self.norm_topk_prob=model_config['norm_topk_prob']
        self.gpu_count = torch.cuda.device_count()
        if self.rope_scaling is None:
            self.rope_scaling = 1.0
        assert model_config["hidden_act"] == "silu"
    
    @staticmethod
    def load_from_model_path(model_path: str) -> "Qwen2MoEModelConfig":
        with open(os.path.join(model_path, "config.json"),"r",encoding = 'utf-8') as f:
            model_config_dict = json.load(f)
        return Qwen2MoEModelConfig(model_config_dict)