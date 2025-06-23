from brownoutserve.model import Qwen2MoE
from brownoutserve.model import Expert
import torch
from tqdm import tqdm


class ExpertLoader:
    def __init__(self,model:Qwen2MoE):
        self.model = model
        print("Experts Loader init successfully!")
    
    def update_united_experts(self,united_experts_weight_path:str,way:int):
        num_united_experts = (self.model.model_config.num_experts+way-1) // way
        weight_dict={}
        experts_once = 8
        for i in range(1,4):
                result = torch.load(f'{united_experts_weight_path}/w{i}.pth',map_location=torch.device('cpu'))
                weight_dict.update(result)
        
        if not self.model.brownout_config.use_fused_moe:
            for i in range(self.model.model_config.num_layers):
                for j in range(num_united_experts):
                    weight_dict[f'{i}_{j}_up_gate_proj.weight'] = torch.cat((weight_dict[f'{i}_{j}_up_proj.weight'],weight_dict[f'{i}_{j}_gate_proj.weight']),dim=0)
                    
            with tqdm(total=(self.model.model_config.num_layers-1+experts_once)//experts_once, desc="rolling updating") as pbar:
                for k in range((self.model.model_config.num_layers-1+experts_once)//experts_once):
                    for i in range(experts_once):
                        layer_id = k*experts_once+i
                        for j in range(num_united_experts):
                            device =  self.model.transformer_layers[layer_id].united_experts[j].down_proj.device
                            weight_dict[f'{layer_id}_{j}_up_gate_proj.weight'] = weight_dict[f'{layer_id}_{j}_up_gate_proj.weight'].to(device)
                            weight_dict[f'{layer_id}_{j}_down_proj.weight'] = weight_dict[f'{layer_id}_{j}_down_proj.weight'].to(device)
                        self.model.transformer_layers[layer_id].united_experts = [Expert(weight_dict[f'{layer_id}_{j}_down_proj.weight'],weight_dict[f'{layer_id}_{j}_up_gate_proj.weight'],self.model.model_config,layer_id,j) for j in range(num_united_experts)]
                        self.model.transformer_layers[layer_id].way = way
                    pbar.update(1)
        
        if self.model.brownout_config.use_fused_moe:
            num_experts = self.model.model_config.num_experts
            moe_intermediate_size = self.model.model_config.moe_intermediate_size
            hidden_size = self.model.model_config.hidden_size
            with tqdm(total=(self.model.model_config.num_layers-1+experts_once)//experts_once, desc="rolling updating") as pbar:
                for k in range((self.model.model_config.num_layers-1+experts_once)//experts_once):
                    for i in range(experts_once):
                        layer_id = k*experts_once+i
                        device =  self.model.transformer_layers[layer_id].mlp_experts_up_proj_packaged.device
                        up_proj_pacakged = torch.cat([weight_dict[f'{layer_id}_{j}_up_proj.weight'] for j in range(num_united_experts)],dim=0).to(device)
                        gate_proj_pacakged = torch.cat([weight_dict[f'{layer_id}_{j}_gate_proj.weight'] for j in range(num_united_experts)],dim=0).to(device)
                        down_proj_pacakged = torch.cat([weight_dict[f'{layer_id}_{j}_down_proj.weight'] for j in range(num_united_experts)],dim=0).to(device)
                        self.model.transformer_layers[layer_id].mlp_experts_up_proj_packaged = torch.cat((self.model.transformer_layers[layer_id].mlp_experts_up_proj_packaged[:num_experts*moe_intermediate_size],up_proj_pacakged),dim=0)
                        self.model.transformer_layers[layer_id].mlp_experts_gate_proj_packaged = torch.cat((self.model.transformer_layers[layer_id].mlp_experts_gate_proj_packaged[:num_experts*moe_intermediate_size],gate_proj_pacakged),dim=0)
                        self.model.transformer_layers[layer_id].mlp_experts_down_proj_packaged = torch.cat((self.model.transformer_layers[layer_id].mlp_experts_down_proj_packaged[:num_experts*hidden_size],down_proj_pacakged),dim=0)
                        # self.model.transformer_layers[layer_id].united_experts = [Expert(weight_dict[f'{layer_id}_{j}_down_proj.weight'],weight_dict[f'{layer_id}_{j}_up_gate_proj.weight'],self.model.model_config,layer_id,j) for j in range(num_united_experts)]
                        # torch.cuda.synchronize(device=device)
                        self.model.transformer_layers[layer_id].way = way
                    pbar.update(1)
            
            
                        
                    
                
    
    def show_experts_shape(self,layer_id:int):
        if not self.model.brownout_config.use_fused_moe:
            experts = self.model.transformer_layers[layer_id].united_experts
            print("non fused experts info is:")
            print(len(experts))
            
        else:
            print("non fused experts info is:")
            print(self.model.transformer_layers[layer_id].mlp_experts_up_proj_packaged.shape)
        
        