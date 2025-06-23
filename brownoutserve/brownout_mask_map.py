from typing import List
import torch


class BrownoutMaskMap:
    def __init__(self,devices:List[str],num_experts:int):
        defalut_device = devices[0]
        self.devices = devices
        self.num_experts = num_experts
        self.mask_map={}
        for device in devices:
            self.mask_map[device]=[None for _ in range(self.num_experts+1)]
        
        for i in range(self.num_experts+1):
            self.mask_map[defalut_device][i] = self.generate_random_bool_tensor(num_experts,i,defalut_device)
        
        for device in self.devices[1:]:
            for i in range(self.num_experts+1):
                self.mask_map[device][i] = self.mask_map[defalut_device][i].to(device)

    def generate_random_bool_tensor(self,n, m,device):
        # 创建一个长度为n的全False tensor
        tensor = torch.zeros(n, dtype=torch.bool,device=device)
        
        # 随机选择m个位置，将它们设置为True
        true_indices = torch.randperm(n)[:m]
        tensor[true_indices] = True
        
        return tensor

    def get_mask_by_threshold(self,threshold:float,device:str):
        return self.mask_map[device][int((1-threshold)*self.num_experts)]