from dataclasses import dataclass
from typing import List

from brownoutserve.scheduler import Scheduler

@dataclass
class BrownoutConfig:
    top_p:float = 0.6
    way:int = 2
    full_brownout_mode:bool=False
    united_experts_weight_dirctory:str=""
    greedy:bool=True
    debug_info:int=1
    use_fused_moe:bool=False
    trace:dict=None
    scheduler:Scheduler=None

    
    
        
        
        
            

