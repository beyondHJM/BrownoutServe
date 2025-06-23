from typing import List

import torch
from brownoutserve.generation import LLM
from brownoutserve.brownout_config import BrownoutConfig



def chat_template(prompts:List[str])->List[str]:
    prompts_with_chat_template=[]
    for prompt in prompts:
        prompt_with_chat_template=f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompts_with_chat_template.append(prompt_with_chat_template)
    
    return prompts_with_chat_template



way=8
greedy=True
debug_info=[[] for _ in range(24)]
brownout_config = BrownoutConfig(top_p=0.6,way=way,united_experts_weight_dirctory=f"/root/hujianmin/qwen2_moe_i/{way}_way_united_experts_test",use_fused_moe=True)
prompts =[
 
"what is Huawei?",

]
model = LLM.build(
            model_path="/root/llm-resource/Models/Qwen1.5-MoE-A2.7B-Chat", # Replace with your own model path
            max_seq_len=256, # 2048 # 512
            max_batch_size=128,
            brownout_config=brownout_config,
            dtype=torch.float16,
            devices=["cuda:1","cuda:2","cuda:3"]
            )
prompts_with_chat_template = chat_template(prompts)
input_ids = model.tokenizer(prompts_with_chat_template)['input_ids']

outputs = model.generate(input_ids,max_gen_len=100,temperature = 0,mode="super",brownout_config=brownout_config)


for i,prompt in enumerate(prompts):
        output_tokens = outputs[i]
        output_text = model.tokenizer.decode(output_tokens, skip_special_tokens=False)
        print(f"User: {prompt} Assistant: {output_text}")
        print("-"*100)
