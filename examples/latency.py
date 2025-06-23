import json
from typing import List

import torch
from brownoutserve.generation import LLM
from brownoutserve.brownout_config import BrownoutConfig





way=8
greedy=True
debug_info=[[] for _ in range(24)]
brownout_config = BrownoutConfig(top_p=1,way=way,united_experts_weight_dirctory=f"/root/hujianmin/qwen2_moe_i/{way}_way_united_experts_test",debug_info=debug_info,greedy=greedy,use_fused_moe=False,full_brownout_mode=True)
batch_size = 64
model = LLM.build(
            model_path="/root/llm-resource/Models/Qwen1.5-MoE-A2.7B-Chat",
            max_seq_len=256, # 2048 # 512
            max_batch_size=64,
            brownout_config=brownout_config,
            dtype=torch.float16,
            devices=["cuda:0","cuda:1","cuda:2","cuda:3"]
            )


with open("/root/hujianmin/qwen2_moe_i/alpaca_data.json") as f:
    data = json.load(f)

total = len(data)


for i in range((total+batch_size-1)//batch_size):
    partial_data  = data[i:i+batch_size]
    prompts = [item['instruction']+'\n'+item['input'] for item in partial_data]

    input_ids = model.tokenizer(prompts)['input_ids']

    outputs = model.generate(input_ids,max_gen_len=2,temperature = 0,mode="infer",brownout_config=brownout_config)


from brownoutserve.generation import total_time_usage
from brownoutserve.model import moe_time_usage


print('moe_time_usage',moe_time_usage)
print('total_time_usage',total_time_usage)

with open('/root/hujianmin/qwen2_moe_i/result/decoding_moe_time_usage.json','w')as f:
    f.write(json.dumps(moe_time_usage))

with open('/root/hujianmin/qwen2_moe_i/result/decoding_total_time_usage.json','w')as f:
    f.write(json.dumps(total_time_usage))
# for i,prompt in enumerate(prompts):
#         output_tokens = outputs[i]
#         print(output_tokens)
#         output_text = model.tokenizer.decode(output_tokens, skip_special_tokens=False)
#         print(f"User: {prompt} Assistant: {output_text}")
#         print("-"*100)
# #test