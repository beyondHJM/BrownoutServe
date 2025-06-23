
import json
import time
import torch
from brownoutserve.generation import LLM
from brownoutserve.brownout_config import BrownoutConfig

# from input import prompts, max_new_token
top_p=0
way=8
debug_info=[[] for _ in range(24)]
debug_info=debug_info
brownout_config = BrownoutConfig(top_p=top_p,way=way,united_experts_weight_dirctory=f"/root/hujianmin/qwen2_moe_i/{way}_way_united_experts_test",use_fused_moe=True,debug_info=debug_info)
model = LLM.build(
            model_path="/root/llm-resource/Models/Qwen1.5-MoE-A2.7B-Chat",
            max_seq_len=1024, # 2048 # 512
            max_batch_size=64,
            brownout_config=brownout_config,
            dtype=torch.float16,
            devices=["cuda:3","cuda:2","cuda:0","cuda:1"]
        )


prompts=[
#      "A B C D E",
     "A B C D E",
     "one two three four five",
     "When I was just a lad of ten, my father said to",
     "Today I bought some",
     "A majestic tiger walking through ",
     "马龙是一名乒乓球",
#   
     "中国的首都在",

]
with open("/root/hujianmin/qwen2_moe_i/input_with_template.json") as f:
            data = json.load(f)
            text_list =[]
            for item in data:
                    if len(item)>512:
                        text_list.append(item[:512])
prompts=text_list[:32]

 

input_ids = model.tokenizer(prompts)['input_ids']
# torch.cuda.synchronize()
for i in range(8):
        t1 = time.perf_counter()
        outputs = model.generate(input_ids,max_gen_len=50,temperature = 0,mode="super" if brownout_config.use_fused_moe else 'infer',brownout_config=brownout_config,early_stop=False)
        t2 = time.perf_counter()
        print(t2-t1)

# for i,prompt in enumerate(prompts):
#         output_tokens = outputs[i]
#         output_text = model.tokenizer.decode(output_tokens, skip_special_tokens=False)
#         print(f"{prompt}|{output_text}")
#         print("-"*100)


