
import argparse
import json
import random
import time
from typing import List
import torch
from qwen2_moe_i.brownoutserve.generation import LLM
from transformers import  AutoTokenizer
from qwen2_moe_i.brownoutserve.brownout_config import BrownoutConfig

# from input import prompts, max_new_token
# from model import count,gpu_count
from input import prompts

diff = 19
def chat_template(prompts:List[str])->List[str]:
    prompts_with_chat_template=[]
    for prompt in prompts:
        prompt_with_chat_template=f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompts_with_chat_template.append(prompt_with_chat_template)
    
    return prompts_with_chat_template



if __name__=="__main__":
        # torch.manual_seed(100)
        # random.seed(42)
        parser = argparse.ArgumentParser(description="处理命令行输入的参数")
        parser.add_argument("--way", type=int, help="way",default=2)
        parser.add_argument("--batch_size", type=int, help="batch_size",default=16)
        parser.add_argument("--input_length", type=int, help="input_length",default=128)
        parser.add_argument("--greedy", type=str, help="greedy",default="True")
        parser.add_argument("--fused", type=str, help="fused",default="False")
        parser.add_argument("--write", action="store_true", help="write")
        parser.add_argument("--epoch", type=int, help="epoch",default=20)
        args = parser.parse_args()

                # print(input_data[0])
        debug_info=[[] for _ in range(24)]
        batch_size = args.batch_size
        input_length= args.input_length

        top_p_list=[1,0.99,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]
        way=args.way
        full_brownout_mode = True if way==-1 else False
        greedy=True if args.greedy=="True" else False
        fused=True if args.fused=="True" else False
        write = args.write
        epoch = args.epoch
        print(f'batch size:{batch_size}, input_length:{input_length}, way:{way}, greedy:{greedy}')
        
        brownout_config = BrownoutConfig(top_p=top_p_list[0],way=way,united_experts_weight_dirctory=f"/root/hujianmin/qwen2_moe_i/{way}_way_united_experts_test",debug_info=debug_info,greedy=greedy,use_fused_moe=fused,full_brownout_mode=full_brownout_mode)
        # tokenizer = AutoTokenizer.from_pretrained("/root/llm-resource/Models/Qwen1.5-MoE-A2.7B-Chat")
        model = LLM.build(
                model_path="/root/llm-resource/Models/Qwen1.5-MoE-A2.7B-Chat",
                max_seq_len=2048, # 2048 # 512
                max_batch_size=batch_size,
                brownout_config=brownout_config,
                dtype=torch.float16,
                devices=["cuda:0","cuda:1","cuda:2","cuda:3"]
                )


        info=[]
        if write:
                with open('./info_latency.json')as f:
                        info=json.load(f)
        greedy_list=[True]
        for top_p in top_p_list:
          greedy_prefilling=0
          greedy_decoding=0
          for greedy in greedy_list:
                debug_info=[[] for _ in range(24)]
                record={
                'prefilling_latency':[],
                'decoding_latency':[]
        }
                brownout_config.top_p=top_p
                brownout_config.debug_info=debug_info
                brownout_config.greedy=greedy
                for i in range(epoch):
                        with open('./input_with_template.json')as f:
                                input_data = json.load(f)
                                random.shuffle(input_data)
                        prompts = input_data[:batch_size]
                        input_ids = model.tokenizer(prompts)['input_ids']
                        for i in range(len(input_ids)):
                                input_ids[i]=input_ids[i][:input_length]
                        outputs = model.generate(input_ids,max_gen_len=10,temperature = 0,mode="super" if brownout_config.use_fused_moe else "infer",brownout_config=brownout_config,record=record)
                record["prefilling_latency"]=sorted(record["prefilling_latency"][1:])
                record["decoding_latency"] = sorted(record["decoding_latency"])
                # record["p90_prefilling_latency"]=record["prefilling_latency"][int(0.9*len(record["prefilling_latency"]))]
                # record["p90_decoding_latency"]=record["decoding_latency"][int(0.9*len(record["decoding_latency"]))]
                record["average_prefilling_latency"]=sum(record["prefilling_latency"])/len(record["prefilling_latency"])
                record["average_decoding_latency"]=sum(record["decoding_latency"])/len(record["decoding_latency"])
                info_item={
                "way":way,
                "threshold":top_p,
                "batch size":batch_size,
                "input length":input_length,
                "average_prefilling_latency":[],
                "average_decoding_latency":[],
                
                }
                # info_item["p90_prefilling_latency"]=record["p90_prefilling_latency"]
                # info_item["p90_decoding_latency"]=record["p90_decoding_latency"]
                info_item["average_prefilling_latency"]=record["average_prefilling_latency"]
                info_item["average_decoding_latency"]=record["average_decoding_latency"]
                # if greedy:
                #         greedy_prefilling=info_item["p90_prefilling_latency"]
                #         greedy_decoding=info_item["p90_decoding_latency"]
                # else:
                #       print(f'top_p:{top_p},ungreedy/greedy:{info_item["p90_prefilling_latency"]/greedy_prefilling} and {info_item["p90_decoding_latency"]/greedy_decoding}')
                info.append(info_item)
                print(f'top_p:{top_p},average_prefilling_latency:{info_item["average_prefilling_latency"]},average_decoding_latency:{info_item["average_decoding_latency"]}')
                # print(brownout_config.debug_info)
        if write:
                with open('./info_latency.json','w')as f:
                        f.write(json.dumps(info,indent=""))
        else:
                with open('./temp.json','w')as f:
                        f.write(json.dumps(info,indent=""))