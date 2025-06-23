import argparse
from datetime import datetime
import json
import random
import threading
import time
from typing import List

import numpy as np
import torch
from brownoutserve.generation import LLM
from brownoutserve.brownout_config import BrownoutConfig
from brownoutserve.scheduler import Scheduler
from brownoutserve.slo_analyzer import SLOAnalyzer

def poisson_emitter(
    a, config:dict,scheduler: Scheduler = None, dataset_path=None,file_path=None, duration: int = 1000000,middle_point:float=1000000,tokenizer=None
):
    """
    泊松过程模拟，每秒平均输出 a 次 "hello world"
    :param a: 每秒平均事件数
    """
    text_list=[]
    if dataset_path is not None:
        with open(dataset_path) as f:
            text_list = json.load(f)
    else:
        # default text_list
        text_list = [
            "What is Apple?",
            "Is banana a fruit or a nut?",
            "Please briefly explain swimmer Sun Yang",
            "How to calculate 1+2?",
            "What was the outbreak time of the First World War?",
        ]
    print(f'text list length is {len(text_list)}')
    total = len(text_list)

    start_time = time.time()
    cnt = 0
    flag=False
    with open(file_path) as f:
        data = json.load(f)
    while True:
        # interval = np.random.exponential(1 / a)  # 生成指数分布的间隔时间
        interval = 1/a
        time.sleep(interval)  # 等待下一次输出
        # text = text_list[random.randint(0, total - 1)]
        text = text_list[cnt]
        scheduler.add_new_request_output_length(len(tokenizer(text['output'])['input_ids'])//7)
        # scheduler.add_new_request(text['instruction']+'\n'+text['input'])
        scheduler.add_new_request(text['input'])
        cnt += 1
        # print(f'have send {cnt}')
        cur_time = time.time()
        if  cur_time - start_time > middle_point and not flag:
            a=a*2
            flag=True
            print("increase rps")
        if  cur_time - start_time > duration:
            print(
                f"INFO: poisson_emitter quit successfully, and send {cnt} requests totally."
            )
            break
    
    # while scheduler.existing_requests():
        # 这里的逻辑有问题
    time.sleep(1)
    print(f'{len(scheduler.waiting_queue)},{len(scheduler.running_queue)}')
    scheduler.finished_all_requests=True

    # print(json.dumps(scheduler.throughput_recoder,indent=" "))
    sum=0
    for value in scheduler.throughput_recorder.values():
        sum+=value
    avg = sum/len(scheduler.throughput_recorder)
    print(f'avg:{avg:.4}')
    config['throughput'] = avg
    data.append(config)
    # scheduler.report_inference_info()
    scheduler.report_timely_latency()
    # with open(file_path,'w')as f:
    #     f.write(json.dumps(data,indent=" "))

def chat_template(prompts:List[str])->List[str]:
    prompts_with_chat_template=[]
    for prompt in prompts:
        prompt_with_chat_template=f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompts_with_chat_template.append(prompt_with_chat_template)
    
    return prompts_with_chat_template


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(100)
    start_time = time.time()
    parser = argparse.ArgumentParser(description="处理命令行输入的参数")
    parser.add_argument("--way", type=int, help="way")
    parser.add_argument("--threshold", type=float, help="threshold")
    parser.add_argument("--use_fused_moe", action="store_true", help="use_fused_moe")
    parser.add_argument("--rps", type = float,help="rps")
    parser.add_argument("--duration", type = int,help="duration",default=20)
    parser.add_argument("--input_length", type = int,help="output_length",default=1024)
    parser.add_argument("--output_length", type = int,help="output_length",default=128)
    parser.add_argument("--max_batch_size", type = int,help="output_length",default=32)
    parser.add_argument("--max_new_token", type = int,help="max_new_token",default=-1)
    parser.add_argument("--file_path", type = str,help="file_path",default='/root/hujianmin/qwen2_moe_i/result/temp.json')
    parser.add_argument("--middle_point", type = float,help="middle_point",default=5.0)
    parser.add_argument("--brownout_point", type = float,help="middle_point",default=5.0)
    parser.add_argument("--brownout", action="store_true", help="use_fused_moe")
    args = parser.parse_args()
    way=args.way
    threshold = args.threshold
    debug_info=[[] for _ in range(24)]
    # debug_info={}
    greedy=True
    max_batch_size=args.max_batch_size
    use_fused_moe=args.use_fused_moe
    rps=args.rps
    duration=args.duration
    input_length = args.input_length
    output_length = args.output_length
    file_path = args.file_path
    middle_point = args.middle_point
    brownout_point = args.brownout_point
    brownout = args.brownout
    if args.max_new_token==-1:
        max_new_token=output_length
    else:
        max_new_token=args.max_new_token
    env_config={
        'use_fused_moe':use_fused_moe,
        'way':way,
        'max_batch_size':max_batch_size,
        # 'input_length':input_length,
        'output_length':output_length,
        'max_new_token':max_new_token,
        'threshold':threshold,
        'rps':rps,
        'duration':duration,
        'middle_point':middle_point,
        'brownout_point':brownout_point
    }
    print('environment configuration:')
    print(json.dumps(env_config,indent=" "))
    scheduler = Scheduler(max_batch_size=max_batch_size)
    brownout_config = BrownoutConfig(top_p=threshold,way=way,united_experts_weight_dirctory=f"/root/hujianmin/qwen2_moe_i/{way}_way_united_experts_test",debug_info=debug_info,greedy=greedy,full_brownout_mode=False,use_fused_moe=use_fused_moe,trace={},scheduler=scheduler)
    # tokenizer = AutoTokenizer.from_pretrained("/root/llm-resource/Models/Qwen1.5-MoE-A2.7B-Chat")
    model = LLM.build(
            model_path="/root/llm-resource/Models/Qwen1.5-MoE-A2.7B-Chat",
            max_seq_len=output_length, # 2048 # 512
            max_batch_size=max_batch_size,
            brownout_config=brownout_config,
            dtype=torch.float16,
            devices=["cuda:0","cuda:1","cuda:2","cuda:3"]
            )



    slo_analyzer = SLOAnalyzer(max_storage_seconds=30, max_search_seconds=2, prefilling_slo=0.25,decoding_slo=0.15,slo_warning_factor=0.8)

    def excute():
        while not scheduler.finished_all_requests:
            input_ids = []
            decoding_seq_ids_list=[]
            new_requests_list=[]
            output_length_list=[]
            if (not scheduler.waiting_queue_is_empty()) or (not scheduler.running_queue_is_empty()):
                if not scheduler.waiting_queue_is_empty():
                    new_requests_list,output_length_list = scheduler.get_all_requests_from_waiting_queue()
                    #  = scheduler.get_all_output_length()
                    if len(new_requests_list)>0:
                        # prompts_with_chat_template = chat_template(new_requests_list)
                        input_ids.extend(model.tokenizer(new_requests_list)['input_ids'])
                
                if not scheduler.running_queue_is_empty():
                    decoding_seq_ids_list = scheduler.get_all_requests_from_running_queue()
                    input_ids.extend(scheduler.get_last_tokens(decoding_seq_ids_list))
            else:
                # print("There is no waiting requests and running requests!")
                time.sleep(0.01)
                continue
            # print("aa")
            decoding_seq_ids_list,finished_seq_ids = model.online_inference_slo(input_ids,decoding_seq_ids_list,scheduler,slo_analyzer,temperature = 0,max_gen_len=max_new_token,
                                                                            mode='super'if use_fused_moe else 'infer' ,early_stop=False, brownout_config=brownout_config)

            scheduler.add_requests_to_running_queue(decoding_seq_ids_list)
            if len(new_requests_list)>0:
                scheduler.set_prompts_and_length(decoding_seq_ids_list[:len(new_requests_list)],new_requests_list,output_length_list)
            scheduler.clear_finished_seqs(finished_seq_ids)
        print('excute finished','waiting queue length is',len(scheduler.waiting_queue),'running queue length is',len(scheduler.running_queue))



    # def add_request():
    #     while True:
    #         input("请输入任何字母:\n")
    #         scheduler.mock_add_request()

    def mock_send_request():
        poisson_emitter(rps,env_config,scheduler,'/root/hujianmin/qwen2_moe_i/partial_sharegpt.json',file_path,duration=duration,middle_point=middle_point,tokenizer=model.tokenizer)

    thread1 = threading.Thread(target=excute)
    thread2 = threading.Thread(target=mock_send_request)

    thread1.start()
    thread2.start()


    thread1.join()
    thread2.join()
    end_time = time.time()
    print("All tasks completed.",end_time-start_time)
    timestamp = time.time()+8*60*60  # Beijing Time
    dt_object = datetime.fromtimestamp(timestamp)
    formatted_date = dt_object.strftime('%Y-%m-%d %H:%M:%S')
    print('当前时间:',formatted_date)
    # for i,prompt in enumerate(prompts):
    #     output_tokens = outputs[i]
    #     output_text = model.tokenizer.decode(output_tokens, skip_special_tokens=True)
    #     print(f"{prompt}|{output_text}")

    # for i,prompt in enumerate(new_prompts):
    #     output_tokens = outputs[i+5]
    #     output_text = model.tokenizer.decode(output_tokens, skip_special_tokens=True)
    #     print(f"{prompt}|{output_text}")