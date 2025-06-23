import argparse
import threading
import time
from typing import List
from flask import Flask, request, jsonify
import torch
from brownoutserve.generation import LLM
from brownoutserve.scheduler import Scheduler
from brownoutserve.brownout_config import BrownoutConfig
def chat_template(prompts:List[str])->List[str]:
    prompts_with_chat_template=[]
    for prompt in prompts:
        prompt_with_chat_template=f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompts_with_chat_template.append(prompt_with_chat_template)
    
    return prompts_with_chat_template
use_fused_moe=True
# 创建 FastAPI 实例
app = Flask(__name__)
def excute(scheduler:Scheduler):
    while True:
        input_ids = []
        decoding_seq_ids_list=[]
        new_requests_list=[]
        output_length_list=[]
        if (not scheduler.waiting_queue_is_empty()) or (not scheduler.running_queue_is_empty()):
            if not scheduler.waiting_queue_is_empty():
                new_requests_list,output_length_list = scheduler.get_all_requests_from_waiting_queue()
                #  = scheduler.get_all_output_length()
                if len(new_requests_list)>0:
                    prompts_with_chat_template = chat_template(new_requests_list)
                    input_ids.extend(model.tokenizer(prompts_with_chat_template)['input_ids'])
            if not scheduler.running_queue_is_empty():
                decoding_seq_ids_list = scheduler.get_all_requests_from_running_queue()
                input_ids.extend(scheduler.get_last_tokens(decoding_seq_ids_list))
        else:
            # print("There is no waiting requests and running requests!")
            time.sleep(0.01)
            continue

        decoding_seq_ids_list,finished_seq_ids = model.online_inference(input_ids,decoding_seq_ids_list,scheduler,temperature = 0,max_gen_len=512,
                                                                        mode='super'if use_fused_moe else 'infer' ,early_stop=True, brownout_config=brownout_config)

        scheduler.add_requests_to_running_queue(decoding_seq_ids_list)
        if len(new_requests_list)>0:
            scheduler.set_prompts_and_length(decoding_seq_ids_list[:len(new_requests_list)],new_requests_list,output_length_list)
        scheduler.clear_finished_seqs(finished_seq_ids)

# 使用 argparse 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="FastAPI Example with command line arguments")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for the server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port for the server (default: 8000)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (default: False)")
    parser.add_argument("--max_batch_size", type=int, default=32, help="Max batch size for LLM server.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length for LLM server.")
    return parser.parse_args()

# 定义命令行参数
global_scheduler=None
global_model = None

@app.route('/')
def hello_world():
    return 'Hello, Flask!'

# 带参数的 API 示例
@app.route('/generate', methods=['GET'])
def add():
    t1 = time.time()
    prompt = request.args.get('prompt', "")
    condition = threading.Condition()
    global_scheduler.condition_dict[prompt] = condition
    scheduler.add_new_request(prompt)
    scheduler.add_new_request_output_length(1024)
    
    with condition:
        condition.wait()
        output_text  = scheduler.get_output_by_promot(prompt)
        t2 = time.time()
        # output_text = global_model.tokenizer.decode(output_tokens, skip_special_tokens=True)
        print(f'prompt:{prompt}')
        print(f'output:{output_text}')
        return jsonify(result=output_text,latency = t2-t1)


@app.route('/rolling_update', methods=['POST'])
def rolling_update():
    way = int(request.json.get('way', ""))
    united_experts_weight_path = request.json.get('united_experts_weight_path', "")
    model.expert_loader.update_united_experts(united_experts_weight_path=united_experts_weight_path,way=way)
    model.expert_loader.show_experts_shape(4)
    return jsonify(result='yes')
@app.route('/show', methods=['GET'])
def show():
    
    model.expert_loader.show_experts_shape(4)
    return jsonify(result='yes')
if __name__ == '__main__':
    print("aaa")

    args = parse_args()
    threshold =0.6
    way = 2
    scheduler = Scheduler(max_batch_size=args.max_batch_size)
    global_scheduler = scheduler
    trace={
        'use_fused_moe':use_fused_moe,
        'way':way,
        'threshold':threshold,
        'prefilling_moe':0,
        'prefilling_attention':0,
        'prefilling_iteration':0,
        'prefilling_total':0,
        'decoding_attention':0,
        'decoding_moe':0,
        'decoding_iteration':0,
        'decoding_total':0,
        'average_prefilling_moe':0,
        'average_decoding_moe':0
    }
    brownout_config = BrownoutConfig(top_p=threshold,way=way,united_experts_weight_dirctory=f"/root/hujianmin/qwen2_moe_i/{way}_way_united_experts_test",full_brownout_mode=False,use_fused_moe=use_fused_moe,scheduler=scheduler,trace=trace,debug_info={})
    model = LLM.build(
                model_path="/root/llm-resource/Models/Qwen1.5-MoE-A2.7B-Chat",
                max_seq_len=args.max_seq_length, # 2048 # 512
                max_batch_size=args.max_batch_size,
                brownout_config=brownout_config,
                dtype=torch.float16,
                devices=["cuda:1","cuda:2","cuda:3"]
                )
    global_model = model
    # time.sleep(100)
    thread1 = threading.Thread(target=excute,args=(scheduler,))
    thread1.start()
    
    app.run(debug=True, use_reloader=False)
    thread1.join()
