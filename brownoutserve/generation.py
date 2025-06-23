import json
import time
from typing import List, Optional, Tuple
import torch

from brownoutserve.gpu_manager_config import GPUManagerConfig

from transformers import  AutoTokenizer

from brownoutserve.model import Qwen2MoE
from brownoutserve.brownout_config import BrownoutConfig
from brownoutserve.slo_analyzer import SLOAnalyzer
from brownoutserve.scheduler import Scheduler
from brownoutserve.expert_loader import ExpertLoader

total_time_usage=[]

class LLM:
    @staticmethod
    def build(
        model_path: str,
        max_seq_len: int,
        max_batch_size: int,
        brownout_config:BrownoutConfig=None,
        devices: str = 'cuda:0',
        dtype:torch.dtype=torch.float16,
        seed: int = 1,

    ):
        # torch.manual_seed(seed)
        devices =sorted(list(set(devices)))

        default_device = devices[0]
        torch.cuda.set_device(default_device)
        gpu_manager_config = GPUManagerConfig(
                gpu_mem_utilization = 0.6,
                max_seqs_in_table = max_batch_size, # By default, the max number of seqs in the block table is set equal to max_batch_size.
                max_seq_len = max_seq_len,
                max_batch_size = max_batch_size,
                block_size = 16,
                dtype=dtype
        )  
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = Qwen2MoE(model_path,gpu_manager_config,brownout_config,devices=devices,dtype=dtype)
        expert_loader = ExpertLoader(model)
        model.warm_up()
        print("model warms up end!")
        return LLM(model, tokenizer,expert_loader, default_device)
    def __init__(self, model:Qwen2MoE, tokenizer: AutoTokenizer,expert_loader: ExpertLoader, default_device:str):
        self.model = model
        self.tokenizer = tokenizer
        self.expert_loader = expert_loader
        self.default_device = default_device
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6, #0.6 is an advisable value
        top_p: float = 0.9,
        mode: str = "normal",
        layer_id: int =-1,
        brownout_config:BrownoutConfig = None,
        early_stop:bool=True,
        record=None,
    ):
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            input_ids (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
           
        """
        self.model.gpu_manager.reset()
        outputs = [[] for _ in range(len(input_ids))]
        eos_mask = [False for _ in range(len(input_ids))]

         # prefill phase
        torch.cuda.synchronize(self.default_device)
        t1 = time.perf_counter()
        logits,seq_ids = self.model.forward(
                        input_ids_list = input_ids,
                        num_decoding_seqs=0,
                        decoding_seq_ids = torch.empty(0,device=self.model.device,dtype=torch.int32),
                        use_kv_cache=True,
                        mode = mode,
                        layer_id = layer_id,
                        brownout_config=brownout_config
            )
        # print(seq_ids)
        if temperature>0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)


        else:
            next_token  = torch.argmax(logits, dim=1)


        if early_stop:
            cur_eos_mask = (next_token == self.tokenizer.eos_token_id)
        else:
             cur_eos_mask = torch.zeros(next_token.shape[0],dtype=torch.bool, device=next_token.device)
        # cur_eos_mask = next_token == self.tokenizer.eos_token_id  # [seq_len_not_end]
        finished_seq_ids = seq_ids[cur_eos_mask]
        if finished_seq_ids.shape[0] > 0:
            self.model.release_finised_seqs_resource(finished_seq_ids)
        seq_ids_list=seq_ids.tolist()
        next_token_list = next_token.tolist()
        for i in range(len(next_token_list)):
            outputs[seq_ids_list[i]].append(next_token_list[i])
        seq_ids = seq_ids[~cur_eos_mask]
        next_token = next_token[~cur_eos_mask].tolist()
        t2 = time.perf_counter()
        if record is not None:
            record["prefilling_latency"].append(t2-t1)

        # print('total_time_usage:',t2-t1)
        # decoding phase
        for _ in range(max_gen_len-1):
            # print("bsz is",len(next_token))
            if len(next_token) == 0:
                break
            torch.cuda.synchronize(self.default_device)
            t1 = time.perf_counter()
            logits,_ = self.model.forward(
                input_ids_list = [[x] for x in next_token],
                num_decoding_seqs = len(next_token),
                decoding_seq_ids =seq_ids,
                mode = mode,
                layer_id=layer_id,
                brownout_config=brownout_config
            )
            if temperature>0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token  = torch.argmax(logits, dim=1)
            torch.cuda.synchronize(self.default_device)
            t2 = time.perf_counter()
            total_time_usage.append(t2-t1)
            if record is not None:
                record["decoding_latency"].append(t2-t1)
            if early_stop:
                cur_eos_mask = (next_token == self.tokenizer.eos_token_id)
            else:
                cur_eos_mask = torch.zeros(next_token.shape[0],dtype=torch.bool, device=next_token.device)
            # cur_eos_mask = next_token == self.tokenizer.eos_token_id  # [seq_len_not_end]
            finished_seq_ids = seq_ids[cur_eos_mask]
            if finished_seq_ids.shape[0] > 0:
                self.model.release_finised_seqs_resource(finished_seq_ids)
            seq_ids_list=seq_ids.tolist()
            next_token_list = next_token.tolist()
            for i in range(len(next_token_list)):
                outputs[seq_ids_list[i]].append(next_token_list[i])
            seq_ids = seq_ids[~cur_eos_mask]
            next_token = next_token[~cur_eos_mask].tolist()

        self.model.release_finised_seqs_resource(seq_ids)

        return outputs
    @torch.inference_mode()
    def online_inference(
        self,
        input_ids: List[List[int]],  # first part is prefilling, second part is decoding
        decoding_seqs_ids_list: List[int],
        scheduler:Scheduler,
        max_gen_len: int=100,
        temperature: float = 0.6,  # 0.6 is an advisable value
        top_p: float = 0.9,
        mode: str = "normal",
        early_stop: bool = True,
        layer_id: int =-1,
        brownout_config:BrownoutConfig = None,
        record=None,
    )-> Tuple[List[int],List[int]]:
        # torch.cuda.synchronize(device=0)
        t1 = time.perf_counter()
        brownout_config.trace['decoding_seq_ids_list'] = decoding_seqs_ids_list
        # print(f'this iteration,nums of prefilling is {len(input_ids)-len(decoding_seqs_ids_list)}, nums of decoding is {len(decoding_seqs_ids_list)}, total is {len(input_ids)}')
        logits, prefilling_seq_ids = self.model.forward(
            input_ids_list=input_ids,
            num_decoding_seqs=len(decoding_seqs_ids_list),

            decoding_seq_ids=torch.tensor(
                decoding_seqs_ids_list, device=self.model.device, dtype=torch.int64
            ),
            use_kv_cache=True,
            mode=mode,
            layer_id=layer_id,
            brownout_config=brownout_config
        )
        num_prefilling_seq = 0 if prefilling_seq_ids is None else prefilling_seq_ids.shape[0]
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)

        else:
            next_token = torch.argmax(logits, dim=1)

    
        if prefilling_seq_ids is not None:
            prefilling_seq_ids = prefilling_seq_ids.to(self.default_device)
            decoding_ids = torch.cat(
                (prefilling_seq_ids,
                torch.tensor(decoding_seqs_ids_list, device=self.default_device,dtype=torch.int64)),
                dim=0

            )  # [cur_process_seq_len]
            # print(decoding_ids.dtype)
        else:
            decoding_ids = torch.tensor(decoding_seqs_ids_list, device=self.default_device,dtype=torch.int64)
        if early_stop:
            # print(next_token,self.tokenizer.eos_token_id)
            cur_eos_mask = (next_token == self.tokenizer.eos_token_id).to(self.default_device)
        else:
             cur_eos_mask = torch.zeros(next_token.shape[0],dtype=torch.bool, device=self.default_device)
        decoding_ids_list = decoding_ids.tolist()

        max_len_seq_indices=[]
        for i,id in enumerate(decoding_ids_list):
            if len(scheduler.outputs[id])>=max_gen_len-1 or len(scheduler.outputs[id])==scheduler.output_length_list[id]-1 or scheduler.output_length_list[id]==1:
                # print(f'id:{id},length:{len(scheduler.outputs[id])})')
                # print(len(scheduler.outputs[id]),scheduler.output_length_list[id])
                max_len_seq_indices.append(i)
        cur_eos_mask[max_len_seq_indices]=True

        finished_seq_ids = decoding_ids[cur_eos_mask]
        if finished_seq_ids.shape[0] > 0:
            self.model.release_finised_seqs_resource(finished_seq_ids)
        next_token_list = next_token.tolist()
        scheduler.record_throughput(len(next_token_list))
        # torch.cuda.synchronize(device=0)
        t2 = time.perf_counter()
        if num_prefilling_seq>0:
            self.model.brownout_config.trace['prefilling_total']+=t2-t1
            self.model.brownout_config.trace['prefilling_iteration']+=1
        else:
            self.model.brownout_config.trace['decoding_total']+=t2-t1
            self.model.brownout_config.trace['decoding_iteration']+=1
        for i in range(len(decoding_ids_list)):
            scheduler.outputs[decoding_ids_list[i]].append(next_token_list[i])
            scheduler.time_usage[decoding_ids_list[i]].append(t2-t1)
        finished_seq_ids_list = finished_seq_ids.tolist()
        self.process_finished_seqs(finished_seq_ids_list,scheduler)
       
        return decoding_ids[~cur_eos_mask].tolist(), finished_seq_ids_list
        
    def process_finished_seqs(self,seq_ids:List[int],scheduler: Scheduler):
        for id in seq_ids:
            output_tokens = scheduler.outputs[id]
            output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=False)
            prompt  = scheduler.prompts[id]
            scheduler.add_output_to_dict(prompt,output_text)
            info = {
                "seq_id": id,
                "prompt": prompt,
                "output_text": output_text,
                "output_length": len(scheduler.outputs[id]),
                # "prefilling_latency": scheduler.get_prefilling_latency(id),
                # "decoding_average_latency": scheduler.get_average_decoding_latency(id),
            }
            condition = scheduler.condition_dict[prompt]
            with condition:
                condition.notify()
                print(json.dumps(info,indent=2,ensure_ascii=False))
        
            # scheduler.add_decoding_latency(info['decoding_average_latency'])
            # scheduler.add_prefilling_latency(info['decoding_average_latency'])
    @torch.inference_mode()
    def offline_inference(
            self,
            input_ids: List[List[int]],
            max_gen_len: int,
            temperature: float = 0.6,  # 0.6 is an advisable value
            top_p: float = 0.9,
            mode: str = "normal",
            layer_id: int =-1,
            brownout_config:BrownoutConfig = None,
        ):
            """
            Generate text sequences based on provided prompts using the language generation model.

            Args:
                input_ids (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
                max_gen_len (int): Maximum length of the generated text sequence.
                temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
                top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.

            """
            self.model.gpu_manager.reset()
            outputs = [[] for _ in range(len(input_ids))]
            eos_mask = [False for _ in range(len(input_ids))]
            # prefill phase

            logits, seq_ids = self.model.forward(
                input_ids_list=input_ids,
                num_decoding_seqs=0,
                decoding_seq_ids=torch.empty(
                    0, device=self.model.device, dtype=torch.int32
                ),
                mode = mode,
                layer_id = layer_id,
                brownout_config=brownout_config,
                use_kv_cache=True,
            )

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)

            else:
                next_token = torch.argmax(logits, dim=1)

            cur_eos_mask = next_token == self.tokenizer.eos_token_id  # [seq_len_not_end]
            seq_ids = seq_ids[~cur_eos_mask]
            cur_eos_mask_list = cur_eos_mask.tolist()
            next_token_list = next_token.tolist()
            j = 0
            for i in range(len(eos_mask)):
                if eos_mask[i]:
                    continue
                eos_mask[i] = cur_eos_mask_list[j]
                outputs[i].append(next_token_list[j])
                j += 1
            next_token = next_token[~cur_eos_mask].tolist()

            # decoding phase
            for _ in range(max_gen_len - 1):
    
                if len(next_token) == 0:
                    break
                # print('input_ids_list',[[x] for x in next_token],'num_decoding_seqs',len(next_token),'decoding_seq_ids',seq_ids)
                logits, _ = self.model.forward(
                    input_ids_list=[[x] for x in next_token],
                    num_decoding_seqs=len(next_token),
                    decoding_seq_ids=seq_ids,
                    mode = mode,
                    layer_id = layer_id,
                    brownout_config=brownout_config,
                )
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p)
                else:
                    next_token = torch.argmax(logits, dim=1)
                cur_eos_mask = (
                    next_token == self.tokenizer.eos_token_id
                )  # [seq_len_not_end]
                seq_ids = seq_ids[~cur_eos_mask]
                cur_eos_mask_list = cur_eos_mask.tolist()
                next_token_list = next_token.tolist()
                j = 0
                for i in range(len(eos_mask)):
                    if eos_mask[i]:
                        continue
                    eos_mask[i] = cur_eos_mask_list[j]
                    outputs[i].append(next_token_list[j])
                    j += 1
                next_token = next_token[~cur_eos_mask].tolist()

            return outputs
    
    @torch.inference_mode()
    def online_inference_slo(
        self,
        input_ids: List[List[int]],  # first part is prefilling, second part is decoding
        decoding_seqs_ids_list: List[int],
        scheduler:Scheduler,
        slo_analyzer:SLOAnalyzer,
        max_gen_len: int=100,
        temperature: float = 0.6,  # 0.6 is an advisable value
        top_p: float = 0.9,
        mode: str = "normal",
        early_stop: bool = True,
        layer_id: int =-1,
        brownout_config:BrownoutConfig = None,
        record=None,
    )-> Tuple[List[int],List[int]]:
        # torch.cuda.synchronize(device=0)
        t1 = time.perf_counter()
        # decoding 
        if len(input_ids)==len(decoding_seqs_ids_list):
            self.model.brownout_config.top_p = slo_analyzer.decoding_brownout_threshold
        else:
            # prefilling
            self.model.brownout_config.top_p = slo_analyzer.prefilling_brownout_threshold
        # brownout_config.trace['decoding_seq_ids_list'] = decoding_seqs_ids_list
        # print(f'this iteration,nums of prefilling is {len(input_ids)-len(decoding_seqs_ids_list)}, nums of decoding is {len(decoding_seqs_ids_list)}, total is {len(input_ids)}')
        logits, prefilling_seq_ids = self.model.forward(
            input_ids_list=input_ids,
            num_decoding_seqs=len(decoding_seqs_ids_list),

            decoding_seq_ids=torch.tensor(
                decoding_seqs_ids_list, device=self.model.device, dtype=torch.int64
            ),
            use_kv_cache=True,
            mode=mode,
            layer_id=layer_id,
            brownout_config=brownout_config
        )
        num_prefilling_seq = 0 if prefilling_seq_ids is None else prefilling_seq_ids.shape[0]
        # print(f'num_prefilling_seq:{num_prefilling_seq },num_decoding_seq:{len(decoding_seqs_ids_list)}')
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)

        else:
            next_token = torch.argmax(logits, dim=1)

    
        if prefilling_seq_ids is not None:
            prefilling_seq_ids = prefilling_seq_ids.to(self.default_device)
            decoding_ids = torch.cat(
                (prefilling_seq_ids,
                torch.tensor(decoding_seqs_ids_list, device=self.default_device,dtype=torch.int64)),
                dim=0

            )  # [cur_process_seq_len]
            # print(decoding_ids.dtype)
        else:
            decoding_ids = torch.tensor(decoding_seqs_ids_list, device=self.default_device,dtype=torch.int64)
        if early_stop:
            cur_eos_mask = (next_token == self.tokenizer.eos_token_id).to(self.default_device)
        else:
             cur_eos_mask = torch.zeros(next_token.shape[0],dtype=torch.bool, device=self.default_device)
        decoding_ids_list = decoding_ids.tolist()

        max_len_seq_indices=[]
        for i,id in enumerate(decoding_ids_list):
            if len(scheduler.outputs[id])==max_gen_len-1 or len(scheduler.outputs[id])==scheduler.output_length_list[id]-1 or scheduler.output_length_list[id]==1:
                # print(f'id:{id},length:{len(scheduler.outputs[id])})')
                # print(len(scheduler.outputs[id]),scheduler.output_length_list[id])
                max_len_seq_indices.append(i)
        cur_eos_mask[max_len_seq_indices]=True
       

        finished_seq_ids = decoding_ids[cur_eos_mask]
        if finished_seq_ids.shape[0] > 0:
            self.model.release_finised_seqs_resource(finished_seq_ids)
        next_token_list = next_token.tolist()
        scheduler.record_throughput(len(next_token_list))
        # torch.cuda.synchronize(device=0)
        t2 = time.perf_counter()
        token_sum=0
        for item in input_ids:
            token_sum+=len(item)
        # print('prefilling latency:',t2-t1,token_sum)
        latency = t2-t1
        cur_time = int(time.time())
        if num_prefilling_seq>0:
            # scheduler.add_prefilling_latency(t2-t1)
            
            scheduler.record_prefilling_latency(latency,self.model.brownout_config.top_p,cur_time)
            slo_analyzer.record_prefilling_latency(latency,cur_time)
            slo_analyzer.dynamic_prefilling_threshold_adjustment(cur_time)

            print('prefilling latency:',latency,token_sum,slo_analyzer.prefilling_brownout_threshold)
        else:
            scheduler.record_decoding_latency(latency,self.model.brownout_config.top_p,cur_time)
            slo_analyzer.record_decoding_latency(latency,cur_time)
            slo_analyzer.dynamic_decoding_threshold_adjustment(cur_time)

        for i in range(len(decoding_ids_list)):
            scheduler.outputs[decoding_ids_list[i]].append(next_token_list[i])
            scheduler.time_usage[decoding_ids_list[i]].append(latency)
        finished_seq_ids_list = finished_seq_ids.tolist()
        # self.process_finished_seqs(finished_seq_ids_list,scheduler)
       
        return decoding_ids[~cur_eos_mask].tolist(), finished_seq_ids_list   
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token).view(-1)
    return next_token


