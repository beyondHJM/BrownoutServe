import json
import threading
import time
from typing import List
from brownoutserve.utils import calculate_average_and_p90

class Scheduler:
    def __init__(self,max_batch_size):
        self.condition = threading.Condition()
        self.max_batch_size = max_batch_size
        self.waiting_queue = [] #List[str]
        self.prompts=[[] for _ in range(max_batch_size)]
        self.running_queue = [] #List[int]
        self.finished_queue= [] #List[object]
        self.outputs = [[] for _ in range(max_batch_size)]
        self.time_usage = [[] for _ in range(max_batch_size)]
        self.finished_all_requests = False
        self.output_length_list = [0 for _ in range(max_batch_size)]
        self.waiting_queue_output_length = []
        self.prefilling_latency_recorder={}
        self.prefilling_threshold_recorder={}
        self.decoding_latency_recorder={}
        self.decoding_threshold_recorder={}
        self.throughput_recorder={}
        self.output_dict_list = {}
        self.condition_dict={}
    def waiting_queue_is_empty(self)->bool:
        return len(self.waiting_queue)==0
    
    def add_new_request(self,prompt:str|List[str]):
        if isinstance(prompt,str):
            self.waiting_queue.append(prompt)
        else :
            self.waiting_queue.extend(prompt)
    
    def add_new_request_output_length(self,length:int):
        if isinstance(length,int):
            self.waiting_queue_output_length.append(length)
        else :
            self.waiting_queue_output_length.extend(length)
    
    def add_requests_to_running_queue(self,seq_ids:List[int]):
        self.running_queue.extend(seq_ids)

    def get_all_requests_from_waiting_queue(self)->List[int]:
         num = min(self.free_requests_num,len(self.waiting_queue))
         new_requests_list = self.waiting_queue[:num]
         output_length_list = self.waiting_queue_output_length[:num]
         self.waiting_queue=self.waiting_queue[num:]
         self.waiting_queue_output_length = self.waiting_queue_output_length[num:]
         return new_requests_list,output_length_list

    def get_partial_requests_from_waiting_queue(self,num:int)->List[str]:
        result = self.waiting_queue[:num]
        self.waiting_queue=  self.waiting_queue[num:]
        return result

    def running_queue_is_empty(self)->bool:
        return len(self.running_queue)==0
    
    def get_all_requests_from_running_queue(self)->List[int]:
         result = self.running_queue
         self.running_queue=[]
         return result


    def get_partial_requests_from_running_queue(self,num:int)->List[int]:
        result = self.running_queue[:num]
        self.running_queue=  self.running_queue[num:]
        return result
    
    def get_last_tokens(self,seqs_ids:List[int])->List[List[int]]:
        return [[self.outputs[id][-1]] for id in seqs_ids]


    def clear_finished_seqs(self,finished_seq_ids:List[int]):
        for id in finished_seq_ids:
            self.outputs[id]=[]
            self.prompts[id]=[]
            self.output_length_list[id]=0
            self.time_usage[id]=[]

    def set_prompts_and_length(self,prefilling_seq_ids:List[int],new_requests_list:List[str],output_length_list:List[int]):
        for i in range(len(prefilling_seq_ids)):
                self.prompts[prefilling_seq_ids[i]] = new_requests_list[i]
                self.output_length_list[prefilling_seq_ids[i]] = output_length_list[i]
    def get_p90_decoding_latency(self,seq_id):
        times_decoding = self.time_usage[seq_id][1:]
        l =  len(times_decoding)
        if l==0:
            return None
        return times_decoding[int(l*0.9)]
    
    def get_prefilling_latency(self,seq_id):
        return self.time_usage[seq_id][0]
    
    def mock_add_request(self):
        with open('./waiting_request.json') as f:
            new_requests = json.load(f)
        for request in new_requests:
            self.add_new_request(request)
        #清空
        with open('./waiting_request.json','w') as f:
            f.write(json.dumps([]))
    
    @property
    def free_requests_num(self):
        return self.max_batch_size-len(self.running_queue)

    def record_prefilling_latency(self,latency:float,threshold:float,cur_time:int=None):
        if cur_time is None:
            cur_time = int(time.time())
        if self.prefilling_latency_recorder.get(cur_time) is None:
            self.prefilling_latency_recorder[cur_time]=[latency]
        else:
            self.prefilling_latency_recorder[cur_time].append(latency)
        
        if self.prefilling_threshold_recorder.get(cur_time) is None:
            self.prefilling_threshold_recorder[cur_time]=[threshold]
        else:
            self.prefilling_threshold_recorder[cur_time].append(threshold)
    
    def record_decoding_latency(self,latency:float,threshold:float,cur_time:int=None):
        if cur_time is None:
            cur_time = int(time.time())
        if self.decoding_latency_recorder.get(cur_time) is None:
            self.decoding_latency_recorder[cur_time]=[latency]
        else:
            self.decoding_latency_recorder[cur_time].append(latency)
    
        if self.decoding_threshold_recorder.get(cur_time) is None:
            self.decoding_threshold_recorder[cur_time]=[threshold]
        else:
            self.decoding_threshold_recorder[cur_time].append(threshold)


    def record_throughput(self,throughput:int):
        cur_time = int(time.time())
        if self.throughput_recorder.get(cur_time) is None:
            self.throughput_recorder[cur_time]=throughput
        else:
            self.throughput_recorder[cur_time]+=throughput
    
    def report_timely_latency(self):
        timely_prefilling_latency=[]
        timely_decoding_latency=[]
        for key in self.prefilling_latency_recorder.keys():
            timestamp = key
            average,p90 = calculate_average_and_p90(self.prefilling_latency_recorder[key])
            average_threshold,p90_threshold = calculate_average_and_p90(self.prefilling_threshold_recorder[key])
            timely_prefilling_latency.append({
                'timestamp':timestamp,
                'average_prefilling_latency':average,
                'P90_prefilling_latency':p90,
                'average_threshold': average_threshold
            })
        for key in self.decoding_latency_recorder.keys():
            timestamp = key
            average,p90 = calculate_average_and_p90(self.decoding_latency_recorder[key])
            average_threshold,p90_threshold = calculate_average_and_p90(self.decoding_threshold_recorder[key])
            timely_decoding_latency.append({
                'timestamp':timestamp,
                'average_decoding_latency':average,
                'P90_decoding_latency':p90,
                'average_threshold': average_threshold
            })

        # with open('/root/hujianmin/qwen2_moe_i/result/timely_prefilling_latency_list.json','w')as f:
        #     f.write(json.dumps(timely_prefilling_latency,indent=" "))

        # with open('/root/hujianmin/qwen2_moe_i/result/timely_decoding_latency_list.json','w')as f:
        #     f.write(json.dumps(timely_decoding_latency,indent=" "))

        with open('/root/hujianmin/qwen2_moe_i/result/timely_prefilling_latency_list_brownout.json','w')as f:
            f.write(json.dumps(timely_prefilling_latency,indent=" "))

        with open('/root/hujianmin/qwen2_moe_i/result/timely_decoding_latency_list_brownout.json','w')as f:
            f.write(json.dumps(timely_decoding_latency,indent=" "))

    def existing_requests(self):
        cnt=0
        for prompt in self.prompts:
            if len(prompt)>0:
                cnt+=1
            if cnt>0:
                return True
        return len(self.waiting_queue)+cnt>0

    
    def get_output_by_promot(self,prompt:str):
        outputs_list = self.output_dict_list.get(prompt)
        if outputs_list is None:
            return None
        result = self.output_dict_list[prompt][0]
        self.output_dict_list[prompt] = self.output_dict_list[prompt][1:]
        return result
    
    def add_output_to_dict(self,prompt:str,output:str):
        outputs_list = self.output_dict_list.get(prompt)
        if outputs_list is None:
            self.output_dict_list[prompt]=[]

        self.output_dict_list[prompt].append(output)

