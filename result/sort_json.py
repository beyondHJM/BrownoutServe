import json

file=  './unfused_throughput_alpaca.json'

with open(file)as f:
    data = json.load(f)

# 按照 age 然后按照 score 排序
sorted_data = sorted(data, key=lambda x: (x['threshold'], x['rps']))


# print(json.dumps(data,indent=" "))
with open(file,'w')as f:
    f.write(json.dumps(sorted_data,indent=" "))
