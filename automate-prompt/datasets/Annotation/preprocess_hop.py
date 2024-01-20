from datasets import load_dataset
import json
import re
from collections import Counter

# with open('./gsm8k_test.json', 'r') as f:
#     data = json.load(f)
dataset = load_dataset("gsm8k","main")
gsm8k_train = dataset["train"]
gsm8k_test = dataset["test"]

return_list = []

complex_list = []
for i in range(0,100):
    ground_truth_match = re.search('####',gsm8k_train[i]["answer"])
    
    
    temp = {}
    temp['Question'] = gsm8k_train[i]["question"]

    count = Counter(gsm8k_train[i]["answer"])
    temp["hop"] = str(count["\n"])
    temp["Answer"] = gsm8k_train[i]["answer"]
    temp["Ground_truth"] = gsm8k_train[i]["answer"][ground_truth_match.span()[0]+5:]
    
    # if (count["\n"] == 9):
    complex = {}
    complex['Question'] = gsm8k_train[i]["question"]

    complex["Rationale"] =  " " + gsm8k_train[i]["answer"][0:ground_truth_match.span()[0]-1] #+ "."
    complex["Answer"] = "The answer is " + gsm8k_train[i]["answer"][ground_truth_match.span()[0]+5:]  + "."
    complex["Ground_truth"] = gsm8k_train[i]["answer"][ground_truth_match.span()[0]+5:]        
    complex_list.append(complex)
    return_list.append(temp)
with open("gsm8k_pool_original_100.json","w") as f:
    json.dump(complex_list,f,indent=2) 

# with open("gsm8k_hop_train.json","w") as f:
#     json.dump(return_list,f,indent=2) 
# with open('./gsm8k_hop_train.json', 'r') as f:
#     data = json.load(f)

# list_count_hop = []
# for i in range(0,len(data)):
#     list_count_hop.append(data[i]["hop"])


    
# count = Counter(list_count_hop)
# print(count)
# f = open("./gsm_stream_predictions", "r")



# with open("./gsm_stream_predictions","r") as f:
#     lines = f.readlines()
# hop_total = {'3': 370, '2': 326, '4': 298, '5': 174, '6': 88, '7': 40, '8': 20, '9': 2, '11': 1}
# hop_count = {'3': 0, '2': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '11': 0}
# for i in range(0,len(lines)):

#     pattern = "The answer is \d{1,}\."
#     sttr = re.search(pattern, lines[i].replace("$","").replace(",","").replace("%",""))

#     if(sttr is not None):
#         if(sttr.group(0)[14:-1] == data[i]["Ground_truth"]):
#             hop_count[data[i]["hop"]] += 1

# print(hop_count)
# for key in hop_total:
#     print("hop : {}, acc is {}".format(key,hop_count[key]/hop_total[key]))

# with open("./test_for_hop.json","r") as f:
#     dataset = json.load(f)

# match_count = 0
# for i in range(0,len(dataset)):
#     pattern = "The answer is \d{1,}\."
#     sttr = re.search(pattern, dataset[i]["Answer"].replace("$","").replace(",","").replace("%",""))  
#     if (sttr is not None):
#         #check if match the ground truth
#         if(sttr.group(0)[14:-1] == dataset[i]["Ground_truth"]):
#             hop_count[data[i]["hop"]] += 1
#             match_count += 1

# print(hop_count)
# for key in hop_total:
#     print("hop : {}, acc is {}".format(key,hop_count[key]/hop_total[key]))