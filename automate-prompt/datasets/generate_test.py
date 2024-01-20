from datasets import load_dataset
import os
import re
import json



dataset = load_dataset("gsm8k","main")
gsm8k_test = dataset["test"]


def generate_test():
    return_list = []
    for index in range(0,len(gsm8k_test["answer"])):
        
        temp = {}
        ground_truth_match = re.search('####',gsm8k_test["answer"][index])
 
        temp['Question'] = gsm8k_test["question"][index]
        temp["Rationale"] = "N/A"
        temp["Answer"] = "N/A"

        temp["Ground_truth"] = gsm8k_test["answer"][index][ground_truth_match.span()[0]+5:]
        return_list.append(temp)
        print("Finished process {}".format(index))
    return return_list    


def main():
    return_list = generate_test()
    print(len(return_list))
    with open("gsm8k_test_1319.json","w") as f:
        json.dump(return_list,f,indent=2) 

  
main()