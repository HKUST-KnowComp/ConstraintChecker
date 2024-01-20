from datasets import load_dataset
import os
import openai
import re
import json
import argparse
import time
import sys

dataset = load_dataset("esnli")
esnli_train = dataset["train"]
esnli_valid = dataset["validation"]
esnli_test = dataset["test"]



def generate_pool_for_esnli(n):
    return_list_esnli_pool = []
    for index in range(0,n):
        temp = {}
        temp["Question"] = 'Premise:\n"' + esnli_train[index]['premise'] + '"\nBased on this premise,can we conclude the hypothesis "' + \
                            esnli_train[index]['hypothesis'] + '" is true?\nOPTIONS:\n- yes\n- no\n- it is not possible to tell\n'
        temp["Rationale"] = esnli_train[index]['explanation_1']
        if(esnli_train[index]['label'] == 0):
            temp["Answer"] = "The answer is yes."
            temp["Ground_truth"] = 0
        elif(esnli_train[index]['label'] == 1):
            temp["Answer"] = "The answer is it is not possible to tell."
            temp["Ground_truth"] = 1
        elif(esnli_train[index]['label'] == 2):
            temp["Answer"] = "The answer is no."
            temp["Ground_truth"] = 2
        return_list_esnli_pool.append(temp)
    return return_list_esnli_pool


def generate_test_for_esnli():
    return_list_esnli_test = []
    for index in range(0,len(esnli_test)):
        temp = {}
        temp["Question"] = 'Premise:\n"' + esnli_test[index]['premise'] + '"\nBased on this premise,can we conclude the hypothesis "' + \
                            esnli_test[index]['hypothesis'] + '" is true?\nOPTIONS:\n- yes\n- no\n- it is not possible to tell\n'
        temp["Rationale"] = esnli_test[index]['explanation_1']
        if(esnli_test[index]['label'] == 0):
            temp["Answer"] = "The answer is yes."
            temp["Ground_truth"] = 0
        elif(esnli_test[index]['label'] == 1):
            temp["Answer"] = "The answer is it is not possible to tell."
            temp["Ground_truth"] = 1
        elif(esnli_test[index]['label'] == 2):
            temp["Answer"] = "The answer is no."
            temp["Ground_truth"] = 2
        return_list_esnli_test.append(temp)
    return return_list_esnli_test



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["text-davinci-002","code-davinci-002"],help="whcih OpenAI model to chose",default="code-davinci-002")
    args = parser.parse_args()


    # Construct Pool
    return_list_esnli_pool = generate_pool_for_esnli(100)
    with open("esnli_pool_{}.json".format(len(return_list_esnli_pool)),"w") as f:
        json.dump(return_list_esnli_pool,f,indent=2)   
    # Construct Train set
    

    # Construct Valid Set

    # Construct Test Set
    return_list_esnli_test = generate_test_for_esnli()
    with open("esnli_test_{}.json".format(len(return_list_esnli_test)),"w") as f:
        json.dump(return_list_esnli_test,f,indent=2)    