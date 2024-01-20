from datasets import load_dataset
import os
import openai
import re
import json
import argparse
import time
import sys

dataset = load_dataset("sst2")
sst2_train = dataset["train"]
sst2_valid = dataset["validation"]
# sst2_test = dataset["test"] # No available Label

prompt = 'What is the sentiment of the following sentence?\n"that loves its characters and communicates something rather beautiful about human nature"\nA: "loves its characters" indicates positive sentiment. The answer is positive.\n\nWhat is the sentiment of the following sentence?\n"hide new secretions from the parental units"\nA: If people are hiding something, it means the sentiment is on the negative side. The answer is negative.\n\nWhat is the sentiment of the following sentence?\n"the greatest musicians"\nA: By saying someone being the "greatest", it means positive sentiment. The answer is positive.\n\nWhat is the sentiment of the following sentence?\n"contains no wit , only labored gags"\nA: "contains no wit" is clearly a negative sentiment. The answer is negative.\n\nWhat is the sentiment of the following sentence?\n"demonstrates that the director of such hollywood blockbusters as patriot games can still turn out a small , personal ﬁlm with an emotional wallop ."\nA: "can still turn out a small , personal ﬁlm with an emotional wallop ." indicates sentiment on the positive side. The answer is positive.\n\nWhat is the sentiment of the following sentence?\n"that ’s far too tragic to merit such superﬁcial treatment"\nA: "far too tragic" and "to merit such superﬁcial treatment" both mean negative sentiments. The answer is negative.\n\n'
code_api_list = ["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",
                "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",
                "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",
                "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",
                "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",
                "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",
                "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",
                "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",
                "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",
                "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",
                "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"
]

def complete_gpt3(input, count, model = "code-davinci-002",temperature = 0, max_tokens = 256):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            openai.api_key = code_api_list[count % len(code_api_list)]
            response = openai.Completion.create(
                model=model,
                prompt=input,
                max_tokens=max_tokens,
                temperature = temperature,                
            )
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:  # something is wrong: e.g. prompt too long
                print(
                    f"InvalidRequestError\nPrompt passed in:\n\n{input}\n\n")
                assert False
            print("API error:", error)
            count += 1
            time.sleep(5)
    return response

def generate_pool_for_sst2(n):
    count = 0
    return_list_sst_pool = []
    for index in range(0,n):
        input_to_gpt3 = prompt + 'What is the sentiment of the following sentence?\n"' + sst2_train[index]['sentence'] + '"' + "\nA:"

        response = complete_gpt3(input=input_to_gpt3, count=count, model="code-davinci-002")             
        generated_full = response["choices"][0]["text"]
        pattern = "The answer is (positive|negative)."
        generated_match = re.search(pattern,generated_full)
     
        temp = {}
        temp["Question"] = 'What is the sentiment of the following sentence?\n"'+ sst2_train[index]['sentence'] + '"' 
        if( generated_match != None):
            temp["Rationale"] = generated_full[0:generated_match.span()[0]-1]
            temp["Answer"] = generated_match.group(0)
        else:
            continue

        if(sst2_train[index]['label'] == 0):
            temp["Ground_truth"] = 0
        elif(sst2_train[index]['label'] == 1):
            temp["Ground_truth"] = 1   


        return_list_sst_pool.append(temp)
        count = count + 1
        print("sst2 pool generation : {}".format(index))
    return return_list_sst_pool




def generate_test_for_sst2():
    return_list_sst2_test = []
    for index in range(0,len(sst2_valid)):
        temp = {}
        temp["Question"] = 'What is the sentiment of the following sentence?\n"' + sst2_valid[index]['sentence'] + '"'
        temp["Rationale"] = "N/A"
        
        if(sst2_valid[index]['label'] == 0):
            temp["Answer"] = "The answer is negative."
            temp["Ground_truth"] = 0
        elif(sst2_valid[index]['label'] == 1):
            temp["Answer"] = "The answer is positive."
            temp["Ground_truth"] = 1

        return_list_sst2_test.append(temp)
    return return_list_sst2_test



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["text-davinci-002","code-davinci-002"],help="whcih OpenAI model to chose",default="code-davinci-002")
    args = parser.parse_args()


    # Construct Pool
    return_list_sst2_pool = generate_pool_for_sst2(50)
    with open("sst2_pool_{}.json".format(len(return_list_sst2_pool)),"w") as f:
        json.dump(return_list_sst2_pool,f,indent=2)   
    # # Construct Train set
    

    # # Construct Valid Set

    # # Construct Test Set
    # return_list_sst2_test = generate_test_for_sst2()
    # with open("sst2_test_{}.json".format(len(return_list_sst2_test)),"w") as f:
    #     json.dump(return_list_sst2_test,f,indent=2)    