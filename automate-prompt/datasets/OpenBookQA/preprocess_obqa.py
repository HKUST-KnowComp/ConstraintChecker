from datasets import load_dataset
import os
import openai
import re
import json
import argparse
import time
import sys

dataset = load_dataset("openbookqa","main")
obqa_train = dataset["train"]
obqa_valid = dataset["validation"]
obqa_test = dataset["test"]
prompt = "Q: Poison causes harm to which of the following? (a) a Tree (b) a robot (c) a house (d) a car\nA: Poison will harm living things, only a tree is a living thing. The answer is (a).\n\nQ: As you look deeper into a Marbel you can see (a) the future (b) minut defects (c) colors (d) the other side\nA: Marbel is not transparent, so you can not see the other side. Marbel does not necessarily have multiple colors. You will see minut defects. The answer is (b).\n\nQ: When food is reduced in the stomach (a) the mind needs time to digest (b) take a second to digest what I said (c) nutrients are being deconstructed (d) reader’s digest is a body of works\nA: The food is being deconstructed in the stomach during digestion. The answer is (c).\n\nQ: The sun is responsible for (a) puppies learning new tricks (b) children growing up and getting old (c) ﬂowers wilting in a vase (d) plants sprouting, blooming and wilting\nA: The sun can affect the growing of living things, like plants. The answer is (d).\n\n"
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

def generate_pool_for_obqa(n):
    count = 0
    return_list_obqa_pool = []
    for index in range(0,n):
        input_to_gpt3 = prompt + 'Q: ' + obqa_train[index]["question_stem"] + ' (a) ' + obqa_train[index]["choices"]["text"][0] + ' (b) ' + obqa_train[index]["choices"]["text"][1] + ' (c) ' + obqa_train[index]["choices"]["text"][2] +\
                            ' (d) ' + obqa_train[index]["choices"]["text"][3] + "\nA:"

        response = complete_gpt3(input=input_to_gpt3, count=count, model="code-davinci-002")             
        generated_full = response["choices"][0]["text"]
        pattern = "The answer is \([a-z|A-Z]\)."
        generated_match = re.search(pattern,generated_full)
     
        temp = {}
        temp["Question"] = 'Q: ' + obqa_train[index]["question_stem"] + ' (a) ' + obqa_train[index]["choices"]["text"][0] + ' (b) ' + obqa_train[index]["choices"]["text"][1] + ' (c) ' + obqa_train[index]["choices"]["text"][2] +\
                            ' (d) ' + obqa_train[index]["choices"]["text"][3]
        if( generated_match != None):
            temp["Rationale"] = generated_full[0:generated_match.span()[0]-1]
            temp["Answer"] = generated_match.group(0)
        else:
            continue
        
        temp["Ground_truth"] = obqa_train[index]["answerKey"]
        return_list_obqa_pool.append(temp)
        count = count + 1
        print("OpenBookQA pool generation : {}".format(index))
    return return_list_obqa_pool


def generate_test_for_obqa():
    return_list_obqa_test = []
    for index in range(0,len(obqa_test)):
        temp = {}
        temp["Question"] = 'Q: ' + obqa_test[index]["question_stem"] + ' (a) ' + obqa_test[index]["choices"]["text"][0] + ' (b) ' + obqa_test[index]["choices"]["text"][1] + ' (c) ' + obqa_test[index]["choices"]["text"][2] +\
                            ' (d) ' + obqa_test[index]["choices"]["text"][3]
        temp["Rationale"] = "N/A"
        
        temp["Answer"] = "The answer is (" + obqa_test[index]["answerKey"].lower() + ")"
        temp["Ground_truth"] = obqa_test[index]["answerKey"]


        return_list_obqa_test.append(temp)
    return return_list_obqa_test



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["text-davinci-002","code-davinci-002"],help="whcih OpenAI model to chose",default="code-davinci-002")
    args = parser.parse_args()


    # Construct Pool
    return_list_obqa_pool = generate_pool_for_obqa(50)
    with open("obqa_pool_{}.json".format(len(return_list_obqa_pool)),"w") as f:
        json.dump(return_list_obqa_pool,f,indent=2)   
    # # Construct Train set
    

    # # Construct Valid Set

    # # Construct Test Set
    # return_list_obqa_test = generate_test_for_obqa()
    # with open("obqa_test_{}.json".format(len(return_list_obqa_test)),"w") as f:
    #     json.dump(return_list_obqa_test,f,indent=2)    