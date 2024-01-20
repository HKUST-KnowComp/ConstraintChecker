# This is the file for generate rationale pool for CommonsenseQA
import jsonlines
import openai
import re
import time
import sys
import json
import argparse

code_api_list = ["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",
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
# The human written 7 COT examplars 
prompt = "Q: What do people use to absorb extra ink from a fountain pen?\nAnswer Choices:\n(a) shirt pocket\n(b) calligrapher's hand\n(c) inkwell\n(d) desk drawer\n(e) blotter\nA: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. So the answer is (e).\n\nQ: What home entertainment equipment requires cable?\nAnswer Choices:\n(a) radio shack\n(b) substation\n(c) television\n(d) cabinet\nA: The answer must require cable. Of the above choices, only television requires cable. So the answer is (c).\n\nQ: The fox walked from the city into the forest, what was it looking for?\nAnswer Choices:\n(a) pretty flowers\n(b) hen house\n(c) natural habitat\n(d) storybook\nA: Answer: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. So the answer is (b).\n\nQ: Sammy wanted to go to where the people were. Where might he go?\nAnswer Choices:\n(a) populated areas\n(b) race track\n(c) desert\n(d) apartment\n(e) roadblock\nA: The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. So the answer is (a).\n\nQ: Where do you put your grapes just before checking out?\nAnswer Choices:\n(a) mouth\n(b) grocery cart\n(c)super market\n(d) fruit basket\n(e) fruit market\nA: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. So the answer is (b).\n\nQ: Google Maps and other highway and street GPS services have replaced what?\nAnswer Choices:\n(a) united states\n(b) mexico\n(c) countryside\n(d) atlas\nA: The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. So the answer is (d).\n\nQ: Before getting a divorce, what did the wife feel who was doing all the work?\nAnswer Choices:\n(a) harder\n(b) anguish\n(c) bitterness\n(d) tears\n(e) sadness\nA: The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. So the answer is (c).\n\n"

csqa_dataset = []
csqa_valset = []
with open("/Users/kashun/Documents/Research/2022 Fall/Chain of Thought/Experiment/example-prompt/datasets/CSQA/raw/train_rand_split.jsonl","r+",encoding="utf-8") as f:
    for item in jsonlines.Reader(f):
        csqa_dataset.append(item)

with open("/Users/kashun/Documents/Research/2022 Fall/Chain of Thought/Experiment/example-prompt/datasets/CSQA/raw/dev_rand_split.jsonl","r+",encoding="utf-8") as f:
    for item in jsonlines.Reader(f):
        csqa_valset.append(item)

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

def generate_pool_for_csqa(args, n, zero_shot = False):
    count = 0
    return_list_csqa_pool = []

    for index in range(0,n):
        if(zero_shot == True):
            first_stage_input = 'Q: ' + csqa_dataset[index]["question"]["stem"] + '\nAnswer Choices:\n(a) ' +  csqa_dataset[index]["question"]["choices"][0]["text"] + "\n(b) " + \
                        csqa_dataset[index]["question"]["choices"][1]["text"] + "\n(c) " + csqa_dataset[index]["question"]["choices"][2]["text"] + "\n(d) " + csqa_dataset[index]["question"]["choices"][3]["text"]  +\
                        "\n(e) " + csqa_dataset[index]["question"]["choices"][4]["text"] + "\nAnswer: Let's think step by step."
            response = complete_gpt3(input = first_stage_input, count = count, model = args.model, max_tokens=128)
            rationale = response["choices"][0]["text"]
            # Combine rationale to generate again
            count += 1
            second_stage_input = first_stage_input + rationale + " So the answer is"
            response = complete_gpt3(input = second_stage_input, count = count, model = args.model, max_tokens=32)
            pred = response["choices"][0]["text"]
            # Answer cleaning
            pred = re.findall(r'(a|b|c|d|e)', pred)
            if len(pred) == 0:
                pred = ""
            else:
                pred = pred[0]
            if pred != "":
                if pred[-1] == ".":
                    pred = pred[:-1]
                temp = {}
                temp["Question"] = csqa_dataset[index]["question"]["stem"] + '\nAnswer Choices:\n(a) ' +  csqa_dataset[index]["question"]["choices"][0]["text"] + "\n(b) " + \
                                csqa_dataset[index]["question"]["choices"][1]["text"] + "\n(c) " + csqa_dataset[index]["question"]["choices"][2]["text"] + "\n(d) " + csqa_dataset[index]["question"]["choices"][3]["text"]  +\
                                "\n(e) " + csqa_dataset[index]["question"]["choices"][4]["text"] 
                temp["Rationale"] = rationale.replace("\n","")
                temp["Answer"] = "So the answer is (" + pred + ")."
                temp["Ground_truth"] = csqa_dataset[index]["answerKey"] # Note this is in uppercase
                return_list_csqa_pool.append(temp)
        # input_to_gpt3 = prompt + 'Q: ' + csqa_dataset[index]["question"]["stem"] + '\nAnswer Choices:\n(a) ' +  csqa_dataset[index]["question"]["choices"][0]["text"] + "\n(b) " + \
        #                 csqa_dataset[index]["question"]["choices"][1]["text"] + "\n(c) " + csqa_dataset[index]["question"]["choices"][2]["text"] + "\n(d) " + csqa_dataset[index]["question"]["choices"][3]["text"]  +\
        #                 "\n(e) " + csqa_dataset[index]["question"]["choices"][4]["text"] + "\nA:"

        # received = False
        # while not received:
        #     try:
        #         openai.api_key = code_api_list[count % len(code_api_list)]

        #         result = openai.Completion.create(
        #         model="code-davinci-002",
        #         prompt=input_to_gpt3,
        #         max_tokens=256,
        #         temperature = 0
        #         )
        #         received = True
        #     except:
        #         error = sys.exc_info()[0]

        #         print("API error:", error)
        #         count += 1
        #         time.sleep(5)               
        # # returned full sentance (rationale + answer)
        # generated_full = result["choices"][0]["text"]

        # pattern = "So the answer is \([a-z|A-Z]\)."
        # generated_match = re.search(pattern,generated_full)

        # ground_truth = csqa_dataset[index]["answerKey"] # Note this is in uppercase
        
        # temp = {}
        # temp['Question'] = csqa_dataset[index]["question"]["stem"] + '\nAnswer Choices:\n(a) ' +  csqa_dataset[index]["question"]["choices"][0]["text"] + "\n(b) " + \
        #                 csqa_dataset[index]["question"]["choices"][1]["text"] + "\n(c) " + csqa_dataset[index]["question"]["choices"][2]["text"] + "\n(d) " + csqa_dataset[index]["question"]["choices"][3]["text"]  +\
        #                 "\n(e) " + csqa_dataset[index]["question"]["choices"][4]["text"] 
        # if( generated_match != None):
        #     temp["Rationale"] = generated_full[0:generated_match.span()[0]-1]
        #     temp["Answer"] = generated_match.group(0)
        # else:
        #     temp["Rationale"] = generated_full
        #     temp["Answer"] = "N/A"
        # temp["Ground_truth"] = ground_truth
        # return_list.append(temp)
        # print("Finished process {}".format(index))

        # count = count + 1
        
    return return_list_csqa_pool   


def calculate_accuracy():
    with open('./csqa_pool_zeroshot.json', 'r') as f:
        data = json.load(f)
    match_count = 0
    length = len(data)
    for i in range (0,length):
        if (data[i]["Answer"][-3:-2].lower() == data[i]["Ground_truth"].lower()):
            match_count = match_count + 1
    return match_count, length

def select_correct_exemplars():
    correct = []
    with open('./text-davinci-002/csqa_text_pool_zeroshot.json','r') as f:
        data1 = json.load(f)  
    count = 0
    for i in range(0,len(data1)):
        if (data1[i]["Answer"][-3:-2].lower()== data1[i]["Ground_truth"].lower()):
            count+= 1
            correct.append(data1[i])
    return correct


# Preprocess to generate the train set
# The train set is from index 200-300 of the raw train split
TRAIN_BEGIN_INDEX = 200
def generate_train(n):
    return_list = []
    for index in range(TRAIN_BEGIN_INDEX,TRAIN_BEGIN_INDEX + n):
        temp = {}

        temp['Question'] = csqa_dataset[index]["question"]["stem"] + '\nAnswer Choices:\n(a) ' +  csqa_dataset[index]["question"]["choices"][0]["text"] + "\n(b) " + \
                        csqa_dataset[index]["question"]["choices"][1]["text"] + "\n(c) " + csqa_dataset[index]["question"]["choices"][2]["text"] + "\n(d) " + csqa_dataset[index]["question"]["choices"][3]["text"]  +\
                        "\n(e) " + csqa_dataset[index]["question"]["choices"][4]["text"] 
        temp["Rationale"] = "N/A"
        temp["Answer"] = "N/A"

        temp["Ground_truth"] = csqa_dataset[index]["answerKey"]
        return_list.append(temp)
        print("Finished process {}".format(index))
    return return_list    

VAL_BEGIN_INDEX = 300
def generate_val(n):
    return_list = []
    for index in range(VAL_BEGIN_INDEX,VAL_BEGIN_INDEX + n):
        temp = {}

        temp['Question'] = csqa_dataset[index]["question"]["stem"] + '\nAnswer Choices:\n(a) ' +  csqa_dataset[index]["question"]["choices"][0]["text"] + "\n(b) " + \
                        csqa_dataset[index]["question"]["choices"][1]["text"] + "\n(c) " + csqa_dataset[index]["question"]["choices"][2]["text"] + "\n(d) " + csqa_dataset[index]["question"]["choices"][3]["text"]  +\
                        "\n(e) " + csqa_dataset[index]["question"]["choices"][4]["text"] 
        temp["Rationale"] = "N/A"
        temp["Answer"] = "N/A"

        temp["Ground_truth"] = csqa_dataset[index]["answerKey"]
        return_list.append(temp)
        print("Finished process {}".format(index))
    return return_list  

def generate_test():
    return_list = []
    for index in range(0,len(csqa_valset)):
        temp = {}

        temp['Question'] = csqa_valset[index]["question"]["stem"] + '\nAnswer Choices:\n(a) ' +  csqa_valset[index]["question"]["choices"][0]["text"] + "\n(b) " + \
                        csqa_valset[index]["question"]["choices"][1]["text"] + "\n(c) " + csqa_valset[index]["question"]["choices"][2]["text"] + "\n(d) " + csqa_valset[index]["question"]["choices"][3]["text"]  +\
                        "\n(e) " + csqa_valset[index]["question"]["choices"][4]["text"] 
        temp["Rationale"] = "N/A"
        temp["Answer"] = "N/A"

        temp["Ground_truth"] = csqa_valset[index]["answerKey"]
        return_list.append(temp)
    assert len(return_list) == 1221   
    return return_list  
# def main():
    # return_list = generate_rationale(200)
    # with open("csqa_pool_1.json","w") as f:
    #     json.dump(return_list,f,indent=2) 
    # match_count, length = calculate_accuracy()
    # print(match_count)
    # print(match_count/length) 
    # return_list = select_correct_rationale()
    # return_list = return_list[0:100]
    # with open("csqa_code_pool_100.json","w") as f:
    #     json.dump(return_list,f,indent=2)    

    # For generate train set
    # return_train_list = generate_train(100)
    # with open("csqa_train_100.json","w") as f:
    #     json.dump(return_train_list,f,indent=2)    
    # # For generate valid set
    # return_val_list = generate_val(100)
    # with open("csqa_val_100.json","w") as f:
    #     json.dump(return_val_list,f,indent=2)
    # For generate test set
    # return_list = generate_pool_for_csqa()
    # with open("csqa_test_1221.json","w") as f:
    #     json.dump(return_list,f,indent=2)      

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", action="store_true",help="Preprocess for csqa pool")
    parser.add_argument("--train", action="store_true",help="Preprocess for csqa train set")
    parser.add_argument("--valid", action="store_true",help="Preprocess for csqa validation set")

    parser.add_argument("--zero", action="store_true",help="Use Zero-shot method to generate pool")
    parser.add_argument("--pool_size", type=int, default=100)
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--val_size", type=int, default=100)

    parser.add_argument("--model", type=str, choices=["text-davinci-002","code-davinci-002"],help="whcih OpenAI model to chose",default="text-davinci-002")
    args = parser.parse_args()

    # return_list_gsm8k_pool = generate_pool_for_csqa(args,int(args.pool_size * 2),zero_shot=True)
    # with open("csqa_pool_zeroshot","w") as f:
    #     json.dump(return_list_gsm8k_pool,f,indent=2)
    if(args.pool == True):
        if(args.zero == True):
            # return_list_gsm8k_pool = generate_pool_for_gsm8k(args,int(args.pool_size * 2.5),zero_shot=True)
            # print(len(return_list_gsm8k_pool))
            # filter the correct ones
            return_list_csqa_pool_correct = select_correct_exemplars()
            print("Total there are {} correct examplars".format(len(return_list_csqa_pool_correct)))
            if(len(return_list_csqa_pool_correct) >= args.pool_size):
                return_list_csqa_pool_correct = return_list_csqa_pool_correct[0:args.pool_size]
                print("choose the frist {} examplars".format(args.pool_size))
                with open("./{}/csqa_pool_{}_zeroshot.json".format(args.model,args.pool_size),"w") as f:
                    json.dump(return_list_csqa_pool_correct,f,indent=2)
            else:
                print("Not enough correct examplars, please increase the generation size")
                with open("./{}/csqa_pool_{}_zeroshot.json".format(args.model,len(return_list_csqa_pool_correct)),"w") as f:
                    json.dump(return_list_csqa_pool_correct,f,indent=2)
        # else:
        #     return_list_gsm8k_pool = generate_pool_for_gsm8k(args.pool_size * 2)
        #     # filter the correct ones
        #     return_list_gsm8k_pool_correct = select_correct_examplars(return_list_gsm8k_pool)
        #     print("Total there are {} correct examplars".format(len(return_list_gsm8k_pool_correct)))
        #     if(len(return_list_gsm8k_pool_correct) >= args.pool_size):
        #         return_list_gsm8k_pool_correct = return_list_gsm8k_pool_correct[0:args.pool_size]
        #         print("choose the frist {} examplars".format(args.pool_size))
        #         with open("./gsm8k_/{}/gsm8k_pool_{}.json".format(args.model,args.pool_size),"w") as f:
        #             json.dump(return_list_gsm8k_pool_correct,f,indent=2) 
        #     else:
        #         print("Not enough correct examplars, please increase the generation size")
   