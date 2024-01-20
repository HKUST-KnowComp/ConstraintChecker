from datasets import load_dataset
import os
import openai
import re
import json
import argparse
import time
import sys

dataset = load_dataset("gsm8k","main")
gsm8k_train = dataset["train"]
gsm8k_test = dataset["test"]

# The human written 8 COT examplars 
prompt = "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.\n\n"
# The human written 8 COT examplars with "Let's think step by step." + QA -> Question, Answer
#prompt = "Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nAnswer: Let's think step by step. We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.\n\nQuestion: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nAnswer: Let's think step by step. There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.\n\nQuestion: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nAnswer: Let's think step by step. Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.\n\nQuestion: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nAnswer: Let's think step by step. Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.\n\nQuestion: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nAnswer: Let's think step by step. He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.\n\nQuestion: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nAnswer: Let's think step by step. There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.\n\nQuestion: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nAnswer: Let's think step by step. Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.\n\nQuestion: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nAnswer: Let's think step by step. She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.\n\n"
# The human written 8 COT examplars with  QA -> Question, Answer
#prompt = "Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nAnswer: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.\n\nQuestion: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nAnswer: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.\n\nQuestion: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nAnswer: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.\n\nQuestion: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nAnswer: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.\n\nQuestion: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nAnswer: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.\n\nQuestion: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nAnswer: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.\n\nQuestion: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nAnswer: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.\n\nQuestion: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nAnswer: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.\n\n"

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
def generate_pool_for_gsm8k(args, n, zero_shot = False):
    count = 0
    return_list_gsm8k_pool = []
    for index in range(0+500,n+500):
        # Did not use Manual-COT
        if(zero_shot == True):
            first_stage_input = 'Q: ' + gsm8k_train["question"][index] + "\nAnswer: Let's think step by step."
            response = complete_gpt3(input = first_stage_input, count = count, model = args.model, max_tokens=128)
            rationale = response["choices"][0]["text"]
            # Combine rationale to generate again
            count += 1
            second_stage_input = first_stage_input + rationale + " The answer is"
            response = complete_gpt3(input = second_stage_input, count = count, model = args.model, max_tokens=32)
            pred = response["choices"][0]["text"]
            # Answer cleaning
            pred = pred.replace(",", "")
            pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
            if len(pred) == 0:
                pred = ""
            else:
                pred = pred[0]
            ground_truth_match = re.search('####',gsm8k_train["answer"][index])
            if pred != "":
                if pred[-1] == ".":
                    pred = pred[:-1]
                    temp = {}
                    temp["Question"] = gsm8k_train["question"][index]
                    temp["Rationale"] = rationale.replace("\n","")
                    temp["Answer"] = "The answer is " + pred + "."
                    temp["Ground_truth"] = gsm8k_train["answer"][index][ground_truth_match.span()[0]+5:]
                    return_list_gsm8k_pool.append(temp)
        # Use Manual-CoT to generate the pool
        else:
            input_to_gpt3 = prompt + 'Q: ' + gsm8k_train["question"][index] + "\nA:"
            response = complete_gpt3(input=input_to_gpt3, count=count, model=args.model)             
            # returned full sentance (rationale + answer)
            generated_full = response["choices"][0]["text"]
            pattern = "The answer is \d{1,}\."
            generated_match = re.search(pattern,generated_full)
            #generated_match = re.search(pattern,generated_full.replace("$","").replace(",","").replace("%",""))
            ground_truth_match = re.search('####',gsm8k_train["answer"][index])
            
            temp = {}
            temp['Question'] = gsm8k_train["question"][index]
            if( generated_match != None):
                temp["Rationale"] = generated_full[0:generated_match.span()[0]-1]
                temp["Answer"] = generated_match.group(0)
            else:
                temp["Rationale"] = generated_full
                temp["Answer"] = "N/A"
            
            temp["Ground_truth"] = gsm8k_train["answer"][index][ground_truth_match.span()[0]+5:]
            return_list_gsm8k_pool.append(temp)
            count = count + 1
        print("gsm8k pool generation : {}".format(index))
    return return_list_gsm8k_pool       


def calculate_accuracy():
    with open('./pool_300.json', 'r') as f:
        data = json.load(f)
    match_count = 0
    length = len(data)
    for i in range (0,length):
        if (data[i]["Answer"][14:-1] == data[i]["Ground_truth"]):
            match_count = match_count + 1
    return match_count, length

def select_correct_examplars(input_pool):
    correct = []
    count = 0
    for i in range(0,len(input_pool)):
        if (input_pool[i]["Answer"][14:-1] == input_pool[i]["Ground_truth"]):
            count+= 1
            correct.append(input_pool[i])
    return correct

# USE index [400,500] for train and index [7000,7100] for validation
TRAIN_BEGIN_INDEX = 600
VAL_BEGIN_INDEX = 7000

def generate_train_for_gsm8k(n ,zero_shot = False):
    count = 0
    return_list_gsm8k_train = []
    for index in range(TRAIN_BEGIN_INDEX,TRAIN_BEGIN_INDEX + n):
        if(zero_shot == True):
            first_stage_input = 'Q: ' + gsm8k_train["question"][index] + "\nAnswer: Let's think step by step."
            response = complete_gpt3(input = first_stage_input, count = count,max_tokens=128)
            rationale = response["choices"][0]["text"]
            # Combine rationale to generate again
            count += 1
            second_stage_input = first_stage_input + rationale + " The answer is"
            response = complete_gpt3(input = second_stage_input, count = count,max_tokens=32)
            pred = response["choices"][0]["text"]
            # Answer cleaning
            pred = pred.replace(",", "")
            pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
            if len(pred) == 0:
                pred = ""
            else:
                pred = pred[0]
            if pred != "":
                if pred[-1] == ".":
                    pred = pred[:-1]
                    temp = {}
                    temp["Question"] = gsm8k_train["question"][index]
                    temp["Rationale"] = "N/A"
                    temp["Answer"] = "N/A"
                    temp["Ground_truth"] = pred
                    return_list_gsm8k_train.append(temp)
        else:
            temp = {}
            ground_truth_match = re.search('####',gsm8k_train["answer"][index])

            temp['Question'] = gsm8k_train["question"][index]
            temp["Rationale"] = "N/A"
            temp["Answer"] = gsm8k_train["answer"][index]
            temp["Ground_truth"] = gsm8k_train["answer"][index][ground_truth_match.span()[0]+5:]
            return_list_gsm8k_train.append(temp)
        print("gsm8k train generation : {}".format(index))
    return return_list_gsm8k_train 

def generate_val_for_gsm8k(n, zero_shot = False):
    count = 0
    return_list_gsm8k_val = []
    for index in range(VAL_BEGIN_INDEX,VAL_BEGIN_INDEX + n):
        if(zero_shot == True):
            first_stage_input = 'Q: ' + gsm8k_train["question"][index] + "\nAnswer: Let's think step by step."
            response = complete_gpt3(input = first_stage_input, count = count,max_tokens=128)
            rationale = response["choices"][0]["text"]
            # Combine rationale to generate again
            count += 1
            second_stage_input = first_stage_input + rationale + " The answer is"
            response = complete_gpt3(input = second_stage_input, count = count,max_tokens=32)
            pred = response["choices"][0]["text"]
            # Answer cleaning
            pred = pred.replace(",", "")
            pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
            if len(pred) == 0:
                pred = ""
            else:
                pred = pred[0]
            if pred != "":
                if pred[-1] == ".":
                    pred = pred[:-1]
                    temp = {}
                    temp["Question"] = gsm8k_train["question"][index]
                    temp["Rationale"] = "N/A"
                    temp["Answer"] = "N/A"
                    temp["Ground_truth"] = pred
                    return_list_gsm8k_val.append(temp)
        else:
            temp = {}
            ground_truth_match = re.search('####',gsm8k_train["answer"][index])

            temp['Question'] = gsm8k_train["question"][index]
            temp["Rationale"] = "N/A"
            temp["Answer"] = gsm8k_train["answer"][index]
            temp["Ground_truth"] = gsm8k_train["answer"][index][ground_truth_match.span()[0]+5:]
            return_list_gsm8k_val.append(temp)
        print("gsm8k validation generation : {}".format(index))
    return return_list_gsm8k_val 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", action="store_true",help="Preprocess for gsm8k pool")
    parser.add_argument("--train", action="store_true",help="Preprocess for gsm8k train set")
    parser.add_argument("--valid", action="store_true",help="Preprocess for gsm8k validation set")

    parser.add_argument("--zero", action="store_true",help="Use Zero-shot method to generate pool")
    parser.add_argument("--pool_size", type=int, default=100)
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--val_size", type=int, default=100)

    parser.add_argument("--model", type=str, choices=["text-davinci-002","code-davinci-002"],help="whcih OpenAI model to chose",default="code-davinci-002")
    args = parser.parse_args()

    if(args.pool == True):
        if(args.zero == True):
            return_list_gsm8k_pool = generate_pool_for_gsm8k(args,int(args.pool_size * 2.5),zero_shot=True)
            print(len(return_list_gsm8k_pool))
            # filter the correct ones
            return_list_gsm8k_pool_correct = select_correct_examplars(return_list_gsm8k_pool)
            print("Total there are {} correct examplars".format(len(return_list_gsm8k_pool_correct)))
            if(len(return_list_gsm8k_pool_correct) >= args.pool_size):
                return_list_gsm8k_pool_correct = return_list_gsm8k_pool_correct[0:args.pool_size]
                print("choose the frist {} examplars".format(args.pool_size))
                with open("./gsm8k_/{}/gsm8k_pool_{}_zeroshot_2.json".format(args.model,args.pool_size),"w") as f:
                    json.dump(return_list_gsm8k_pool_correct,f,indent=2)
            else:
                print("Not enough correct examplars, please increase the generation size")
                with open("./gsm8k_/{}/gsm8k_pool_{}_zeroshot_2.json".format(args.model,len(return_list_gsm8k_pool_correct)),"w") as f:
                    json.dump(return_list_gsm8k_pool_correct,f,indent=2)
        else:
            return_list_gsm8k_pool = generate_pool_for_gsm8k(args.pool_size * 2)
            # filter the correct ones
            return_list_gsm8k_pool_correct = select_correct_examplars(return_list_gsm8k_pool)
            print("Total there are {} correct examplars".format(len(return_list_gsm8k_pool_correct)))
            if(len(return_list_gsm8k_pool_correct) >= args.pool_size):
                return_list_gsm8k_pool_correct = return_list_gsm8k_pool_correct[0:args.pool_size]
                print("choose the frist {} examplars".format(args.pool_size))
                with open("./gsm8k_/{}/gsm8k_pool_{}.json".format(args.model,args.pool_size),"w") as f:
                    json.dump(return_list_gsm8k_pool_correct,f,indent=2) 
            else:
                print("Not enough correct examplars, please increase the generation size")
   
            
    # For train set generation
    if(args.train == True):
        if(args.zero):
            return_list_gsm8k_train = generate_train_for_gsm8k(n = int(args.train_size * 2.2),zero_shot= True)
            print("Total there are {} training examplars and we don't know if they are correct".format(len(return_list_gsm8k_train)))
            if(len(return_list_gsm8k_train) >= args.train_size):
                return_list_gsm8k_train = return_list_gsm8k_train[0:args.train_size]
                print("choose the frist {} examplars".format(args.train_size))
                with open("./gsm8k_/gsm8k_train_{}_zeroshot.json".format(args.train_size),"w") as f:
                    json.dump(return_list_gsm8k_train,f,indent=2)
            else:
                print("Not enough examplars, please increase the generation size")      
        else:
            return_list_gsm8k_train = generate_train_for_gsm8k(n = args.train_size)
            assert len(return_list_gsm8k_train) == args.train_size
            with open("gsm8k_train_{}_3.json".format(args.train_size),"w") as f:
                json.dump(return_list_gsm8k_train,f,indent=2) 
    
    if(args.valid == True):
        if(args.zero):
            return_list_gsm8k_val = generate_val_for_gsm8k(n = int(args.val_size * 3),zero_shot= True)
            print("Total there are {} validation examplars and we don't know if they are correct".format(len(return_list_gsm8k_val)))
            if(len(return_list_gsm8k_val) >= args.val_size):
                return_list_gsm8k_val = return_list_gsm8k_val[0:args.val_size]
                print("choose the frist {} examplars".format(args.val_size))
                with open("./gsm8k_/gsm8k_val_{}_zeroshot.json".format(args.train_size),"w") as f:
                    json.dump(return_list_gsm8k_val,f,indent=2)
            else:
                print("Not enough examplars, please increase the generation size")      
        else:
            return_list_gsm8k_val = generate_val_for_gsm8k(n = args.val_size)
            assert len(return_list_gsm8k_val) == args.val_size
            with open("gsm8k_val_{}.json".format(args.val_size),"w") as f:
                json.dump(return_list_gsm8k_val,f,indent=2)           