from datasets import load_dataset
import os
import openai
import re
import json


dataset = load_dataset("gsm8k","main")
gsm8k_train = dataset["train"]
gsm8k_test = dataset["test"]
import time
import sys
# The human written 8 COT rationales 
prompt = "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.\n\n"
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


def generate_rationale(n):
    count = 0
    return_list = []

    for index in range(7000,n+7000):
        input_to_gpt3 = prompt + 'Q: ' + gsm8k_train["question"][index] + '\nA:'
        received = False
        while not received:
            try:
                openai.api_key =code_api_list[count % len(code_api_list)]

                result = openai.Completion.create(
                model="code-davinci-002",
                prompt=input_to_gpt3,
                max_tokens=256,
                temperature = 0
                )
                received = True
            except:
                error = sys.exc_info()[0]

                print("API error:", error)
                count += 1
                time.sleep(5)               
        # returned full sentance (rationale + answer)
        generated_full = result["choices"][0]["text"]

        pattern = "The answer is \d{1,}\."
        generated_match = re.search(pattern,generated_full.replace("$","").replace(",","").replace("%",""))

        ground_truth_match = re.search('####',gsm8k_train["answer"][index])
        
        temp = {}
        temp['Question'] = gsm8k_train["question"][index]
        if( generated_match != None):
            temp["Rationale"] = "N/A"
            temp["Answer"] = "N/A"
        else:
            continue
            # temp["Rationale"] = generated_full
            # temp["Answer"] = "N/A"
        
        temp["Ground_truth"] = generated_match.group(0)[14:-1]
        return_list.append(temp)
        print("Finished process {}".format(index))

        count = count + 1
        
    return return_list       


def calculate_accuracy():
    with open('./rationale_code_4.json', 'r') as f:
        data = json.load(f)
    match_count = 0
    length = len(data)
    for i in range (0,length):
        if (data[i]["Answer"][14:-1] == data[i]["Ground_truth"]):
            match_count = match_count + 1
    return match_count, length

def select_correct_rationale():
    correct = []
    with open('./rationale_code_temp.json','r') as f:
        data1 = json.load(f)
    with open('./rationale_code_4.json','r') as f:
        data2 = json.load(f)  
    count = 0
    for i in range(0,len(data1)):
        if (data1[i]["Answer"][14:-1] == data1[i]["Ground_truth"]):
            count+= 1
            correct.append(data1[i])
    for i in range(0,len(data2)):
        if (data2[i]["Answer"][14:-1] == data2[i]["Ground_truth"]):
            count+= 1
            correct.append(data2[i])
    print("count = {}".format(count))
    return correct


def main():
    return_list = generate_rationale(120)
    print("the length is {}".format(len(return_list)))
    if(len(return_list)>= 100):
        return_list = return_list[0:100]
    with open("test_for_pseudo_val.json","w") as f:
        json.dump(return_list,f,indent=2) 

    # match_count, length = calculate_accuracy()
    # print(match_count)
    # print(match_count/length)
    # return_list = select_correct_rationale()
    # return_list = return_list[0:100]
    # with open("rationale_code_temp.json","w") as f:
    #     json.dump(return_list,f,indent=2)    
main()