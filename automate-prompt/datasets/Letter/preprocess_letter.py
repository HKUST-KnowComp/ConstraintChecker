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
                "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"]
prompt = 'Q: Take the last letters of the words in "Bill Gates" and concatenate them.\nA: The last letter of "Bill" is "l". The last letter of "Gates" is "s". Concatenating them is "ls". So the answer is ls.\n\nQ: Take the last letters of the words in "Larry Page" and concatenate them.\nA: The last letter of "Larry" is "y". The last letter of "Page" is "e". Concatenating them is "ye". So the answer is ye.\n\nQ: Take the last letters of the words in "Sergey Brin" and concatenate them.\nA: The last letter of "Sergey" is "y". The last letter of "Brin" is "n". Concatenating them is "yn". So the answer is yn.\n\nQ: Take the last letters of the words in "Elon Musk" and concatenate them.\nA: The last letter of "Elon" is "n". The last letter of "Musk" is "k". Concatenating them is "nk". So the answer is nk.\n\n'


# Pool set + train set + val set
dataset = []
dataset_answer = []
# Train set + val set
ood3_dataset = []
ood3_dataset_answer = []
# This is also the test set
ood_dataset = []
ood_dataset_answer = []
with open("./raw/concat2_4shot.txt") as f:
    for line in f.readlines(): 
        dataset.append(line[2:-2].replace("\\n","\n"))
with open("./raw/concat_ood4.txt") as f:
    for line in f.readlines(): 
        ood_dataset.append(line[2:-2].replace("\\n","\n"))
with open("./raw/concat_ood3.txt") as f:
    for line in f.readlines(): 
        ood3_dataset.append(line[2:-2].replace("\\n","\n"))
with open("./raw/concat2_4shot_answer.txt") as f:
    for line in f.readlines(): 
        dataset_answer.append(line.replace("\n",""))
with open("./raw/concat_ood4_answer.txt") as f:
    for line in f.readlines(): 
        ood_dataset_answer.append(line.replace("\n",""))
with open("./raw/concat_ood3_answer.txt") as f:
    for line in f.readlines(): 
        ood3_dataset_answer.append(line.replace("\n",""))

# Extract only the question
for i in range(0,len(dataset)):
    question_match = re.search(pattern=prompt,string=dataset[i])
    if(question_match != None):
        #Extract the question begin with last index of prompt
        dataset[i] = dataset[i][question_match.span()[1]:]
    else:
        # does not match
        print("There exits does not match item")
        sys.exit()

for i in range(0,len(ood_dataset)):
    question_match = re.search(pattern=prompt,string=ood_dataset[i])
    if(question_match != None):
        #Extract the question begin with last index of prompt
        ood_dataset[i] = ood_dataset[i][question_match.span()[1]:]
    else:
        # does not match
        print("There exits does not match item")
        sys.exit()

for i in range(0,len(ood3_dataset)):
    question_match = re.search(pattern=prompt,string=ood3_dataset[i])
    if(question_match != None):
        #Extract the question begin with last index of prompt
        ood3_dataset[i] = ood3_dataset[i][question_match.span()[1]:]
    else:
        # does not match
        print("There exits does not match item")
        sys.exit()
assert len(dataset) ==  len(ood_dataset) == len(ood3_dataset) == 500


def generate_pseudo_pool(n):
    count = 0
    return_list = []

    for index in range(0,n):
        input_to_gpt3 = prompt + dataset[index]

        received = False
        while not received:
            try:
                openai.api_key = code_api_list[count % len(code_api_list)]

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

        pattern = "So the answer is [a-zA-Z]+."
        generated_match = re.search(pattern,generated_full)

        ground_truth = dataset_answer[index]
        
        temp = {}
        temp['Question'] = dataset[index]
        if( generated_match != None):
            temp["Rationale"] = generated_full[0:generated_match.span()[0]-1]
            temp["Answer"] = generated_match.group(0)
        else:
            temp["Rationale"] = generated_full
            temp["Answer"] = "N/A"
        temp["Ground_truth"] = ground_truth
        return_list.append(temp)
        print("Finished process {}".format(index))

        count = count + 1
        
    return return_list   


def calculate_accuracy():
    with open('./letter_pseudo_pool.json', 'r') as f:
        data = json.load(f)
    match_count = 0
    length = len(data)
    for i in range (0,length):
        if (data[i]["Answer"][17:-1].lower() == data[i]["Ground_truth"].lower()):
            match_count = match_count + 1

    return match_count, length

def select_pool():
    correct = []
    with open('./letter_pseudo_pool.json','r') as f:
        data1 = json.load(f)  
    count = 0
    for i in range(0,len(data1)):
        if (data1[i]["Answer"][17:-1].lower() == data1[i]["Ground_truth"].lower()):
            count+= 1
            correct.append(data1[i])

    print("count = {}".format(count))
    return correct


# Preprocess to generate the train set
# The train set is from index 0-100 of the raw ood 3 set
TRAIN_BEGIN_INDEX = 0
def generate_train(n):
    return_list = []
    for index in range(TRAIN_BEGIN_INDEX,TRAIN_BEGIN_INDEX + n):
        temp = {}

        temp['Question'] = ood3_dataset[index]
        temp["Rationale"] = "N/A"
        temp["Answer"] = "N/A"

        temp["Ground_truth"] = ood3_dataset_answer[index]
        return_list.append(temp)
        print("Finished process {}".format(index))
    return return_list    

# The val set is from index 100-200 of the raw ood3 set

VAL_BEGIN_INDEX = 100
def generate_val(n):
    return_list = []
    for index in range(VAL_BEGIN_INDEX,VAL_BEGIN_INDEX + n):
        temp = {}

        temp['Question'] = ood3_dataset[index]
        temp["Rationale"] = "N/A"
        temp["Answer"] = "N/A"

        temp["Ground_truth"] = ood3_dataset_answer[index]
        return_list.append(temp)
        print("Finished process {}".format(index))
    return return_list  

def generate_test():
    return_list = []
    for index in range(0,len(ood_dataset)):
        temp = {}

        temp['Question'] = ood_dataset[index]
        temp["Rationale"] = "N/A"
        temp["Answer"] = "N/A"

        temp["Ground_truth"] = ood_dataset_answer[index]
        return_list.append(temp)
    assert len(return_list) == 500   
    return return_list  
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model")

    parser.add_argument("--generate_pseudo_pool", action="store_true",help="generate the pool with incorrect")
    parser.add_argument("--generate_train", action="store_true",help="generate the train set")
    parser.add_argument("--generate_val", action="store_true",help="generate the validation set")
    parser.add_argument("--generate_test", action="store_true",help="generate the test set")

    args = parser.parse_args()

    if(args.generate_pseudo_pool):
        return_list = generate_pseudo_pool(200)
        with open("letter_pseudo_pool.json","w") as f:
            json.dump(return_list,f,indent=2) 
    # match_count, length = calculate_accuracy()
    # print(match_count)
    # print(match_count/length) 
    return_list = select_pool()
    return_list = return_list[0:100]
    with open("letter_code_pool_100.json","w") as f:
        json.dump(return_list,f,indent=2)   
    if(args.generate_train):
        return_train_list = generate_train(100)
        with open("letter_train_100.json","w") as f:
            json.dump(return_train_list,f,indent=2)
    if(args.generate_val):
        return_val_list = generate_val(100)
        with open("letter_val_100.json","w") as f:
            json.dump(return_val_list,f,indent=2)
    if(args.generate_test):
        return_test_list = generate_test()
        with open("letter_test_500.json","w") as f:
            json.dump(return_test_list,f,indent=2)                 