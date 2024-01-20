# This is the file for generate rationale pool for CommonsenseQA
import jsonlines
import openai
import re
import time
import sys
import json

prompt = "Q: Do hamsters provide food for any animals?\nA: Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So the answer is yes.\n\nQ: Could Brooke Shields succeed at University of Pennsylvania?\nA: Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. So the answer is yes.\n\nQ: Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?\nA: Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5. So the answer is no.\n\nQ: Yes or no: Is it common to see frost during some college commencements?\nA: College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. So the answer is yes.\n\nQ: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?\nA: The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So the answer is no.\n\nQ: Yes or no: Would a pear sink in water?\nA: The density of a pear is about 0.6 g/cm^3, which is less than water. Objects less dense than water float. Thus, a pear would float. So the answer is no.\n\n"
sqa_dataset = []

with open('./raw/task.json','r') as f:
    data = json.load(f)

# The first ten are exclueded because they are used for prompt in original COT paper
sqa_dataset = data["examples"][10:] 

def generate_rationale(n):
    count = 0
    return_list = []

    for index in range(0,n):
        input_to_gpt3 = prompt + 'Q: Yes or no: ' + sqa_dataset[index]["input"] + "\nA:"

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

        pattern = "So the answer is (yes|no)."
        generated_match = re.search(pattern,generated_full)

        pattern_answer = "(Yes|No)."
        ground_truth_match = re.search(pattern_answer,sqa_dataset[index]["target"])
        ground_truth = ground_truth_match.group(0)[:-1]
        
        temp = {}
        temp['Question'] = sqa_dataset[index]["input"]
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
    with open('./sqa_pool_1.json', 'r') as f:
        data = json.load(f)
    match_count = 0
    length = len(data)
    for i in range (0,length):
        if (data[i]["Answer"][17:-1].lower() == data[i]["Ground_truth"].lower()):
            match_count = match_count + 1
    return match_count, length

def select_correct_rationale():
    correct = []
    with open('./sqa_pool_1.json','r') as f:
        data1 = json.load(f)  
    count = 0
    for i in range(0,len(data1)):
        if (data1[i]["Answer"][17:-1].lower() == data1[i]["Ground_truth"].lower()):
            count+= 1
            correct.append(data1[i])

    print("count = {}".format(count))
    return correct


# Preprocess to generate the train set
# The train set is from index 200-300 of the raw train split
TRAIN_BEGIN_INDEX = 200
def generate_train(n):
    return_list = []
    for index in range(TRAIN_BEGIN_INDEX,TRAIN_BEGIN_INDEX + n):
        temp = {}

        temp['Question'] = sqa_dataset[index]["input"]
        temp["Rationale"] = "N/A"
        temp["Answer"] = "N/A"

        pattern_answer = "(Yes|No)."
        ground_truth_match = re.search(pattern_answer,sqa_dataset[index]["target"])
        ground_truth = ground_truth_match.group(0)[:-1]
        temp["Ground_truth"] = ground_truth
        return_list.append(temp)
        print("Finished process {}".format(index))
    return return_list    

VAL_BEGIN_INDEX = 300
def generate_val(n):
    return_list = []
    for index in range(VAL_BEGIN_INDEX,VAL_BEGIN_INDEX + n):
        temp = {}

        temp['Question'] = sqa_dataset[index]["input"]
        temp["Rationale"] = "N/A"
        temp["Answer"] = "N/A"

        pattern_answer = "(Yes|No)."
        ground_truth_match = re.search(pattern_answer,sqa_dataset[index]["target"])
        ground_truth = ground_truth_match.group(0)[:-1]
        temp["Ground_truth"] = ground_truth
        return_list.append(temp)
        print("Finished process {}".format(index))
    return return_list  

def generate_test():
    return_list = []
    sqa_testset = sqa_dataset[400:]
    for index in range(0,len(sqa_testset)):
        temp = {}

        temp['Question'] = sqa_testset[index]["input"]
        temp["Rationale"] = "N/A"
        temp["Answer"] = "N/A"

        pattern_answer = "(Yes|No)."
        ground_truth_match = re.search(pattern_answer,sqa_testset[index]["target"])
        ground_truth = ground_truth_match.group(0)[:-1]
        temp["Ground_truth"] = ground_truth
        return_list.append(temp)
    assert len(return_list) == 1880   
    return return_list  
def main():
    # TODO using argument to call each function
    # return_list = generate_rationale(200)
    # with open("sqa_pool_1.json","w") as f:
    #     json.dump(return_list,f,indent=2) 
    # match_count, length = calculate_accuracy()
    # print(match_count)
    # print(match_count/length) 
    # return_list = select_correct_rationale()
    # return_list = return_list[0:100]
    # with open("sqa_code_pool_100.json","w") as f:
    #     json.dump(return_list,f,indent=2)    

    # For generate train set
    return_train_list = generate_train(100)
    with open("sqa_train_100.json","w") as f:
        json.dump(return_train_list,f,indent=2)    
    # # For generate valid set
    return_val_list = generate_val(100)
    with open("sqa_val_100.json","w") as f:
        json.dump(return_val_list,f,indent=2)
    # For generate test set
    return_list = generate_test()
    with open("sqa_test.json","w") as f:
        json.dump(return_list,f,indent=2)      
main()

