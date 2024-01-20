# code
import concurrent
import openai
from concurrent.futures.thread import ThreadPoolExecutor
import random
# for complex
# selected_index =[54, 96, 18, 63, 82, 52, 87]

#selected_index = [1,2,3,4,5,6,7,8]
# only 1 think
#selected_index = [17, 57, 41, 36, 55, 39, 86, 73]#[71, 11, 37,  9,  9, 99, 64, 58]# [15, 17, 10, 95, 12, 41, 42, 30]#[17, 57, 41, 36, 55, 39, 86, 73]
#random.shuffle(selected_index)
# selected_index =  [33, 34, 98, 81, 83, 41, 54, 57]# Input selected index from trained sample probability
#selected_index = [46, 42,  2, 14, 43, 89, 28]
#selected_index = [25,  1, 74, 76, 34,  2] 
#selected_index = [35, 80, 54, 99]

#--------------text davinci index
# selected_index=[14, 16, 73, 96, 64, 17, 43, 58] 
# selected_index = [9,18,33,56,87,91]
# selected_index = [33,15,27,39,48,88]
# selected_index = [52,15,27,39,48,88] # esnli
# selected_index = [9,16,23,8,4,33]
# selected_index = [9,17,23,8] # obqa

# selected_index =[1,0,0,1,0,1,0,1]

# selected_index = [1,0,1,0,0,1,0,1]#[0, 1, 0, 1, 1, 0, 1, 0]
# selected_index = [1,1,1,1,1,1,1,1]
#[28, 87, 54, 33, 46, 69, 50, 15]#[35,54,89,99]# [35, 89, 54, 99]
# for i in range(1,8):
#     selected_index[i] = i*2 + selected_index[i]

selected_index = [12,  2, 11,  0, 24]

#[75, 82, 75, 30, 19,  9, 49, 69]
# [17, 57, 41, 36, 55, 39, 86, 73] 
# random.shuffle(selected_index)

manual_cot = []
for i in range(0,4):
    if(selected_index[i] == 1):
        manual_cot.append('Q: Poison causes harm to which of the following? (a) a Tree (b) a robot (c) a house (d) a car\nA: Poison will harm living things, only a tree is a living thing. The answer is (a).\n\n')
        #manual_cot.append('What is the sentiment of the following sentence?\n"that loves its characters and communicates something rather beautiful about human nature"\nA: "loves its characters" indicates positive sentiment. The answer is positive.\n\n')
        # manual_cot.append("Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: Let's think about it happily. We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.\n\n")
        # manual_cot.append("Q: What do people use to absorb extra ink from a fountain pen?\nAnswer Choices:\n(a) shirt pocket\n(b) calligrapher's hand\n(c) inkwell\n(d) desk drawer\n(e) blotter\nA: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. So the answer is (e).\n\n")
    elif(selected_index[i] == 2):
        manual_cot.append('Q: As you look deeper into a Marbel you can see (a) the future (b) minut defects (c) colors (d) the other side\nA: Marbel is not transparent, so you can not see the other side. Marbel does not necessarily have multiple colors. You will see minut defects. The answer is (b).\n\n')
        #manual_cot.append('What is the sentiment of the following sentence?\n"hide new secretions from the parental units"\nA: If people are hiding something, it means the sentiment is on the negative side. The answer is negative.\n\n')
        # manual_cot.append("Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: Let's think about it happily. There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.\n\n")
        # manual_cot.append("Q: What home entertainment equipment requires cable?\nAnswer Choices:\n(a) radio shack\n(b) substation\n(c) television\n(d) cabinet\nA: The answer must require cable. Of the above choices, only television requires cable. So the answer is (c).\n\n")
    elif(selected_index[i] == 3):
        manual_cot.append('Q: When food is reduced in the stomach (a) the mind needs time to digest (b) take a second to digest what I said (c) nutrients are being deconstructed (d) reader’s digest is a body of works\nA: The food is being deconstructed in the stomach during digestion. The answer is (c).\n\n')
        #manual_cot.append('What is the sentiment of the following sentence?\n"the greatest musicians"\nA: By saying someone being the "greatest", it means positive sentiment. The answer is positive.\n\n')
        # manual_cot.append("Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Let's think about it happily. Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.\n\n")
        # manual_cot.append("Q: The fox walked from the city into the forest, what was it looking for?\nAnswer Choices:\n(a) pretty flowers\n(b) hen house\n(c) natural habitat\n(d) storybook\nA: Answer: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. So the answer is (b).\n\n")
    elif(selected_index[i] == 4):
        manual_cot.append('Q: The sun is responsible for (a) puppies learning new tricks (b) children growing up and getting old (c) ﬂowers wilting in a vase (d) plants sprouting, blooming and wilting\nA: The sun can affect the growing of living things, like plants. The answer is (d).\n\n')
        #manual_cot.append('What is the sentiment of the following sentence?\n"contains no wit , only labored gags"\nA: "contains no wit" is clearly a negative sentiment. The answer is negative.\n\n')
        # manual_cot.append("Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Let's think about it happily. Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.\n\n")
        # manual_cot.append("Q: Sammy wanted to go to where the people were. Where might he go?\nAnswer Choices:\n(a) populated areas\n(b) race track\n(c) desert\n(d) apartment\n(e) roadblock\nA: The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. So the answer is (a).\n\n")
    elif(selected_index[i] == 5):
        manual_cot.append('What is the sentiment of the following sentence?\n"demonstrates that the director of such hollywood blockbusters as patriot games can still turn out a small , personal ﬁlm with an emotional wallop ."\nA: "can still turn out a small , personal ﬁlm with an emotional wallop ." indicates sentiment on the positive side. The answer is positive.\n\n')
        # manual_cot.append("Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: Let's think about it happily. He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.\n\n")
        # manual_cot.append("Q: Where do you put your grapes just before checking out?\nAnswer Choices:\n(a) mouth\n(b) grocery cart\n(c)super market\n(d) fruit basket\n(e) fruit market\nA: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. So the answer is (b).\n\n")
    elif(selected_index[i] == 6):
        manual_cot.append('What is the sentiment of the following sentence?\n"that ’s far too tragic to merit such superﬁcial treatment"\nA: "far too tragic" and "to merit such superﬁcial treatment" both mean negative sentiments. The answer is negative.\n\n')
        # manual_cot.append("Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: Let's think about it happily. There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.\n\n")
        # manual_cot.append("Q: Google Maps and other highway and street GPS services have replaced what?\nAnswer Choices:\n(a) united states\n(b) mexico\n(c) countryside\n(d) atlas\nA: The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. So the answer is (d).\n\n")
    elif(selected_index[i] == 7):
        manual_cot.append("Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Let's think about it happily. Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.\n\n")
        # manual_cot.append("Q: Before getting a divorce, what did the wife feel who was doing all the work?\nAnswer Choices:\n(a) harder\n(b) anguish\n(c) bitterness\n(d) tears\n(e) sadness\nA: The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. So the answer is (c).\n\n")
    elif(selected_index[i] == 8):
        manual_cot.append("Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Let's think about it happily. She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.\n\n")


# selected_index = []
# import random
# for i in range(0,6):
#     selected_index.append(random.randint(0,100))
# for pseudo 
# selected_index = [19,  5, 71, 40, 83, 69, 42, 20]
# selected_index = [4, 1, 1, 0, 8, 7, 2, 5]
# selected_index = [ 85,  16,   0,  50, 107, 127, 117,  16]
import openai
import time
import sys
import os
import re
import argparse
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from api_key import code_api_list
from data.knowledge import KnowledgeDataset
import wandb

def complete_gpt3(input,count,model = "text-davinci-002",temperature = 0):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            openai.api_key = code_api_list[count % len(code_api_list)]
            response = openai.Completion.create(
                model=model,
                prompt=input,
                max_tokens=256,
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
            print(code_api_list[count % len(code_api_list)])

            count += 1
            time.sleep(5)
    return response
# TODO Move into function later
############################ For HOP calculation  ##############
with open('./datasets/Annotation/hop.json', 'r') as f:
    data = json.load(f)
hop_total = {'3': 370, '2': 326, '4': 298, '5': 174, '6': 88, '7': 40, '8': 20, '9': 2, '11': 1}
hop_count = {'3': 0, '2': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '11': 0}
############################ For HOP calculation  ##############
batch_size = 20
def generate_result(args,pool_dataset,test_dataset):
    match_count = []
    total_count = []
    for index in range(0, int(len(test_dataset)/batch_size)+1 if len(test_dataset) % batch_size != 0 else int(len(test_dataset)/batch_size) ):
        # handle last batch 
        if (index == int(len(test_dataset)/batch_size)):
            question, ground_truth = test_dataset[0+index*batch_size:]['Question'],test_dataset[0+index*batch_size:]['Ground_truth']
        else:
            question, ground_truth = test_dataset[0+index*batch_size:batch_size+index*batch_size]['Question'],test_dataset[0+index*batch_size:batch_size+index*batch_size]['Ground_truth']
        prompts = []
        targets = []

        total_count.append(len(question))
        for i in range(0,len(question)):
            if(args.task == "strategyqa"):
                prompts.append("Q: Yes or no: " + question[i] + "\n" + "A:")
            elif(args.task == "letter" or args.task == "svamp" or args.task == "asdiv" or args.task == "singleop" or args.task == "singleeq"):
                #prompts.append("Question" + question[i][1:-2] + "Answer: Let's think step by step.")
                prompts.append(question[i])            
            elif(args.task == "babi"):
                prompts.append(question[i] + "\nAnswer:")
            elif(args.task == "esnli"):
                prompts.append(question[i] + "A:")
            elif(args.task == "sst2"):
                prompts.append(question[i] + "\nA:")
            elif(args.task == "obqa"):
                prompts.append(question[i] + "\nA:")
            else:
                # prompts.append("Question: " + question[i] + "\n" + "Answer: Let's think step by step.")
                prompts.append("Q: " + question[i] + "\n" + "A:")
            targets.append(ground_truth[i])
        
        prompt_idx = selected_index
        for k in range(0,len(prompt_idx)):
            if(args.task == "gsm8k" or args.task == "svamp" or args.task == "asdiv" or args.task == "singleop"  or args.task == "singleeq"):
                prompt = "Q: " + pool_dataset[prompt_idx[k]]['Question'] + "\nA:" + pool_dataset[prompt_idx[k]]['Rationale'] + " The answer is " + pool_dataset[prompt_idx[k]]['Ground_truth'] + ".\n\n"
            elif(args.task == "csqa"):
                prompt = "Question: " + pool_dataset[prompt_idx[k]]['Question'] + "\nAnswer: Let's think step by step." + pool_dataset[prompt_idx[k]]['Rationale'] + " So the answer is (" + pool_dataset[prompt_idx[k]]['Ground_truth'].lower() + ")" + ".\n\n"
            elif(args.task == "strategyqa"):
                if(k == 4 or k == 5):
                    prompt = "Q: " + pool_dataset[prompt_idx[k]]['Question'] + "\nA:" + pool_dataset[prompt_idx[k]]['Rationale'] + " " + pool_dataset[prompt_idx[k]]['Answer']+ "\n\n"
                else:
                    prompt = "Q: Yes or no: " + pool_dataset[prompt_idx[k]]['Question'] + "\nA:" + pool_dataset[prompt_idx[k]]['Rationale'] + " " + pool_dataset[prompt_idx[k]]['Answer']+ "\n\n"
            elif(args.task == "letter"):
                prompt = pool_dataset[prompt_idx[k]]['Question']  + pool_dataset[prompt_idx[k]]['Rationale'] + " " + pool_dataset[prompt_idx[k]]['Answer'] + "\n\n"
            elif(args.task == "esnli"):
                prompt = pool_dataset[prompt_idx[k]]['Question']  + "A: " + pool_dataset[prompt_idx[k]]['Rationale'] + " " + pool_dataset[prompt_idx[k]]['Answer'] + "\n\n"
            elif(args.task == "obqa"):
                prompt = pool_dataset[prompt_idx[k]]['Question']  + "\nA:" + pool_dataset[prompt_idx[k]]['Rationale'] + " " + pool_dataset[prompt_idx[k]]['Answer'] + "\n\n"
            elif(args.task == "sst2"):                  
                prompt = pool_dataset[prompt_idx[k]]['Question']  + "\nA:" + pool_dataset[prompt_idx[k]]['Rationale'] + " " + pool_dataset[prompt_idx[k]]['Answer'] + "\n\n"

        #prompt = 'Q: Take the last letters of the words in "Bill Gates" and concatenate them.\nA: The last letter of "Bill" is "l". The last letter of "Gates" is "s". Concatenating them is "ls". So the answer is ls.\n\nQ: Take the last letters of the words in "Larry Page" and concatenate them.\nA: The last letter of "Larry" is "y". The last letter of "Page" is "e". Concatenating them is "ye". So the answer is ye.\n\nQ: Take the last letters of the words in "Sergey Brin" and concatenate them.\nA: The last letter of "Sergey" is "y". The last letter of "Brin" is "n". Concatenating them is "yn". So the answer is yn.\n\nQ: Take the last letters of the words in "Elon Musk" and concatenate them.\nA: The last letter of "Elon" is "n". The last letter of "Musk" is "k". Concatenating them is "nk". So the answer is nk.\n\n'
        #prompt = "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.\n\n"
        # A1
        #prompt = "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: Let's think it by using variables. There are x trees originally. Then there were y trees after some more were planted. So there must have been y - x trees. Considering x = 15 and y = 21, the answer is y - x = 21 - 15 = 6. The answer is 6.\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: Let's think it by using variables. There are originally x cars. y more cars arrive. Considering x = 3 and y = 2, the answer is x + y = 3 + 2 = 5. The answer is 5.\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Let's think it by using variables. Originally, Leah had x chocolates. Her sister had y. So in total they had x + y. After eating z, they had x + y - z. Considering x = 32, y = 42, and z = 35, the answer is x + y - z = 32 + 42 - 35 = 39. The answer is 39.\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Let's think it by using variables. Jason started with x lollipops. Then he had y after giving some to Denny. So he gave Denny x - y. Considering x = 20 and y = 12, the answer is x - y = 20 - 12 = 8. The answer is 8.\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: Let's think it by using variables. Shawn started with x toys. If he got y toys each from his mom and dad, then that is 2 * y more toys. So he have x + 2 * y. Considering x = 5 and y = 2, the answer is x + 2 * y = 5 + 2 * 2 = 9. The answer is 9.\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: Let's think it by using variables. There were originally x computers. For each of y days, z more computers were added. So y * z computers were added. Considering x = 9, y = 4, and z = 5, the answer is x + y * z = 9 + 4 * 5 = 29. The answer is 29.\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Let's think it by using variables. Michael started with x golf balls. After losing y on tuesday, he had x - y. After losing z more, he had x - y - z golf balls. Considering x = 58, y = 23, and z = 2, the answer is x - y - z = 58 - 23 - 2 = 33. The answer is 33.\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Let's think it by using variables. Olivia had x dollars. y bagels for z dollars each will be y * z dollars. So she has x - y * z dollars left. Considering x = 23, y = 5 and z = 3, the answer is x - y * z = 23 - 5 * 3 = 8. The answer is 8.\n\n"
        # A2
        #prompt = "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: Let's think it by using variables. There are x trees originally. Then there were y trees after some more were planted. So there must have been y - x trees. Considering x = 15 and y = 21, we substitute the variable into previous equation. The answer is 6.\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: Let's think it by using variables. There are originally x cars. y more cars arrive. So there are x + y cars. Considering x = 3 and y = 2, we substitute the variable into previous equation. The answer is 5.\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Let's think it by using variables. Originally, Leah had x chocolates. Her sister had y. So in total they had x + y. After eating z, they had x + y - z. Considering x = 32, y = 42, and z = 35, we substitute the variable into previous equation. The answer is 39.\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Let's think it by using variables. Jason started with x lollipops. Then he had y after giving some to Denny. So he gave Denny x - y. Considering x = 20 and y = 12, we substitute the variable into previous equation. The answer is 8.\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: Let's think it by using variables. Shawn started with x toys. If he got y toys each from his mom and dad, then that is 2 * y more toys. So he have x + 2 * y. Considering x = 5 and y = 2, we substitute the variable into previous equation. The answer is 9.\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: Let's think it by using variables. There were originally x computers. For each of y days, z more computers were added. So y * z computers were added. Considering x = 9, y = 4, and z = 5, we substitute the variable into previous equation. The answer is 29.\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Let's think it by using variables. Michael started with x golf balls. After losing y on tuesday, he had x - y. After losing z more, he had x - y - z golf balls. Considering x = 58, y = 23, and z = 2, we substitute the variable into previous equation. The answer is 33.\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Let's think it by using variables. Olivia had x dollars. y bagels for z dollars each will be y * z dollars. So she has x - y * z dollars left. Considering x = 23, y = 5 and z = 3, we substitute the variable into previous equation. The answer is 8.\n\n"
        #prompt = "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: Let's think it by using variables. First, let's define some variables: x = 15, and y = 21. There are x trees originally. Then there were y trees after some more were planted. So there must have been y - x = 21 - 15 = 6 trees. The answer is 6.\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: Let's think it by using variables. First, let's define some variables: x = 3, and y = 2. There are originally x cars. y more cars arrive. So there are x + y = 3 + 2 = 5 cars. The answer is 5.\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Let's think it by using variables. First, let's define some variables: x = 32, y = 42, and z = 35. Originally, Leah had x chocolates. Her sister had y. So in total they had x + y. After eating z, they had x + y - z = 32 + 42 - 35 = 39. The answer is 39.\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Let's think it by using variables. First, let's define some variables: x = 20 and y = 12. Jason started with x lollipops. Then he had y after giving some to Denny. So he gave Denny x - y = 20 - 12 = 8. The answer is 8.\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: Let's think it by using variables. First, let's define some variables: x = 5 and y = 2. Shawn started with x toys. If he got y toys each from his mom and dad, then that is 2 * y more toys. So he have x + 2 * y = 5 + 2 * 2 = 9. The answer is 9.\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: Let's think it by using variables. First, let's define some variables: x = 9, y = 4, and z = 5. There were originally x computers. For each of y days, z more computers were added. So y * z computers were added. So there are x + y * z = 9 + 4 * 5 = 29 computers. The answer is 29.\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Let's think it by using variables. First, let's define some variables: x = 58, y = 23, and z = 2. Michael started with x golf balls. After losing y on tuesday, he had x - y. After losing z more, he had x - y - z = 58 - 23 - 2 = 33 golf balls. The answer is 33.\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Let's think it by using variables. First, let's define some variables: x = 23, y = 5 and z = 3. Olivia had x dollars. y bagels for z dollars each will be y * z dollars. So she has x - y * z = 23 - 5 * 3 = 8 dollars left. The answer is 8.\n\n"
        #A 10
        #prompt = "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: There were 15 trees originally. After some more were planted, there were a total of 21 trees. To find out how many trees were planted, we can subtract the original number of trees from the final number of trees: 21 trees - 15 trees = 6 trees. So the answer is 6.\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There were originally 3 cars. When 2 more cars arrived, the total number of cars became 3 cars + 2 cars = 5 cars. The answer is 5.\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Leah and her sister had a total of 74 chocolates originally, because Leah had 32 chocolates and her sister had 42 chocolates. This can be found by adding the number of chocolates each person had: 32 chocolates + 42 chocolates = 74 chocolates. After eating 35 chocolates, they had a total of 39 chocolates remaining. This can be found by subtracting the number of chocolates eaten from the original total: 74 chocolates - 35 chocolates = 39 chocolates. The answer is 39.\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason had 20 lollipops originally, and then gave some to Denny, leaving him with 12 lollipops. To find out how many lollipops Jason gave to Denny, we can subtract the number of lollipops Jason had after giving some away from the original number: 20 lollipops - 12 lollipops = 8 lollipops. The answer is 8.\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: Shawn started with 5 toys, and then received 2 toys each from his mom and dad, for a total of 4 more toys. We can find the total number of toys Shawn has by adding the number of toys he started with to the number of toys he received: 5 toys + 4 toys = 9 toys. The answer is 9.\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There were originally 9 computers. During a 4-day period, 5 more computers were added each day, for a total of 20 additional computers. We can find the total number of computers added by multiplying the number of computers added each day by the number of days: 5 computers/day * 4 days = 20 computers. The final number of computers is found by adding the number of computers originally present to the number of computers added: 9 computers + 20 computers = 29 computers. The answer is 29.\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael started with 58 golf balls. After losing 23 on Tuesday, he had 35 golf balls remaining. This can be found by subtracting the number of golf balls lost from the original number: 58 golf balls - 23 golf balls = 35 golf balls. After losing 2 more golf balls, Michael had 33 golf balls remaining. This can be found by subtracting the number of golf balls lost from the current number: 35 golf balls - 2 golf balls = 33 golf balls. The answer is 33.\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Olivia had 23 dollars originally. She spent 15 dollars on 5 bagels at a cost of 3 dollars each. We can find the cost of the bagels by multiplying the number of bagels by the cost per bagel: 5 bagels * 3 dollars/bagel = 15 dollars. After purchasing the bagels, Olivia had 8 dollars remaining. This can be found by subtracting the cost of the bagels from the original amount of money Olivia had: 23 dollars - 15 dollars = 8 dollars. The answer is 8.\n\n"
        #BABI
        #prompt = "Passage: Cats are afraid of wolves.\nMice are afraid of cats.\nSheep are afraid of cats.\nGertrude is a mouse.\nWolves are afraid of mice.\nWinona is a sheep.\nJessica is a mouse.\nEmily is a cat.\nQuestion: What is gertrude afraid of?\nAnswer: Gertrude is afraid of cat because Gertrude is a mouse and mice are afraid of cats. The answer is cat.\n\nPassage: Wolves are afraid of sheep.\nMice are afraid of wolves.\nCats are afraid of mice.\nEmily is a cat.\nWinona is a wolf.\nSheep are afraid of cats.\nJessica is a cat.\nGertrude is a sheep.\nQuestion: What is emily afraid of?\nAnswer: Emily is afraid of mouse because Emily is a cat and cats are afraid of mice. The answer is mouse.\n\nPassage: Cats are afraid of mice.\nMice are afraid of sheep.\nWolves are afraid of sheep.\nJessica is a cat.\nSheep are afraid of mice.\nGertrude is a sheep.\nEmily is a wolf.\nWinona is a wolf.\nQuestion: What is emily afraid of?\nAnswer: Emily is afraid of sheep because Emily is a wolf and Wolves are afraid of sheep. The answer is sheep.\n\nPassage: Mice are afraid of cats.\nGertrude is a mouse.\nCats are afraid of wolves.\nSheep are afraid of mice.\nJessica is a cat.\nEmily is a cat.\nWolves are afraid of sheep.\nWinona is a sheep.\nQuestion: What is jessica afraid of?\nAnswer: Jessica is afraid of wolf because Jessica is a cat and Cats are afraid of wolves. The answer is wolf.\n\nPassage: Wolves are afraid of mice.\nSheep are afraid of cats.\nWinona is a sheep.\nMice are afraid of wolves.\nJessica is a sheep.\nCats are afraid of sheep.\nGertrude is a wolf.\nEmily is a wolf.\nQuestion: What is winona afraid of?\nAnswer: Winona is afraid of cat because Winona is a sheep and Sheep are afraid of cats. The answer is cat.\n\n"
        # prompt = "".join(manual_cot)

            prompts = [(prompt + x).strip() for x in prompts] # FIXME 缩进一格

        # with ThreadPoolExecutor(max_workers=5) as executor:
        #     future1 = executor.submit(complete_gpt3, prompts[0:10], index)
        #     future2 = executor.submit(complete_gpt3, prompts[10:20], index+5)
            # future3 = executor.submit(complete_gpt3, prompts[20:30], index+10)
        #     # future4 = executor.submit(complete_gpt3, prompts[30:40], index+15)
        #     # future5 = executor.submit(complete_gpt3, prompts[40:50], index+20)
            # response = future1.result()
            # response2 = future2.result()
            # response3 = future3.result()
        #     # response4 = future4.result()
        #     # response5 = future5.result()

        response = complete_gpt3(model="text-davinci-002",temperature = 0,input=prompts,count=index)

        # for j in range(0,10):
        #     response['choices'].append(response2['choices'][j])
        # for j in range(0,10):
            # response['choices'].append(response3['choices'][j])
        # for j in range(0,10):
            # response['choices'].append(response3['choices'][j])
        # for j in range(0,10):
        #     response['choices'].append(response5['choices'][j])
        
        match = 0
        for i , target in enumerate(targets):
            text = response['choices'][i]["text"]
            # For Task gsm8k
            if(args.task == "gsm8k" or args.task == "svamp" or args.task == "asdiv" or args.task == "singleop"  or args.task == "singleeq"):
                pattern = "The answer is \d{1,}\."
                sttr = re.search(pattern, text.replace("$","").replace(",","").replace("%",""))
                if (sttr is not None):
                    #check if match the ground truth
                    if(sttr.group(0)[14:-1] == target.replace(",","")):
                        hop_count[data[index*10 + i]["hop"]] += 1
                        match += 1
                        # print("No.{}, correct".format(index*10 + i))
                #     else:
                #         pattern = "= (\(?\-?\d+\.?\d+ [(|)|\+|\-|\*|\/|\s|\-?\d+\.?\d+]+)"
                #         sttr = re.findall(pattern, text.replace(",", "").replace("$", "").replace("%", ""))
                #         if(len(sttr) > 0):
                #             try:
                #                 answer = str(int(eval(sttr[0])))
                #                 if(answer == target.replace(",","")):
                #                     match += 1
                #                     print("No.{}, correct".format(index*10 + i))
                #                 else:
                #                     print("No.{}, fail".format(index*10 + i))
                #             except:
                #                 print("No.{}, fail".format(index*10 + i))
                #                 continue
                #         else:
                #             print("No.{}, fail".format(index*10 + i))
                # else:
                #     print("No.{}, fail".format(index*10 + i))
  
            # For Task Commonsense QA
            elif(args.task == "csqa"):
                pattern = "So the answer is \([a-z|A-Z]\)."
                sttr = re.search(pattern, text)
                if (sttr is not None):
                    if(sttr.group(0)[-3:-2].lower() == target.lower()):
                        match += 1
            # For Task Strategy QA
            elif(args.task == "strategyqa"):
                pattern = "So the answer is (yes|no)."
                sttr = re.search(pattern, text)
                if (sttr is not None):
                    if(sttr.group(0)[17:-1].lower() == target.lower()):
                        match += 1
            elif(args.task == "letter"):      
                pattern = "So the answer is [a-zA-Z]+."
                sttr = re.search(pattern, text)
                if (sttr is not None):
                    if(sttr.group(0)[17:-1].lower() == target.lower()):
                        match += 1
            elif(args.task == "babi"):
                pattern = "The answer is \w{1,}\."
                sttr = re.search(pattern, text)
                if (sttr is not None):
                    if(sttr.group(0)[14:-1].lower() == target.lower()):
                        match += 1
            elif(args.task == "esnli"):
                pattern = "The answer is (yes|no)."
                sttr = re.search(pattern, text)
                if (sttr is not None): # Yes or No
                    if(sttr.group(0)[14:-1].lower() == "yes" and target == 0):
                        match += 1
                    elif(sttr.group(0)[14:-1].lower() == "no" and target == 2):
                        match += 1
                else: # Not possible to tell
                    if(target == 1):
                        match += 1
            elif(args.task == "sst2"):
                pattern = "The answer is (positive|negative)."
                sttr = re.search(pattern, text)
                if (sttr is not None): # Yes or No
                    if(sttr.group(0)[14:-1].lower() == "positive" and target == 1):
                        match += 1
                    elif(sttr.group(0)[14:-1].lower() == "negative" and target == 0):
                        match += 1
            elif(args.task == "obqa"):
                pattern = "The answer is \([a-z|A-Z]\)."
                sttr = re.search(pattern, text)
                if (sttr is not None):
                    if(sttr.group(0)[-3:-2].lower() == target.lower()):
                        match += 1
        match_count.append(match)
        print("Current Match Count : {}, current Total count :{}/{}, acc is {} \n".format(sum(match_count),sum(total_count),len(test_dataset),sum(match_count)/sum(total_count)))
        wandb.log({"match_count":sum(match_count),"total_count":sum(total_count),"acc":sum(match_count)/sum(total_count)})
        #print(hop_count)
    return match_count

def find_most_frequent(arr,n):
    maxcount = 0
    element_having_max_freq = "N/A"
    for i in range(0,n):
        count = 0
        for j in range(0,n):
            if(arr[i] == arr[j]):
                count += 1
        if(count > maxcount and arr[i] != "N/A"):
            maxcount = count
            element_having_max_freq = arr[i]
    return maxcount,element_having_max_freq

def generate_result_self_consist(args,pool_dataset,test_dataset,multipath):
    match_count_5,match_count_10,match_count_20,match_count_40= [[] for x in range(4)]
    total_count = []
    for index in range(0, int(len(test_dataset)/batch_size)+1 if len(test_dataset) % batch_size != 0 else int(len(test_dataset)/batch_size)):
        # handle last batch
        if (index == int(len(test_dataset)/batch_size)):
            question, ground_truth = test_dataset[0+index*batch_size:]['Question'],test_dataset[0+index*batch_size:]['Ground_truth']
        else:
            question, ground_truth = test_dataset[0+index*batch_size:batch_size+index*batch_size]['Question'],test_dataset[0+index*batch_size:batch_size+index*batch_size]['Ground_truth']
        prompts = []
        targets = []

        total_count.append(len(question))
        for i in range(0,len(question)):
            if(args.task == "strategyqa"):
                prompts.append("Q: Yes or no: " + question[i] + "\n" + "A:")
            elif(args.task == "letter" or args.task == "svamp" or args.task == "asdiv"):
                prompts.append(question[i]) 
            else:
                prompts.append("Q: " + question[i] + "\n" + "A:")
            targets.append(ground_truth[i])

        prompt_idx = selected_index
        for k in range(0,len(prompt_idx)):
            if(args.task == "gsm8k" or args.task == "svamp" or args.task == "asdiv"):
                prompt = "Q: " + pool_dataset[prompt_idx[k]]['Question'] + "\nA:" + pool_dataset[prompt_idx[k]]['Rationale'] + " The answer is " + pool_dataset[prompt_idx[k]]['Ground_truth'] + ".\n\n"
            elif(args.task == "csqa"):
                prompt = "Q: " + pool_dataset[prompt_idx[k]]['Question'] + "\nA:" + pool_dataset[prompt_idx[k]]['Rationale'] + " So the answer is (" + pool_dataset[prompt_idx[k]]['Ground_truth'].lower() + ")" + ".\n\n"
            elif(args.task == "strategyqa"):            
                if(k == 4 or k == 5):
                    prompt = "Q: " + pool_dataset[prompt_idx[k]]['Question'] + "\nA:" + pool_dataset[prompt_idx[k]]['Rationale'] + " " + pool_dataset[prompt_idx[k]]['Answer']+ "\n\n"
                else:
                    prompt = "Q: Yes or no: " + pool_dataset[prompt_idx[k]]['Question'] + "\nA:" + pool_dataset[prompt_idx[k]]['Rationale'] + " " + pool_dataset[prompt_idx[k]]['Answer']+ "\n\n"
            elif(args.task == "letter"):
                prompt = pool_dataset[prompt_idx[k]]['Question']  +pool_dataset[prompt_idx[k]]['Rationale'] + " " + pool_dataset[prompt_idx[k]]['Answer'] + "\n\n"
        #prompt = 'Q: Take the last letters of the words in "Bill Gates" and concatenate them.\nA: The last letter of "Bill" is "l". The last letter of "Gates" is "s". Concatenating them is "ls". So the answer is ls.\n\nQ: Take the last letters of the words in "Larry Page" and concatenate them.\nA: The last letter of "Larry" is "y". The last letter of "Page" is "e". Concatenating them is "ye". So the answer is ye.\n\nQ: Take the last letters of the words in "Sergey Brin" and concatenate them.\nA: The last letter of "Sergey" is "y". The last letter of "Brin" is "n". Concatenating them is "yn". So the answer is yn.\n\nQ: Take the last letters of the words in "Elon Musk" and concatenate them.\nA: The last letter of "Elon" is "n". The last letter of "Musk" is "k". Concatenating them is "nk". So the answer is nk.\n\n'

            prompts = [(prompt + x).strip() for x in prompts]
        response_list = []
        for path in range(0,multipath):
            response = complete_gpt3(model="text-davinci-002",temperature = 0.7,input = prompts,count = index * 40 + path)
            response_list.append(response)
            if(path % 5 == 0):
                print("Batch {} / {} Path {} has finished".format(index+1, int(len(test_dataset)/batch_size)+1,path))
        answer_list = [] 
        
        for i in range(0,len(targets)):
            list_each_question = []
            for j in range(0,multipath):
                text = response_list[j]['choices'][i]["text"]
                if(args.task == "gsm8k" or args.task == "svamp" or args.task == "asdiv"):
                    pattern = "The answer is \d{1,}\."
                    sttr = re.search(pattern, text.replace("$","").replace(",","").replace("%",""))
                    if (sttr is not None):
                        list_each_question.append(sttr.group(0)[14:-1])
                    else:
                        list_each_question.append("N/A")
                elif(args.task == "csqa"):
                    pattern = "So the answer is \([a-z|A-Z]\)."
                    sttr = re.search(pattern, text)
                    if (sttr is not None):
                        list_each_question.append(sttr.group(0)[-3:-2].lower())
                    else:
                        list_each_question.append("N/A")
                elif(args.task == "strategyqa"):
                    pattern = "So the answer is (yes|no)."
                    sttr = re.search(pattern, text)
                    if (sttr is not None):
                        list_each_question.append(sttr.group(0)[17:-1].lower())
                    else:
                        list_each_question.append("N/A")
                elif(args.task == "letter"):
                    pattern = "So the answer is [a-zA-Z]+."
                    sttr = re.search(pattern, text)
                    if (sttr is not None):
                        list_each_question.append(sttr.group(0)[17:-1].lower())
                    else:
                        list_each_question.append("N/A")

            answer_list.append(list_each_question)
        # check majority and match
        match_5,match_10,match_20,match_40 = 0,0,0,0
        for i , target in enumerate(targets):
            # Change this to test different # of paths e.g. 5, 10, 20, 40
            maxcount_5, element_having_max_freq_5 = find_most_frequent(answer_list[i][0:5],len(answer_list[i][0:5]))
            maxcount_10, element_having_max_freq_10 = find_most_frequent(answer_list[i][0:10],len(answer_list[i][0:10]))
            maxcount_20, element_having_max_freq_20 = find_most_frequent(answer_list[i][0:20],len(answer_list[i][0:20]))
            maxcount_40, element_having_max_freq_40 = find_most_frequent(answer_list[i][0:40],len(answer_list[i][0:40]))

            if(element_having_max_freq_5 == target.replace(",","").lower()):
                match_5 += 1
            if(element_having_max_freq_10 == target.replace(",","").lower()):
                match_10 += 1
            if(element_having_max_freq_20 == target.replace(",","").lower()):
                match_20 += 1
            if(element_having_max_freq_40 == target.replace(",","").lower()):
                match_40 += 1
        match_count_5.append(match_5)
        match_count_10.append(match_10)
        match_count_20.append(match_20)
        match_count_40.append(match_40)

        wandb.log({'match_count_5':sum(match_count_5),'match_count_10':sum(match_count_10),'match_count_20':sum(match_count_20),'match_count_40':sum(match_count_40),
                   '5_acc':sum(match_count_5)/sum(total_count),'10_acc':sum(match_count_10)/sum(total_count),'20_acc':sum(match_count_20)/sum(total_count),'40_acc':sum(match_count_40)/sum(total_count)})




if __name__ == '__main__':
    wandb.init(project="COT-Selection-SelfConsistency")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["gsm8k","svamp","asdiv","csqa","strategyqa","letter","babi","singleop","singleeq","esnli","sst2","obqa"],
                    help="Indicate Task Type",default="gsm8k")
    parser.add_argument("--self_consist", action="store_true",help="Use Self Consistency or not")
    args = parser.parse_args()

    print("The task is {}".format(args.task))
    print("The selected index is {}".format(selected_index))
    # print("This is for pseudo label")
    
    # For code-davinci-002
    if(args.task == "gsm8k"):
        pool_dataset = KnowledgeDataset("datasets/gsm8k_/code-davinci-002/gsm8k_style.json")
        # pool_dataset = KnowledgeDataset("datasets/gsm8k_/test_for_pseudo_pool.json")
        test_dataset = KnowledgeDataset("datasets/gsm8k_/gsm8k_test_1319.json")
    elif(args.task == "svamp"):
        pool_dataset = KnowledgeDataset("datasets/gsm8k_/code-davinci-002/rationale_pool_correct_100.json")
        test_dataset = KnowledgeDataset("datasets/SVAMP/SVAMP_test_1000.json")
    elif(args.task == "asdiv"):    
        pool_dataset = KnowledgeDataset("datasets/gsm8k_/code-davinci-002/gsm8k_pool_100.json")
        test_dataset = KnowledgeDataset("datasets/Asdiv/Asdiv_test_2096.json")
    elif(args.task == "csqa"):
        pool_dataset = KnowledgeDataset("datasets/CSQA/code-davinci-002/csqa_code_pool_100.json")
        test_dataset = KnowledgeDataset("datasets/CSQA/csqa_test_1221.json")  
    elif(args.task == "strategyqa"):
        pool_dataset = KnowledgeDataset("datasets/StrategyQA/code-davinci-002/sqa_code_pool_100.json")
        test_dataset = KnowledgeDataset("datasets/StrategyQA/sqa_test.json")
    elif(args.task == "letter"):
        pool_dataset = KnowledgeDataset("datasets/Letter/code-davinci-002/letter_code_pool_100.json")
        test_dataset = KnowledgeDataset("datasets/Letter/letter_test_500.json")   
    elif(args.task == "babi"):
        pool_dataset = KnowledgeDataset("datasets/gsm8k_/code-davinci-002/rationale_pool_correct_100.json")
        test_dataset = KnowledgeDataset("datasets/BABI/babi_q15_test_1000.json")
    elif(args.task == "singleop"):
        pool_dataset = KnowledgeDataset("datasets/gsm8k_/code-davinci-002/rationale_pool_correct_100.json")
        test_dataset = KnowledgeDataset("datasets/MAWPS/SingleOp_test_562.json")
    elif(args.task == "singleeq"):
        pool_dataset = KnowledgeDataset("datasets/gsm8k_/code-davinci-002/rationale_pool_correct_100.json")
        test_dataset = KnowledgeDataset("datasets/MAWPS/MultiArith_test_600.json")
    elif(args.task == "esnli"):
        pool_dataset = KnowledgeDataset("datasets/ESNLI/esnli_pool_100.json")
        test_dataset = KnowledgeDataset("datasets/ESNLI/esnli_test_9824.json")      
    elif(args.task == "sst2"):
        pool_dataset = KnowledgeDataset("datasets/SST2/sst2_pool_50.json")
        test_dataset = KnowledgeDataset("datasets/SST2/sst2_test_872.json") 
    elif(args.task == "obqa"):
        pool_dataset = KnowledgeDataset("datasets/OpenBookQA/obqa_pool_49.json")
        test_dataset = KnowledgeDataset("datasets/OpenBookQA/obqa_test_500.json") 


    # if(args.task == "gsm8k"):
    #     pool_dataset = KnowledgeDataset("datasets/gsm8k_/text-davinci-002/gsm8k_pool_75_zeroshot.json")
    #     test_dataset = KnowledgeDataset("datasets/gsm8k_/gsm8k_test_1319.json")
    # elif(args.task == "svamp"):
    #     pool_dataset = KnowledgeDataset("datasets/gsm8k_/text-davinci-002/rationale_pool_correct_100.json")
    #     test_dataset = KnowledgeDataset("datasets/SVAMP/SVAMP_test_1000.json")
    # elif(args.task == "asdiv"):    
    #     pool_dataset = KnowledgeDataset("datasets/gsm8k_/text-davinci-002/rationale_pool_correct_100.json")
    #     test_dataset = KnowledgeDataset("datasets/Asdiv/Asdiv_test_2096.json")
    # elif(args.task == "csqa"):
    #     pool_dataset = KnowledgeDataset("datasets/CSQA/text-davinci-002/csqa_pool_100_zeroshot.json")
    #     test_dataset = KnowledgeDataset("datasets/CSQA/csqa_test_1221.json")  
    # elif(args.task == "strategyqa"):
    #     pool_dataset = KnowledgeDataset("datasets/StrategyQA/code-davinci-002/sqa_code_pool_100.json")
    #     test_dataset = KnowledgeDataset("datasets/StrategyQA/sqa_test.json")
    # elif(args.task == "letter"):
    #     pool_dataset = KnowledgeDataset("datasets/Letter/code-davinci-002/letter_code_pool_100.json")
    #     test_dataset = KnowledgeDataset("datasets/Letter/letter_test_500.json")   
    if(args.self_consist):
        generate_result_self_consist(args = args,pool_dataset = pool_dataset,test_dataset = test_dataset,multipath=40)
    else:
        generate_result(args = args,pool_dataset = pool_dataset,test_dataset = test_dataset)
        # for key in hop_total:
        #     print("hop : {}, acc is {}".format(key,hop_count[key]/hop_total[key]))

