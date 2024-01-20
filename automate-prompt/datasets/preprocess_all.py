import jsonlines
import openai
import re
import time
import sys
import json
import argparse

# TODO : combine other preprocess scripts to this file
prompt = "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.\n\n"

# MAWPS
testset_mawps_singleop = []
testset_answer_mawps_singleop = []

with open("./MAWPS/SingleOp/mawps_singleop_stream_inputs.txt") as f:
    for line in f.readlines(): 
        testset_mawps_singleop.append(line[:-1].replace("\\n","\n"))
with open("./MAWPS/SingleOp/mawps_singleop_stream_targets.txt") as f:
    for line in f.readlines(): 
        testset_answer_mawps_singleop.append(line[:-1].replace("\\n","\n"))

for i in range(0,len(testset_mawps_singleop)):
    testset_mawps_singleop[i] = testset_mawps_singleop[i][2450:]

testset_mawps_singleeq = []
testset_answer_mawps_singleeq = []

with open("./MAWPS/SingleEq/mawps_singleeq_stream_inputs.txt") as f:
    for line in f.readlines(): 
        testset_mawps_singleeq.append(line[:-1].replace("\\n","\n"))
with open("./MAWPS/SingleEq/mawps_singleeq_stream_targets.txt") as f:
    for line in f.readlines(): 
        testset_answer_mawps_singleeq.append(line[:-1].replace("\\n","\n"))

for i in range(0,len(testset_mawps_singleeq)):
    testset_mawps_singleeq[i] = testset_mawps_singleeq[i][2450:]

testset_mawps_multiarith = []
testset_answer_mawps_multiarith = []

with open("./MAWPS/MultiArith/mawps_multiarith_stream_inputs.txt") as f:
    for line in f.readlines(): 
        testset_mawps_multiarith.append(line[:-1].replace("\\n","\n"))
with open("./MAWPS/MultiArith/mawps_multiarith_stream_targets.txt") as f:
    for line in f.readlines(): 
        testset_answer_mawps_multiarith.append(line[:-1].replace("\\n","\n"))

for i in range(0,len(testset_mawps_multiarith)):
    testset_mawps_multiarith[i] = testset_mawps_multiarith[i][2450:]

testset_mawps_addsub = []
testset_answer_mawps_addsub = []

with open("./MAWPS/AddSub/mawps_addsub_stream_inputs.txt") as f:
    for line in f.readlines(): 
        testset_mawps_addsub.append(line[:-1].replace("\\n","\n"))
with open("./MAWPS/AddSub/mawps_addsub_stream_targets.txt") as f:
    for line in f.readlines(): 
        testset_answer_mawps_addsub.append(line[:-1].replace("\\n","\n"))

for i in range(0,len(testset_mawps_addsub)):
    testset_mawps_addsub[i] = testset_mawps_addsub[i][2450:]

# 562 for singleop, 508 for singleEq, 395 for AddSub, 600 for Multi
def generate_testset_for_MAWPS():
    return_list_singleop, return_list_singleeq, return_list_multiarith, return_list_addsub = [[] for x in range(4)]
    # For SingleOp
    for index in range(0,len(testset_mawps_singleop)):
        temp = {}
        temp['Question'] = testset_mawps_singleop[index]
        temp["Rationale"] ,temp["Answer"] = "N/A", "N/A"
        temp["Ground_truth"] = testset_answer_mawps_singleop[index]
        return_list_singleop.append(temp)
    # For SingleEq
    for index in range(0,len(testset_mawps_singleeq)):
        temp = {}
        temp['Question'] = testset_mawps_singleeq[index]
        temp["Rationale"] ,temp["Answer"] = "N/A", "N/A"
        temp["Ground_truth"] = testset_answer_mawps_singleeq[index]
        return_list_singleeq.append(temp)  
    # For MultiArith
    for index in range(0,len(testset_mawps_multiarith)):
        temp = {}
        temp['Question'] = testset_mawps_multiarith[index]
        temp["Rationale"] ,temp["Answer"] = "N/A", "N/A"
        temp["Ground_truth"] = testset_answer_mawps_multiarith[index]
        return_list_multiarith.append(temp)
    # For AddSub
    for index in range(0,len(testset_mawps_addsub)):
        temp = {}
        temp['Question'] = testset_mawps_addsub[index]
        temp["Rationale"] ,temp["Answer"] = "N/A", "N/A"
        temp["Ground_truth"] = testset_answer_mawps_addsub[index]
        return_list_addsub.append(temp)
    assert (len(return_list_singleop),len(return_list_singleeq),len(return_list_multiarith),len(return_list_addsub)) == (562,508,600,395)

    return return_list_singleop,return_list_singleeq,return_list_multiarith,return_list_addsub




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--singleop", action="store_true",help="Preprocess for MAWPS(SingleOp)")
    parser.add_argument("--singleeq", action="store_true",help="Preprocess for MAWPS(SingleEq)")
    parser.add_argument("--multiarith", action="store_true",help="Preprocess for MAWPS(MultiArith)")
    parser.add_argument("--addsub", action="store_true",help="Preprocess for MAWPS(AddSub)")
    args = parser.parse_args()

    return_list_singleop,return_list_singleeq,return_list_multiarith,return_list_addsub = generate_testset_for_MAWPS()
    if(args.singleop):
        with open("./MAWPS/SingleOp_test_562.json","w") as f:
            json.dump(return_list_singleop,f,indent=2)
    if(args.singleeq):
        with open("./MAWPS/SingleEq_test_508.json","w") as f:
            json.dump(return_list_singleeq,f,indent=2)   
    if(args.multiarith):
        with open("./MAWPS/MultiArith_test_600.json","w") as f:
            json.dump(return_list_multiarith,f,indent=2)
    if(args.addsub):
        with open("./MAWPS/AddSub_test_395.json","w") as f:
            json.dump(return_list_addsub,f,indent=2)