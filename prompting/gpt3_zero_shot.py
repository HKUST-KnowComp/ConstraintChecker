# credit: https://github.com/tqfang/ckbp2.0-comet-discos/blob/master/gpt3/eval_gpt3_zero_shot.py

import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

'''
head = 'PersonX be a horrible person'
relation = 'as a result'
tail = 'PersonY rush out'

prompt = f"Answer whether the following statement is plausible. Answer with only Yes or No: If {head}, {relation}, {tail}."
gen = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=7, temperature=0)
gen_chatgpt = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], max_tokens=7, temperature=0)
'''

CS_RELATIONS_2NL = {
    "AtLocation": "located or found at or in or on",
    "CapableOf": "is or are capable of",
    "Causes" : "causes",
    "CausesDesire": "makes someone want",
    "CreatedBy": " is created by",
    "Desires": "desires",
    "HasA": "has, possesses, or contains",
    "HasFirstSubevent": "begins with the event or action",
    "HasLastSubevent": "ends with the event or action",
    "HasPrerequisite": "to do this, one requires",
    "HasProperty": "can be characterized by being or having",
    "HasSubEvent" : "includes the event or action",
    "HinderedBy" : "can be hindered by",
    "InstanceOf" : " is an example or instance of",
    "isAfter" : "happens after",
    "isBefore" : "happens before",
    "isFilledBy" : "blank can be filled by",
    "MadeOf": "is made of",
    "MadeUpOf": "made up of",
    "MotivatedByGoal": "is a step towards accomplishing the goal",
    "NotDesires": "do not desire",
    "ObjectUse": "used for",
    "UsedFor": "used for",
    "oEffect" : "as a result, PersonY or others will",
    "oReact" : "as a result, PersonY or others feel",
    "oWant" : "as a result, PersonY or others want to",
    "PartOf" : "is a part of",
    "ReceivesAction" : "can receive or be affected by the action",
    "xAttr" : "PersonX is seen as",
    "xEffect" : "as a result, PersonX will",
    "xReact" : "as a result, PersonX feels",
    "xWant" : "as a result, PersonX wants to",
    "xNeed" : "but before, PersonX needed",
    "xIntent" : "because PersonX wanted",
    "xReason" : "because",
    "general Effect" : "as a result, other people or things will",
    "general Want" : "as a result, other people or things want to",
    "general React" : "as a result, other people or things feel",
}

import pandas as pd
from tqdm import tqdm
import time

df = pd.read_csv("data/ckbp2.0.csv")
df_selected = df.head(10)
gens = []

for i in tqdm(range(len(df_selected))):
    head = df_selected.iloc[i]["head"]
    relation = CS_RELATIONS_2NL[df_selected.iloc[i]["relation"]]
    tail = df_selected.iloc[i]["tail"]
    prompt = f"Answer whether the following statement is plausible. Answer with only Yes or No: If {head}, {relation}, {tail}."
    gens.append(openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=7, temperature=0))
    # gens_chatgpt.append(openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], max_tokens=7, temperature=0))
    time.sleep(1)

all_results = [gens[i]["choices"][0]["text"].strip() for i in range(len(df_selected))]
df_selected["gpt3"] = all_results
df_selected.to_csv("data/ckbp2.0_w_gpt3_prediction.csv", index=False)