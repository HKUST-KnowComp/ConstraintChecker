import openai, os
import time, asyncio
import pandas as pd


async def dispatch_openai_requests(
    messages_list,
    max_tokens=4,
    ):
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.

    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model='gpt-3.5-turbo',
            messages=x,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
        )
        for x in messages_list
    ]
    time.sleep(1)
    return await asyncio.gather(*async_responses)


openai.api_key = os.getenv("OPENAI_API_KEY")
model = 'davinci-abstract' # gpt3.5-turbo or davinci, append with batch if passing multiple sample into a prompt
split = 'tst'
path = f'results/prompting_{split}'
filename = 'dev_data_to_tune' if split == 'dev' else 'tst_data_to_eval'

prompts = {
    'template0' : "Answer whether the following statement is plausible. Answer with only Yes or No:\n{}",
    'template1': "Judge the following statement if it's likely to occur, only answer 'True' or 'False':\n{}",
    'template4': "Judge the following statements if they are likely to occur, only list the question number and answer 'True' or 'False' for each:\n{}",
    'concept_template1': """Please read the convention, task description, examples, then do the task:
Convention: There are three dimensions ['persona', 'event', 'mental state']. All ordered dimension pairs are valid, except ('event', 'persona') and ('event', 'mental state') are invalid.

Task description: Given an inference and its corresponding dimension D_i and its two main clauses H and T, you are required to do the following step by step:
1. Focus on the last clause T, determine which dimension best describe what the clause expresses. Denote the dimension as D_c.
2. Judge if the pair (D_c, D_i) is valid. Answer 'Yes' or 'No'.
3. Judge if the main clauses' meaning is all clear/non-ambiguous. Answer 'Yes' or 'No'.
4. Based on result in 2, 3, and the semantics of the inference, judge if the inference is likely 'True' or 'False'. Note that if result of either 2 or 3 is 'No', then result of 4 should be 'False'.

Examples:
* Example 1
Inference: PersonX learns to play the trumpet, thus, PersonY wants to applaud PersonX's talent
D_i = 'event', H = 'PersonX learns to play the trumpet', T = 'PersonY wants to applaud PersonX's talent'
1. D_c = 'event'
2. Yes (because the ordered pair (D_c, D_i) = ('event', 'event') is valid)
3. Yes
4. True. Because 1) "PersonX learns to play the trumpet" means PersonX are talent and effortful in learning musical instrument, and 2) people tend to applaud other people's effort and talent.

* Example 2
Inference: PersonX eat most of those, thus it can be seen about PersonX's intention that, PersonX love curly fries
D_i = 'event', H = 'PersonX eat most of those', T = 'PersonX love curly fries'
1. D_c = 'mental state'
2. Yes (because the ordered pair (D_c, D_i) = ('mental state', 'event') is valid)
3. Yes
4. False. Because 'love curly fries' expresses PersonX's preference rather than intention.

* Example 3
Inference: PersonX do not feel well, thus as a result on PersonX's emotion, PersonX only make it through half.
D_i = 'mental state', H = 'PersonX do not feel well', T = 'PersonX only make it through half'
1. D_c = 'event'
2. No (because the ordered pair (D_c, D_i) = ('event', 'mental state') is not valid)
3. Yes
4. False. Because, result of 2 is No.

* Example 4 
Inference: Before PersonX really empathize with PersonY, PersonX feel.
D_i = 'event', H = 'PersonX really empathize with PersonY', T = 'PersonX feel'
1. D_c = 'mental state'
2. Yes (because the ordered pair (D_c, D_i) = ('mental state', 'event') is valid)
3. No (T is unclear about what 'PersonX feel')
4. False. Because result of 3 is No.

* Example 5
Inference: PersonX have a traumatic incident, thus it can be seen about PersonX's attribute that PersonX be introvert
D_i = 'persona', H = 'PersonX have a traumatic incident', T = 'PersonX be introvert'
1. D_c = 'persona'
2. Yes (because the ordered pair (D_c, D_i) = ('persona', 'persona') is valid)
3. Yes
4. False. Because having a traumatic incident likely result from unfortune or carelessness, while being introvert is not relevant.

Now, please do your task for (each of) the following statement(s):
Inference: {}
D_i = '{}', H = '{}', T = '{}'""",
    'concept_template2': """Please read the task description and examples, then do the task:
Task description: Given an inference, its aspect D, and its two main clauses H and T, you are required to do the following step by step:
1. Determine if clause T expresses about a/an D of the subject. Answer 'Yes' or 'No'.
2. Judge if the meanings of H and T are all clear/non-ambiguous. Answer 'Yes' or 'No'.
3. If result of either 1 or 2 is 'No', please answer 'False'. Otherwise, judge if the inference is likely 'True' or 'False'.

Examples:
* Example 1
Inference: PersonX learns to play the trumpet, thus, PersonY wants to applaud PersonX's talent
D = 'event', H = 'PersonX learns to play the trumpet', T = 'PersonY wants to applaud PersonX's talent'
1. Yes. Because 'PersonY wants to applaud PersonX's talent' expresses an 'event' of PersonY
2. Yes.
3. True. Because 1) "PersonX learns to play the trumpet" means PersonX are talent and effortful in learning musical instrument, and 2) people tend to applaud other people's effort and talent.

* Example 2
Inference: PersonX eat most of those, thus as a result on PersonX's emotion, PersonX feel hungry
D = 'mental state', H = 'PersonX eat most of those', T = 'PersonX feel hungry'
1. Yes. Because 'PersonX feel hungry' expresses a 'mental state' of PersonX
2. Yes.
3. False. Because it's unlikely that PersonX still feel hungry after eat a lot.

* Example 3
Inference: PersonX do not feel well, thus as a result on PersonX's emotion, PersonX only make it through half.
D = 'mental state', H = 'PersonX do not feel well', T = 'PersonX only make it through half'
1. No. Because 'PersonX only make it through half' doesn't expresses an 'mental state' of PersonX
2. Yes.
3. False. Because, result of 1 is No.

* Example 4 
Inference: Before PersonX really empathize with PersonY, PersonX feel.
D = 'event', H = 'PersonX really empathize with PersonY', T = 'PersonX feel'
1. Yes. Because 'PersonX feel' expresses an 'event' of PersonX
2. No. Because the meaning of 'PersonX feel' is unclear/ambiguous
3. False. Because result of 2 is No.

* Example 5
Inference: PersonX have a traumatic incident, thus it can be seen about PersonX's attribute that PersonX be introvert
D = 'persona', H = 'PersonX have a traumatic incident', T = 'PersonX be introvert'
1. Yes. Because 'PersonX be introvert' expresses a persona of PersonX
2. Yes.
3. False. Because having a traumatic incident likely result from unfortune or carelessness, while being introvert is not relevant.

Now, please do your task for (each of) the following statement(s):
Inference: {}
D = '{}', H = '{}', T = '{}'""",
}
prompt_choice = 'concept_template_5exemplars'
result_filename = f'{split}_{model}_{prompt_choice}'

tst_data_df = pd.read_csv(f'prompting_data/{filename}.csv')
tst_data = tst_data_df['assertion'].tolist()
print(len(tst_data))
predictions = []

output = open(f'{path}/output.txt', 'a+')
output.write(f'\n\n#####\nFilename: {filename}, Model: {model}\nPrompt: {prompts[prompt_choice]}\n\n')
output.close()


if model == 'gpt3.5-turbo':
    bs = 50
    for i in range(0, (len(tst_data)-1)//bs+1):
        while True:
            try:
                print(i)
                pred = dispatch_openai_requests(
                    messages_list = [[{"role": "user", "content": prompts['template1'].format(s)}] for s in tst_data[i*bs:(i+1)*bs]],
                    max_tokens=1,
                )
                temp = ', '.join([x['choices'][0]['message']['content'] for x in pred])
                print(temp)
                output = open(f'{path}/output.txt', 'a+')
                output.write(f'{i}\n{temp}\n')
                output.close()
                predictions.extend([x['choices'][0]['message']['content'] for x in pred])
                time.sleep(20)
                break
            except Exception as e:
                print(e)
                time.sleep(120)

elif model == 'davinci': # template 0 is better for template 0
    bs = 20
    total_iter = (len(tst_data)-1)//bs + 1
    for i in range(0, total_iter):
        while True:
            try:
                print(i, '/', total_iter)
                p = openai.Completion.create(
                    model="text-davinci-003", 
                    prompt=[prompts['template0'].format(x) for x in tst_data[i*bs:(i+1)*bs]],
                    max_tokens=20,
                    temperature=0
                )
                temp = ', '.join([x['text'].replace('\n','-') for x in p['choices']])
                print(temp)
                output = open(f'{path}/output.txt', 'a+')
                output.write(f'{i}\n{temp}\n')
                output.close()
                predictions.extend([x['text'].replace('\n','-') for x in p['choices']])
                time.sleep(5)
                break
            except Exception as e:
                print(e)
                time.sleep(60)

elif model == 'davinci-batch':
    bs = 20
    total_iter = (len(tst_data)-1)//bs + 1
    for i in range(0, total_iter):
        while True:
            try:
                print(i, '/', total_iter)
                p = openai.Completion.create(
                    model="text-davinci-003", 
                    prompt=prompts['template4'].format('\n'.join([f'{j}. {x}' for j, x in enumerate(tst_data[i*bs:(i+1)*bs])])),
                    max_tokens=20*bs,
                    temperature=0
                )
                temp = p['choices'][0]['text'].replace('\n', '-')
                print(temp)
                output = open(f'{path}/output.txt', 'a+')
                output.write(f'{i}\n{temp}\n')
                output.close()
                predictions.append(temp)
                time.sleep(5)
                break
            except Exception as e:
                print(e)
                time.sleep(60)

elif model == 'davinci-abstract':
    bs = 20
    total_iter = (len(tst_data)-1)//bs + 1
    for i in range(0, total_iter):
        while True:
            try:
                print(i, '/', total_iter)
                p = openai.Completion.create(
                    model="text-davinci-003", 
                    prompt=[prompts['concept_template_5exemplars'].format(
                        tst_data_df['assertion'][s],
                        tst_data_df['dim'][s],
                        tst_data_df['head'][s],
                        tst_data_df['tail'][s])
                        for s in range(i*bs, min((i+1)*bs, len(tst_data_df)))
                    ],
                    max_tokens=100,
                    temperature=0
                )
                temp = '\n'.join([x['text'].replace('\n','--') for x in p['choices']])
                print(temp)
                output = open(f'{path}/output.txt', 'a+')
                output.write(f'{i}\n{temp}\n')
                output.close()
                predictions.extend([x['text'].replace('\n','--') for x in p['choices']])
                time.sleep(5)
                break
            except Exception as e:
                print(e)
                time.sleep(60)

else:
    raise NotImplementedError

with open(f'{path}/{result_filename}.txt', 'w') as fout:
    fout.write('\n'.join(predictions))
