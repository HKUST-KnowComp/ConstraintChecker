# pwd = /data/vqdo/projects/cctl_cskb_pop/
# CUDA_VISIBLE_DEVICES=4,5,6,7 python
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

# load model and tokenizer
def load_llama7b_model(config):
    max_shard_size = '3GB'
    weights_location = f'../llama/llama-2-{config}-sharded-{max_shard_size}'
    # from accelerate import Accelerator
    # accelerator = Accelerator()
    # accelerator.save_model(model, weights_location, max_shard_size=max_shard_size)
    with init_empty_weights():
        model = LlamaForCausalLM(LlamaConfig())
    device_map = infer_auto_device_map(model,
        max_memory={0: "6GiB", 1: "8GiB", 2: "8GiB", 3: "8GiB"},
        no_split_module_classes=['LlamaDecoderLayer', 'LlamaAttention', 'LlamaMLP']
    )
    model = load_checkpoint_and_dispatch(model,
        checkpoint=weights_location, device_map="auto",
        no_split_module_classes=['LlamaDecoderLayer']
    )
    return model


model = load_llama7b_model('7b-chat')
tokenizer = AutoTokenizer.from_pretrained('../llama/llama-2-7b-chat-hf')
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"
prompt_template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant who help users to solve a language task in a concise manner.
<</SYS>>

{} [/INST]""" #.format(message)

# define generation function
# https://github.com/facebookresearch/llama/issues/435
def generate_full_prompt(message):
    template = "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]" #.format(system_message, message)
    system_message = \
        "You are a helpful, respectful and honest assistant who help users to solve a language task."
    return template.format(system_message, message)

# https://huggingface.co/docs/transformers/main_classes/text_generation
def llama2_generate(messages, max_tokens=20, temperature=0, use_system_prompt=True):
    prompts = [prompt_template.format(m) if use_system_prompt else f"<s>[INST] {m} [/INST]" for m in messages]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(0)
    generate_ids = model.generate(inputs.input_ids,
        max_new_tokens=max_tokens, # max_length=30
        temperature=temperature
    )
    outputs = tokenizer.batch_decode(generate_ids[:, inputs.input_ids.shape[-1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    # outputs = [full[len(pre):] for full, pre in zip(full_outputs, prompts)]
    return outputs

messages = ["Hey, are you conscious? Can you talk to me?", "Assume that you can be hero, who do you want to be?"]
print(llama2_generate(messages))









# run evaluation
import pandas as pd
tst_data_df = pd.read_csv('prompting_data/tst_data_to_eval.csv')
tst_data = tst_data_df['assertion'].tolist()

tst_kate_exemplars = pd.read_csv('prompting_data/tst_kate_exemplars_top5.csv')
tst_kate_exemplars_tq = pd.read_csv('prompting_data/tst_kate_exemplars_tq_top5.csv')
tst_kate_same_relation_exemplars = pd.read_csv('prompting_data/tst_kate_same_relation_exemplars_top5.csv')
tst_kate_same_relation_exemplars_tq = pd.read_csv('prompting_data/tst_kate_same_relation_exemplars_tq_top5.csv')
tst_random_exemplars = pd.read_csv('prompting_data/tst_random_5exemplars.csv')
print(tst_data_df.columns, len(tst_data), len(tst_kate_exemplars))

bs = 2
total = (len(tst_data_df)-1)//bs+1

def llama_loop(customize_prompt, max_tokens, use_system_prompt=1,
        filename_suffix='', resume_at_step=0, end_at_step=total,
    ):
    for i in range(resume_at_step, end_at_step):
        print(i, '/', total)
        p = llama2_generate(
            messages=[customize_prompt(s)
                for s in range(i*bs, min((i+1)*bs, len(tst_data_df)))],
            max_tokens=max_tokens,
            temperature=0,
            use_system_prompt=use_system_prompt
        )
        temp = '\n'.join([x.replace('\n','--') for x in p])
        print(temp)
        output = open(f'results/llama2/tst_llama2-7b_{filename_suffix}.txt', 'a+', encoding='utf-8')
        output.write(f'{i}/{total}\n{temp}\n'.encode().decode('utf-8'))
        output.close()

# llama2 doesn't follow the instruction @@ https://github.com/facebookresearch/llama/issues/435
# may change back to 7b model instead of 7b-chat -> 7b don't follow the instruction well,
#   given the system prompt or not. have to use 7b-chat
# or try the suggested system prompt first -> Yep, very simple, and it works!

'''
python prompting/scoring.py -s tst -q 0 \
  -p tst_llama2-7b_naive_prompt3.txt
'''


# only use prompt design 3
# naive
judge_prompt_1 = "Judge the following statement if it's likely to occur, only answer 'True' or 'False':\n{}"
naive_prompt3 = lambda s:judge_prompt_1.format(tst_data_df['assertion'][s])
for i in range(5):
    print(naive_prompt3(i), '\n', llama2_generate([naive_prompt3(i)], use_system_prompt=1))

llama_loop(naive_prompt3, max_tokens=10, use_system_prompt=1,
    filename_suffix='naive_prompt3', resume_at_step=0)


# random examplar
''' # adding question for each examplar doesn't change the output much!
random_5shot_prompt3 = lambda s:'\n\n'.join(
    [judge_prompt_1.format('Statement: {}\nAnswer: {}'.format(tst_random_exemplars[f'{_}'][s], 'True' if tst_random_exemplars[f'{_}_label'][s] else 'False')) for _ in range(5, 10)] +
    [judge_prompt_1.format('Statement: {}\nAnswer: {}'.format(tst_data_df['assertion'][s], ''))]
)
'''
random_5shot_prompt3 = lambda s:judge_prompt_1.format('\n\n'.join(
    ['Statement: {}\nAnswer: {}'.format(tst_random_exemplars[f'{_}'][s], 'True' if tst_random_exemplars[f'{_}_label'][s] else 'False') for _ in range(5, 10)] +
    ['Statement: {}\nAnswer: {}'.format(tst_data_df['assertion'][s], '')]
))
print(random_5shot_prompt3(0))
for i in range(5):
    print(llama2_generate([random_5shot_prompt3(i)], use_system_prompt=0))

llama_loop(random_5shot_prompt3, max_tokens=5, use_system_prompt=0,
    filename_suffix='random_5shot_prompt3', resume_at_step=0)


# kate
kate_5shot_prompt3 = lambda s:judge_prompt_1.format('\n\n'.join(
    ['Statement: {}\nAnswer: {}'.format(tst_kate_exemplars[f'{_}'][s], 'True' if tst_kate_exemplars[f'{_}_label'][s] else 'False') for _ in [4,3,2,1,0]] +
    ['Statement: {}\nAnswer: {}'.format(tst_data_df['assertion'][s], '')]
))
print(kate_5shot_prompt3(0))
for i in range(5):
    print(llama2_generate([kate_5shot_prompt3(i)], use_system_prompt=0))

llama_loop(kate_5shot_prompt3, max_tokens=5, use_system_prompt=0,
    filename_suffix='kate_5shot_prompt3', resume_at_step=0)


# kate-s
kate_5shot_same_relation_prompt3 = lambda s:judge_prompt_1.format('\n\n'.join(
    ['Statement: {}\nAnswer: {}'.format(tst_kate_same_relation_exemplars[f'{_}'][s], 'True' if tst_kate_same_relation_exemplars[f'{_}_label'][s] else 'False') for _ in [4,3,2,1,0]] +
    ['Statement: {}\nAnswer: {}'.format(tst_data_df['assertion'][s], '')]
))
print(kate_5shot_same_relation_prompt3(0))
for i in range(5):
    print(llama2_generate([kate_5shot_same_relation_prompt3(i)], use_system_prompt=0))

llama_loop(kate_5shot_same_relation_prompt3, max_tokens=5, use_system_prompt=0,
    filename_suffix='kate_5shot_same_relation_prompt3', resume_at_step=0)


# zeroshot
zeroshotCoT_prompt1 = "Judge the statement '{}' if it's likely to occur. Let's think step by step, then conclude by answering 'True' or 'False'." # if add only, it will strictly follow that command, only produce 'True' or 'False' w/o CoT
zeroshot_prompt3 = lambda s:zeroshotCoT_prompt1.format(tst_data_df['assertion'][s])
print(zeroshot_prompt3(0))
for i in range(2):
    print(llama2_generate([zeroshot_prompt3(i)], max_tokens=600, use_system_prompt=0))

llama_loop(zeroshot_prompt3, max_tokens=600, use_system_prompt=0,
    filename_suffix='zeroshot_prompt3', resume_at_step=0)

# randomcot
cot_examplars_3 = """Please answer the last question.

Question: Judge the following statement if it's likely to occur: PersonX occupy PersonY position, thus, PersonX want PersonY want to aid in position.
Answer: Let's think step by step. When PersonX occupy PersonY position, it means PersonY already worked at this position and has experience to do the job. Therefore, it's likely that PersonX want PersonY to aid PerosonX when PersonX is in that job position. Thus, the statement is likely to occur.

Question: Judge the following statement if it's likely to occur: PersonX see that, thus as an result, PersonX want a pet.
Answer: Let's think step by step. In this context, we can refer the word 'that' as some activity where people play with their pet. Therefore, it stimulates PersonX's desire to have a pet. Thus, the statement is likely to occur.

Question: Judge the following statement if it's likely to occur: {}.
Answer: Let's think step by step."""
randomcot_prompt3 = lambda s:cot_examplars_3.format(tst_data_df['assertion'][s])

for i in range(1):
    print(tst_data_df['assertion'][i])
    print(prompt_template.format(randomcot_prompt3(i)))

for i in range(1):
    print(llama2_generate([randomcot_prompt3(i)], max_tokens=100, use_system_prompt=0))
    # shouldn't use the system prompt

llama_loop(randomcot_prompt3, max_tokens=100, use_system_prompt=0,
    filename_suffix='randomcot_prompt3', resume_at_step=0)







# typing constraint
idx_q1 = tst_data_df['relation'].isin(['xReact', 'oReact', 'xAttr'])
temp_df = tst_data_df[idx_q1]
temp_df.reset_index(inplace=True)
total = (len(temp_df)-1)//bs+1
# temp_df

'''
greed search (use or not use system prompt) and modified moe prompt,
choose not use system prompt and new moe prompt
because of the least verbalization (still long answer) and best correctness
LLaMa-2 has very different behaviors comparing to ChatGPT
Don't use "Answer the choice only"
'''
moe_q1_prompt1 = "Which aspect of the subject does the clause '{}' express? Choose one of the following: 1. action, 2. persona, 3. mental state."
# moe_q1_prompt1 = "Which aspect (among three options 1. event/activity, 2. persona, 3. mental state) of the subject does the clause '{}' express."
i = 3
prompt = moe_q1_prompt1.format(temp_df['tail'][i])
print(prompt)
for i in range(5):
    print(moe_q1_prompt1.format(temp_df['tail'][i]), '\n',
        llama2_generate([moe_q1_prompt1.format(temp_df['tail'][i])], max_tokens=50, use_system_prompt=0))

for i in range(0, total):
    print(i, '/', total)
    p = llama2_generate(
        messages=[moe_q1_prompt1.format(s)
            for s in temp_df['tail'][i*bs:(i+1)*bs]],
        max_tokens=50,
        temperature=0,
        use_system_prompt=0
    )
    temp = '\n'.join([x.replace('\n','--') for x in p])
    print(temp)
    output = open(f'results/llama2/tst_llama2-7b_moe_q1_prompt1.txt', 'a+', encoding='utf-8')
    output.write(f'{i}/{total}\n{temp}\n'.encode().decode('utf-8'))
    output.close()


# temporal constraint
idx_q3 = tst_data_df['relation'].isin(['xIntent', 'xNeed'])
temp_df = tst_data_df[idx_q3]
temp_df.reset_index(inplace=True)
total = (len(temp_df)-1)//bs+1
# temp_df

moe_q3_prompt3 = """Which one of the following two statements is more plausible:
0. {0} before {1}
1. {0} after {1}
Answer 0 or 1 only.""" # tail [] head, 1 -> after, 0 -> before
i = 0
prompt = moe_q3_prompt3.format(temp_df['tail'][i], temp_df['head'][i])
print(prompt)
for i in range(5):
    print(moe_q3_prompt3.format(temp_df['tail'][i], temp_df['head'][i]), '\n',
        llama2_generate([moe_q3_prompt3.format(temp_df['tail'][i], temp_df['head'][i])],
            max_tokens=50, use_system_prompt=0)
        )

for i in range(0, total):
    print(i, '/', total)
    p = llama2_generate(
        messages=[moe_q3_prompt3.format(temp_df['tail'][s], temp_df['head'][s])
            for s in range(i*bs, min((i+1)*bs, len(temp_df)))],
        max_tokens=50,
        temperature=0,
        use_system_prompt=0
    )
    temp = '\n'.join([x.replace('\n','--') for x in p])
    print(temp)
    output = open(f'results/llama2/tst_llama2-7b_moe_q3_prompt3.txt', 'a+', encoding='utf-8')
    output.write(f'{i}/{total}\n{temp}\n'.encode().decode('utf-8'))
    output.close()