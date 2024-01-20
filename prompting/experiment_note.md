

# MoE: more efficient, prompt tuning step by step

question 1: type of tail event
  aim: better detect dim-mismatched tail
  requirement: not introduce the case that e.g tail is mental state (for xReact), but pred is event hence make the final decision fails!

  Test question: does tail event NOT comply the relation dim. Pos or Neg.
  precision is more important than recall when we want to prove that the method only aim to solve the case tail = event, rather than use it to spam the judgement as 'negative'. If precision is low, people will argue that better result may be because plausibility portion in these relation is low.
    precision is high -> only cure the part that tail event is 'event'. Not touch the part that tail event is 'mental state' or 'persona', leave it to judgment component

  
  labels[i] != 1 and predictions[i] == 1: # judge if this question introduce false positive
  using it, we achieve 1.0000|0.3333|0.5000

  errors need to be solved: 44, 48, 52, 55, 57 
  BUT do it later, as we don't need a perfect tool


question 2:
  notice that we ask tail if clear, ask head if ambiguous
  hmm, both questions yield more 'No' than 'Yes'!
    is there a case that a statement is neither clear nor ambiguous?

  ask again if tail is ambiguous

  this filter seems not work

next time, just run the filter for a specific relation
  don't have to run for the whole set

MoE works!

! but the last two guidances seem not helpful, because chatgpt performs quite well on the relevant relation
  the error is not related to the guildance!
  but we can hack :)) tune on the test file!

  look at HinderedBy tst error, if there is any time-order error

```
moe_q1_prompt1 = "Which aspect (among three options 1. event/activity, 2. persona, 3. mental state) of the subject does the clause '{}' express. Answer the choice only."
moe_q2_prompt1 = "Judge if the meaning of the clause '{}' is ambiguous. Answer 'Yes' or 'No' only."
moe_q3_prompt1 = "Judge if the event '{}' likely occurs after the event '{}'. Answer 'Yes' or 'No' only."    # pretty unstable
moe_q3_prompt1_reverse = "Judge if the event '{}' likely occurs before the event '{}'. Answer 'Yes' or 'No' only."   # pretty unstable
moe_q3_prompt2 = "Given 2 events in a story: A. '{}', B. '{}'. Is A likely to occur after B? Answer 'Yes' or 'No' only."
moe_q3_prompt3 = """Which one of the following two statements is more plausible:
0. {0} before {1}
1. {0} after {1}
Answer 0 or 1 only.""" # tail [] head, 1 -> after, 0 -> before

moe_q4_prompt1 = "Judge if the event '{}' causes the event '{}'. Answer 'Yes' or 'No' only." # need to observe on dev set. Not easy to judge, thus not System 1 thinking
```

### MoE for xIntent/xNeed

row in dev_data_tune.csv (with header)
62 x
63 (not intention) -> may help
67 (seem the ground-truth label is correct)
69 ?? (prompt affected) -> the only sample we can 'cure' using xIntent/xNeed filter
73 x (ambiguous filter can help, but global effect)
74 (ambiguous, cannot give 1, may be get sick of that smell)  (ambiguous filter can help, but global effect)

It works too!!!!!!!!!!!!

Easier to do analysis!


### MoE for HinderedBy

Since there is no error in dev set (wow!), I have to do error analysis on test set. Just look at False Positive instances. Tend to apply same constraint as xNeed, as tail event of HinderedBy should occur before head event

row in tst_data... (= index + 2)
360 ?
364 ?
374 ?
389 x
392 ?
395 ?
398 ?
419 ?
421 ?
426 x
427 ?
435 ?
437 ?
438 x
449 ?
543 -> may help
546 ?
932 -> may help
934 x sound sensical, but PersonZ is not related
935 duplication
939 duplication
956 x
968 x

? nonsense. I am kinda surprised when ChatGPT can judge them as positive
x argurable

also can try 'Causal' constraint for HinderedBy


### Implicit constraint in relation

~ ~~check ambiguity for all~~ -> decide to exclude it! It's hard to define then do analysis, also not be directly derived from relation. In fact, it doesn't help, so better not spare time to write about it!
~ check extra constraint (for tail type, only need to check with 3 relations). Extra constraint should be derived from relations (rather 'you look at data and come up with') and shouldn't be explicitly questioned the jugdment prompt. Only check 'temporal'.
xWant:
oWant: 
xEffect:
oEffect:
xReact: 1. tail's type
oReact: 1. tail's type
xAttr: 1. tail's type
xIntent: 2. temporal
xNeed: 2. temporal
Causes: 
xReason:
isBefore:
isAfter:
HinderedBy:
HasSubEvent: 

primitive info: temporal, causal, event's type
  seem to only have these 3 types
  DONE: literature review. no relation of two events other than Temporal and Causal is mentioned
    actually we have Entailment. But Entailment seems to be general, i.e not primitive enough
  
  **so we finalize the set of constraint we can set, which are 'temporal', 'causal', 'tail type'**. But need to remove 'causal' as causal is not easy to judge, thus not system 1
    but using causal filter or not depends on you!
    for other two constraints 'ambiguous' and 'similar head and tail', they are kinda irrelevant of relation's implicit constraint.
    if you want to use, please only use 'ambiguity' -> cite Ernest Davis's paper!

  may be for each constraint, we still need to run all, then do the error analysis (reverse of what you actually do :v)
    ya, xWant also has a nuance of Causal. But we need to modify the tail a bit! Hmm, also, xWant may be n-hop (X -> xWant -> Y = X -> causes -> X1 -> leads to -> Y), that means if we ask if X -> cause -> Y (or simplifed version of Y i.e remove 'want to'), that may not be true -> affect True Positive instance -> bad for us, bas for F1 score
    Haizz, but I don't know what's the data distribution. Please run all!
    For Causal, it's not System 1. It's a complex relation.

MAVEN-ERE: A Unified Large-scale Dataset for Event Coreference, Temporal, Causal, and Subevent Relation Extraction

### For persuasiveness

To prove the effectiveness even with a larger eval dataset, run on the full ckbp evaluation set!

Also run on text-davinci-003 and see!

Efficient, as only run filter for related instances


# Think about the name of the paper

Can annotation guideline help you design a better prompt? A study on a commonsense benchmark?

Mixture of Constraints: How explicit annotation rules benefit LLM prompting


# Type-constraint


## do error analysis on dev_davinci-abstract_concept_template2.txt




## do error analysis on gpt3.5-turbo_conceptualization_template2_5exemplars.txt

look at instance of relation xReact, oReact, xAttr (those which need the 'rule' (this step) the most)
Error in step 1:
  20/27 is correct

Error in step 2:
  :)) hard to judge, but at least chatgpt point out some non-sense/ambiguous assertion

Also need to look at other relations
  error in step 1: 18/33 is correct -> hmmm. 5 over 15 incorrect classification leads to answer False -> hmmm


## do error analysis on gpt3.5-turbo_conceptualization_5exemplars.txt

prompt: conceptualization_prompt_chatgpt = """Please read the convention, task description, examples, then do the task:
Convention: There are three dimensions ['persona', 'event', 'mental state'].

Task description: Given an inference and its corresponding dimension D_i and its two main clauses H and T, you are required to do the following step by step:
1. Focus on the last clause T, determine which dimension best describe what the clause expresses. Denote the dimension as D_c.
2. If the ordered pair (D_c, D_i) is ('event', 'persona') or ('event', 'mental state') then write 'No', otherwise write 'Yes'.
3. Judge if the main clauses' meaning is all clear/non-ambiguous. Answer 'Yes' or 'No'.
4. Based on result in 2, 3, and the semantics of the inference, judge if the inference is likely 'True' or 'False'. Note that if result of either 2 or 3 is 'No', then result of 4 should be 'False'.

Examples:
* Example 1
Inference: PersonX learns to play the trumpet, thus, PersonY wants to applaud PersonX's talent
D_i = 'event', H = 'PersonX learns to play the trumpet', T = 'PersonY wants to applaud PersonX's talent'
1. D_c = 'event'
2. Yes
3. Yes
4. True. Because 1) "PersonX learns to play the trumpet" means PersonX are talent and effortful in learning musical instrument, and 2) people tend to applaud other people's effort and talent.

* Example 2
Inference: PersonX eat most of those, thus it can be seen about PersonX's intention that, PersonX love curly fries
D_i = 'event', H = 'PersonX eat most of those', T = 'PersonX love curly fries'
1. D_c = 'mental state'
2. Yes
3. Yes
4. False. Because 'love curly fries' expresses PersonX's preference rather than intention.

* Example 3
Inference: PersonX do not feel well, thus as a result on PersonX's emotion, PersonX only make it through half.
D_i = 'mental state', H = 'PersonX do not feel well', T = 'PersonX only make it through half'
1. D_c = 'event'
2. No
3. Yes
4. False. Because, result of 2 is No.

* Example 4 
Inference: Before PersonX really empathize with PersonY, PersonX feel.
D_i = 'event', H = 'PersonX really empathize with PersonY', T = 'PersonX feel'
1. D_c = 'mental state'
2. Yes
3. No (T is unclear about what 'PersonX feel')
4. False. Because result of 3 is No.

* Example 5
Inference: PersonX have a traumatic incident, thus it can be seen about PersonX's attribute that PersonX be introvert
D_i = 'persona', H = 'PersonX have a traumatic incident', T = 'PersonX be introvert'
1. D_c = 'persona'
2. Yes
3. Yes
4. False. Because having a traumatic incident likely result from unfortune or carelessness, while being introvert is not relevant.

Now, please do your task for (each of) the following statement(s):
Inference: {}
D_i = '{}', H = '{}', T = '{}'"""


"content": "1. D_c = 'event'\n2. No\n3. Yes\n4. False. Because result of 2 is No.",v,1,a1
"content": "1. D_c = 'event'\n2. No\n3. Yes\n4. False. Because result of 2 is No.",v,1,a1
"content": "1. D_c = 'mental state'\n2. No\n3. Yes\n4. False. Because result of 2 is No.",v,0,a1
"content": "1. D_c = 'event'\n2. No\n3. Yes\n4. False. Because result of 2 is No.",v,1,a1
"content": "1. D_c = 'mental state'\n2. No\n3. Yes\n4. False. Because result of 2 is No.",v,0,a1
"content": "1. D_c = 'mental state'\n2. No\n3. Yes\n4. False. Because result of 2 is No.",v,0,a1
"content": "1. D_c = 'persona'\n2. Yes\n3. No (T is unclear about whether 'PersonX be elderly' refers to physical age or mental state)\n4. False. Because result of 3 is No.",v,1,a0
"content": "1. D_c = 'mental state'\n2. No\n3. Yes\n4. False. Because result of 2 is No.",x,0,a1
"content": "1. D_c = 'mental state'\n2. No\n3. No (T is unclear about what 'PersonX don't' refers to)\n4. False. Because result of 2 is No.",v,0,a1
"content": "1. D_c = 'mental state'\n2. No\n3. Yes\n4. False. Because result of 2 is No.",v,0,a1
"content": "1. D_c = 'mental state'\n2. No\n3. Yes\n4. False. Because result of 2 is No.",v,0,a1
"content": "1. D_c = 'event'\n2. No\n3. No (T is unclear about what 'PersonY be here' means)\n4. False. Because result of 2 is No.",v,1,a1
"content": "1. D_c = 'mental state'\n2. No\n3. No (T is unclear about the relationship between PersonX taking liberty and PersonY feeling kind)\n4. False. Because result of either 2 or 3 is No.",v,0,a1
"content": "1. D_c = 'mental state'\n2. No\n3. Yes\n4. False. Because result of 2 is No.",v,0,a1
"content": "1. D_c = 'mental state'\n2. No\n3. No (T is unclear about what 'PersonY be fun' means)\n4. False. Because result of either 2 or 3 is No.",v,0,a1
"content": "1. D_c = 'persona'\n2. No\n3. Yes\n4. False. Because being a babysitter is not a relevant attribute that can be inferred from the death of a baby brother.",v,0,a?
"content": "1. D_c = 'persona'\n2. Yes\n3. Yes\n4. True. Because walking someone through a process implies that PersonX has expertise and knowledge in the field, which is a characteristic of a professional.",v,1,a1
"content": "1. D_c = 'event'\n2. Yes\n3. No (T is unclear about what 'PersonX get')\n4. False. Because result of 3 is No.",v,0,a1
"content": "1. D_c = 'persona'\n2. No\n3. Yes\n4. False. Because result of 2 is No.",?,0,a1
"content": "1. D_c = 'event'\n2. Yes\n3. Yes\n4. False. Because the statement does not provide enough information to infer that PersonX has a tendency to fall or that falling is a characteristic of their",v,0,a1,
"content": "1. D_c = 'persona'\n2. No\n3. Yes\n4. False. Because 'be work' is not a relevant attribute to infer from starting a job.",x,0,a0
"content": "1. D_c = 'mental state'\n2. Yes\n3. Yes\n4. True. Because \"PersonX readily believe so\" implies that PersonX is easily convinced, and \"PersonX be convinced\" expresses PersonX's mental state of being",v,1,a1
"content": "1. D_c = 'event'\n2. Yes\n3. Yes\n4. False. Because there is no clear connection between having nothing else and seeing PersonY as a personal attribute.",v,0,a1
"content": "1. D_c = 'mental state'\n2. Yes\n3. Yes\n4. False. Because being startled is a reaction to a sudden event, while not hearing a car drive up could be due to various reasons and does not necessarily",v,1,a1
"content": "1. D_c = 'mental state'\n2. Yes\n3. No (T is unclear about what 'PersonX be promise' means)\n4. False. Because result of 3 is No.",v,1,a1
"content": "1. D_c = 'mental state'\n2. Yes\n3. Yes\n4. True. Because 'relate' implies that PersonX is able to connect with others on an emotional level, which suggests that PersonX is likely to have feelings.",v,1,a0
"content": "1. D_c = 'mental state'\n2. Yes\n3. Yes\n4. True. Because mistakenly shaving someone's head is likely to cause upset or distress.",v,1,a1


look at instance of relation xReact, oReact, xAttr (those which need the 'rule' (this step) the most)
Error in step 1:
  24/27 is correct

Error in step 2:
  omg, it fails more than half of the case
  11/27 is correct -> that's what gpt3.5-turbo is bad at!

Error in step 3:
  :)) hard to judge, but at least chatgpt point out some non-sense/ambiguous assertion


# NL Conversion

April 28: with the rule in preprocess/convert_triple_to_nl_assertion, get
$ python preprocess/convert_triple_to_nl_assertion.py
p, r, f1 (0.5277777777777778, 0.7307692307692307, 0.6129032258064515, None)
auc 0.7535425101214576
acc 0.7647058823529411

-> more problem with precision! while conceptualization seems to lower precision

re-run, no change


## error analysis

row (=index+1)
1 (reverse triple is correct)
2 x
3 x
15 x
23 x
27 x
30 (lack info)
35 (mistake emotion)
36 (story instead of cms)
38 (head-tail related, but not cms)
43 (ambiguous)
52 (maybe, consider as debatable)
57 (wrong)
61 x
62 (not intention)
66 (seem the ground-truth label is correct)
68 ?? (prompt affected)
72 x
73 (ambiguous, cannot give 1)
74 x (should be isBefore)
77 x 
80 (trivial knowledge, and unclear)
81 x
83 x

x: arguable


## conversion rule draft

{
  'xWant': 'head_event, thus, modified tail_event',
  'oWant': 'head_event, thus, modified tail_event',
  'xEffect': 'head_event, thus as an effect, tail_event',
  'oEffect': 'head_event, thus as an effect, tail_event',
  'xReact': 'head_event, thus as a result on PersonX's emotion, modified tail_event',
  'oReact': 'head_event, thus as a result on PersonY's emotion, modified tail_event',
}

Then double check with training data! and chatgpt!
try to replace (PersonX, PersonY) with (A person, other people) and train
to some extent, xNeed cover xIntent + xAttr; xEffect cover xWant + xReact (state (physical state), action, emotion/feeling (mental state). do have plausible xEffect triple with tail 'PersonX be <feeling>'), similar for o?


- oWant
  oWant,PersonY want [^t]
  
  modify tail_event
  239 total
  227 start with PersonY -> good, may not care about other

  if not start with PersonY -> false
    or honestly add 'PersonY want' at the beginning

  elif start with PersonY:
    if continue with want:
      do nothing :)
    elif not contain word ' not ' and 
        continue with a verb (need to use wordnet to check, if contain a meaning as verb but not a modal verb (!= will, would, can, could, shall, should, must, might, may), then apply):
      change to PersonY want to i.e add ' want to' after PersonY
    else: # can set this option by default, even the case ' have to ' in tail
      add 'PersonY want' before tail_event

- xWant
  608/611 instances start with PersonX
  similar rule to oWant

PersonX shit,xWant,PersonX have to put down some toilet paper in toilet,adv,dev,0
PersonX have sensitive skin,xWant,PersonX have to be careful,adv,tst,0
we want it to be xEffect

- x/oEffect
PersonX be a horrible person,oEffect,PersonY rush out,cs_head,tst,1
-> PersonX be a horrible person, thus as an effect, PersonY rush out
May also try 'PersonX be a horrible person, thus as an effect on other people, PersonY rush out'
  but prefer shorter prompt

- xReact
  529/533 starts with PersonX
  ', thus PersonX feel ...' x
  thus as in term of PersonX's feeling
  thus as a result on PersonX's emotion or feeling, ... (pick)
  thus as a result, PersonX feel 

  ! there are adj describing charateristics, while there are adj describe emotion
  and a lot of conflict with annotation rule! if we turn 'PersonX want to do smt' to 'PersonX feel PersonX want to do smt', the semantic of the triple, i.e ground-truth label, can be changed!

  if start with 'PersonX feel' -> skip
  elif start with 'PersonX be' -> add 'PersonX feel' at the beginning
    ?? why not replace 'be' by 'feel' -> change the meaning
    ?? why do we need to add 'PersonX feel' -> a lot of original tails sound not a feeling
  else: skip, don't add anything

- oReact
  167/183 start with PersonY
  similar

- xAttr
  100% PersonX
  thus it can be seen that -> not clear about the attribute (discriminate with xIntent) and before (xReact)
  thus it can be seen about PersonX's attribute that ...

  need to check the case of attributive and emotional adj!

- xIntent
  thus it can be seen about PersonX's intention that ...


- xNeed:
    374/378 start with PersonX

    thus before that, PersonX needed
    if start with 'PersonX need'
        add 'ed'
    elif start with 'PersonX be'
        modify to 'PersonX needed to be'
    else
        ?? can add 'needed to' between PersonX?
        or add 'PersonX needed' before -> pick this one
        --> cannot modify like this, because it changes semantics

    thus it can be seen about PersonX's motivation that --> oh no, it's not motivation!
    thus before that? --> need to modify the tail event. Don't bring enough sense of causal.
      not suitable for this example "PersonX learn new thing,xNeed,PersonX attend class,test_set,tst,1"

    it's unlikely {} unless {} -> pick, no need to modify. Do need! Remove need to, have to
    head_event implies that tail_event? -> can not discriminate with xWant/x..

    Interesting. equivalent logical form, but in natural language, it has the sense of temporal order

    In order to (remove Person X + modal verb) head_event, tail_event
      forget about it, as we need to add 'need to' to .. or modify further
      headache


- Causes: ---> hmm, is Causes similar to x/oEffect, even in a bradoer sense
    easy!
    'Because head_event, tail_event'
    or a bit more complicated prompt 'Because head_event, it's likely that tail_event'
        a word to trigger the judgement of plausibility by frequency

    need to test with ChatGPT, otherwise can't tell which prompt is better
    also compare to prompts of other relations, level of interaction in prompt?
        if compare, suggest to use the simple prompt

- xReason
    'head_event, because tail_event'

- isBefore
    'After head_event, it's likely that tail_event'

- isAfter
    'Before head_event, it's likely that tail_event'


- HinderedBy (cover inverse of xNeed)
    'The event head_event is likely hindered by the event tail_event'
    How about 'if tail_event, it's not likely that head_event'

- HasSubEvent
    'The event head_event includes the event/action that tail_event'


-----
Given the following ground-truth statements:


PersonX take PersonX cat to vet,PersonX be kind,xAttr,atomic,1
PersonX take the dog to vet,PersonX be responsible,xAttr,atomic,1
PersonX take PersonX cat to vet,PersonX be responsible,xAttr,atomic,1

PersonX take PersonX dog to vet,xNeed,PersonX be a good owner,adv,tst,1

--
PersonX bring solution for problem,xNeed,PersonX be a problem solver


-----
1.
Judge the following statement if it likely to occur, only answer 'True' or 'False':
The action that PersonX take PersonX dog to vet implies that PersonX needed to be a good owner
The action that PersonX do not support PersonY implies that PersonX be sour

2.
try "Judge the following statement if it's commonsense, only answer 'True' or 'False':"
  -> OMG, more incorrect answers, cover what's already incorrect in template1

3.
add instruction, and example

Judge the following statement/inference if it likely to occur, only answer 'True' or 'False'. Please not only consider the relevance of two clauses in the statement, but also the aspect that it refers to, e.g emotion, intention, attribute, etc.


Insight:
1. UNBELIEVABLE! longer prompt, perform worse
  adding few-shot examples/demonstrator makes the model perform worse
  TODO: adding annotation instruction?? redesign the prompt, need to include the relation token

2. using asyncio + paid account v.s for loop + free trial account -> equal performance
  however, doing evaluation for multiple samples within one run will decrease model performance


# Conceptualization


TODO: more conceptualization, force the attention to relation

-----
conceptualization_prompt = """Please read the convention, task description, examples, then do the task:
Convention: There are three dimensions ['persona', 'event', 'mental state']. All ordered dimension pairs are valid, except ('event', 'persona') and ('event', 'mental state') are invalid.

Task description: Given an inference and its corresponding dimension D_i and its two main clauses H and T, you are required to do the following step by step:
1. Focus on the last clause T, determine which dimension best describe what the clause expresses. Denote the dimension as D_c.
2. Judge if the pair (D_c, D_i) is valid. Answer 'Yes' or 'No'.
3. Judge if the main clauses' meaning is all clear/non-ambiguous. Answer 'Yes' or 'No'.
4. Based on result in 2, 3, and the semantics of the inference, judge if the inference is likely 'True' or 'False'. Note that if result of either 2 or 3 is 'No', then result of 4 should be 'No'.

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
D_i = '{}', H = '{}', T = '{}'"""


-----
conceptualization_prompt = """Please do the following problem:
Definition: There are six dimensions ['attribute', 'prerequisite', 'intent', 'emotion', 'want', 'effect', 'causal', 'temporal']
    and three aspects ['state', 'action', 'attribute'] we may deal with.
    Note that the (aspect, dimension) pairs ('action', 'attribute'), ('action', 'emotion'), ('attribute', 'intent') are not valid, while any other pair is valid.

Task description: Given an inference and its corresponding dimension, you are required to do the following step by step:
1. Focus on the last clause of the inference, determine which aspect the clause express.
2. Judge if the (aspect, dimension) is valid. Answer 'Yes' or 'No'.
3. Judge if the inference's meaning is unclear/ambiguous. Answer 'Yes' or 'No'.
4. Based on result in 2, 3, and the semantics of the inference, judge if the inference is likely 'True' or 'False'. Note that if result of either 2 or 3 is 'No', then result of 4 should be 'No'.

Examples:

* Example 1
Inference: PersonX learns to play the trumpet, thus, PersonY wants to applaud PersonX's talent
Dimension: want
1. action
2. Yes
3. Yes
4. True. Because 1) "PersonX learns to play the trumpet" means PersonX are talent and effortful in learning musical instrument, and 2) people tend to applaud other people's effort and talent.

* Example ? TODO: find yes yes False
Inference: PersonX learns to play the trumpet, thus, PersonY wants to applaud PersonX's talent
Dimension: 
1. action
2. Yes
3. Yes
4. True. Because people tend to applaud other people's effort and talent.

* Example 3
Inference: PersonX do not feel well, thus as a result on PersonX's emotion, PersonX only make it through half.
Dimension: emotion
1. action
2. No
3. Yes
4. False. Because, result of 2 is No.

* Example ? TODO: find yes no False


Now, please do your task for each of the following statement(s):
Statement: {}"""


-----
Your task is to do the following step by step, given the original statement.
1. find a phrase in the statement (phrases in the last clause of the statement are preferable)
2. conceptualize it in a concise manner
3. write down the conceptualized statement formed by direct substitution
4. based on both original and conceptualized statements, judge if the original statement likely to occur by answering 'True' or 'False'. After that, you may briefly explain within one sentence.

For example,

## Example 1
Statement: PersonX learns to play the trumpet, thus, PersonY wants to applaud PersonX's talent
1. the trumpet
2. instrument
3. PersonX learns to play instrument, thus PersonY wants to applaud PersonX's talent
4. True. Because people tend to applaud other people's effort and talent.

## Example 2
Statement: PersonX wins the costume contest, thus, PersonX feel excited
1. the costume contest
2. competition
3. PersonX wins competition, thus PersonX feel excited
4. True. Because it's very happy to win a competition.

## Example 3
Statement: PersonX do not feel well, thus as a result on PersonX's emotion, PersonX only make it through half.
1. only make it through half
2. don't finish
3. PersonX do not feel well, thus as a result on PersonX's emotion, PersonX don't finish
4. False. Because "don't finish" is an action, rather than an emotion.

Now, please do your task for each of the following statement(s):
PersonX prevent PersonY, thus as a result on PersonX's emotion, PersonX never reach out to anyone
PersonX go to sleep on hollow, thus as a result on PersonX's emotion, PersonX feel PersonX be tired
PersonX read non fiction, thus as a result on PersonX's emotion, PersonX feel PersonX be curious


Continue that task with the following statement:


PersonX tries to bite PersonY, thus PersonX was sent to detention
PersonX sells things on ebay, thus it means PersonX intended to establish an online store

PersonX gets a good night's rest, thus Person feel refreshed

PersonX take PersonX dog to vet, thus before that, PersonX needed to be a good owner

PersonX be move from apartment, thus before that, PersonX needed mover

PersonX like to feel pretty, thus it means before that, PersonX needed wear makeup

PersonX answer teacher 's question correctly, thus it means before that, PersonX needed to do well academically




PersonX take PersonX cat to vet, thus before that, PersonX needed to be responsible

PersonX get the great white, thus as an effect, PersonX need to hire a shark


~~ thus it means before that ... The prompt need to have interaction

-----

Now design a prompt for target phrase (easier)
Hmm, 

Wait, why do we only do conceptualization?
    we can definitely do specification for already-abstract events, e.g good owner



PersonX gets [a good night's rest],refreshed,xReact,1,6949,2,sleeping,1
PersonX tries [to bite PersonY],sent to detention,xEffect,1,22376,1,aggressive behavior,1

PersonX be move from apartment,xNeed,PersonX need mover,adv,tst,1
PersonX own the truck,xNeed,PersonX be responsible,adv,tst,0
PersonX kiss PersonY,xNeed,PersonX would need to ask PersonY permission


-----
Find triples from train and eval sets which are conceptually similar


eval set: PersonX want a martini,xWant,PersonX want to enjoy a drink,adv,tst,1
train set: PersonX get a cup of coffee,PersonX want to enjoy a drink,xWant,atomic,1



-----
GPT3.5-turbo

Statement 1:
1. prevent PersonY
2. stop someone
3. PersonX stops someone, thus as a result on PersonX's emotion, PersonX never reach out to anyone
4. True. Because if PersonX is preventing PersonY, it may be due to trust issues, which can lead to PersonX not reaching out to anyone.

Statement 2:
1. go to sleep on hollow
2. sleep on an empty stomach
3. PersonX sleeps on an empty stomach, thus as a result on PersonX's emotion, PersonX feels tired
4. True. Because sleeping on an empty stomach can lead to fatigue.

Statement 3:
1. read non fiction
2. read factual books
3. PersonX reads factual books, thus as a result on PersonX's emotion, PersonX feels curious
4. True. Because reading non-fiction can lead to curiosity and a desire to learn more.


GPT4's answer

## Statement 1
1. never reach out to anyone
2. isolate themselves
3. PersonX prevent PersonY, thus as a result of PersonX's emotion, PersonX isolate themselves
4. True. Because preventing someone could lead to feeling guilty or ashamed, making PersonX isolate themselves.

## Statement 2
1. go to sleep on hollow
2. sleep on uncomfortable surface
3. PersonX sleep on an uncomfortable surface, thus as a result of PersonX's emotion, PersonX feels tired
4. False. Sleeping on an uncomfortable surface would lead to physical discomfort, not emotional tiredness.

## Statement 3
1. read non-fiction
2. read informative texts
3. PersonX read informative texts, thus as a result of PersonX's emotion, PersonX feels curious
4. True. Reading informative texts can spark curiosity and a desire to learn more.


Dragonfly's answer

## Statement 1
1. prevent PersonY
2. stop
3. PersonX stop PersonY, thus as a result on PersonX's emotion, PersonX never reach out to anyone
4. False. Because "never reach out to anyone" is an action, rather than an emotion.

## Statement 2
1. go to sleep on hollow
2. rest on an empty bed
3. PersonX rest on an empty bed, thus as a result on PersonX's emotion, PersonX feel PersonX be tired
4. True. Because feeling tired is a common emotion when sleeping on an empty bed.

## Statement 3
1. read non fiction
2. read factual material
3. PersonX read factual material, thus as a result on PersonX's emotion, PersonX feel PersonX be curious
4. True. Because reading factual material often leads to curiosity.