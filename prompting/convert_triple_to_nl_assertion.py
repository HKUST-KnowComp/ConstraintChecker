"""
terminology:
  triple/instance: refer to (h,r,t) format
  assertion/statement: refer to natural language format
"""
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from nltk.corpus import wordnet as wn
import pandas as pd
import random
from tqdm import tqdm


seed = 2023 # try different seed, 2025 gives the best :)) but not vary much
random.seed(seed)

naive_prompt_tq = "Answer whether the following statement is plausible. Answer with only Yes or No: "
templates_tq = {
    "xWant" : "as a result, PersonX wants to",
    "oWant" : "as a result, PersonY or others want to",
    "xEffect" : "as a result, PersonX will",
    "oEffect" : "as a result, PersonY or others will",
    "xReact" : "as a result, PersonX feels",
    "oReact" : "as a result, PersonY or others feel",
    "xAttr" : "PersonX is seen as",
    "xIntent" : "because PersonX wanted",
    "xNeed" : "but before, PersonX needed",
    "Causes" : "causes",
    "xReason" : "because",
    "isBefore" : "happens before",
    "isAfter" : "happens after",
    "HinderedBy" : "can be hindered by",
    "HasSubEvent" : "includes the event or action",
}

naive_prompt0 = "Answer whether the following statement is plausible. Answer with only Yes or No:\n"
naive_prompt1 = "Judge the following statement if it's likely to occur, only answer 'True' or 'False':\n"
modal_verbs = ['will', 'would', 'can', 'could', 'shall', 'should', 'must', 'might', 'may']
templates = {
  'xWant': '{}, thus, {}',
  'oWant': '{}, thus, {}',
  'xEffect': '{}, thus as an result, {}',
  'oEffect': '{}, thus as an result, {}',
  'xReact': '{}, thus as a result on PersonX\'s emotion, {}',
  'oReact': '{}, thus as a result on PersonY\'s emotion, {}',
  'xAttr': '{}, thus it can be seen about PersonX\'s attribute that {}',
  'xIntent': '{}, thus it can be seen about PersonX\'s intention that {}',
  'xNeed': 'The event {} will not happen unless {}', # or '{ will not} unless'
  'Causes': 'Because {}, {}',
  'xReason': '{}, because {}',
  'isBefore': 'After {}, {}',
  'isAfter': 'Before {}, {}',
  'HinderedBy': 'The event {} will not happen, if {}',  # or '{ will not} if'
  'HasSubEvent': 'The event {} includes the event/action that {}',
}
# usage: templates[rel].format(head, tail)
# template design fits data attribute (e.g 'xWant' instances almost all start with PersonX)
def rel2dim(rel):
  if rel == 'xAttr':
    return 'persona'
  elif rel in ['xIntent', 'xReact', 'oReact']:
    return 'mental state'
  else:
    return 'event'


# modification rule + add to template
def convert_triple_to_nl_assertion(data_df):
  assertions_and_tails = []

  for i in tqdm(range(len(data_df)), total=len(data_df)):
    head, rel, tail = data_df['head'][i], data_df['relation'][i], data_df['tail'][i]

    # if rel in ['xEffect', 'oEffect', 'xAttr', 'xIntent', \
    #   'Causes', 'xReason', 'isBefore', 'isAfter', 'HinderedBy', 'HasSubEvent']:
    #   pass
    
    if rel == 'xWant':
      if not tail.startswith('PersonX'):
        tail = 'PersonX want ' + tail

      elif not tail.startswith('PersonX want'):
        p, second_word, _ = (tail+' ').split(' ', 2)
        if second_word not in modal_verbs and \
           ' not ' not in tail and \
           'v' in set([w.pos() for w in wn.synsets(second_word)]):
          tail = 'PersonX want to ' + second_word + ' ' + _
        else:
          tail = 'PersonX want ' + tail

    elif rel == 'oWant':
      if not tail.startswith('PersonY'):
        tail = 'PersonY want ' + tail

      elif not tail.startswith('PersonY want'):
        p, second_word, _ = (tail+' ').split(' ', 2)
        if second_word not in modal_verbs and \
           ' not ' not in tail and \
           'v' in set([w.pos() for w in wn.synsets(second_word)]):
          tail = 'PersonY want to ' + second_word + ' ' + _
        else:
          tail = 'PersonY want ' + tail

    elif rel == 'xReact':
      if tail.startswith('PersonX be'):
        tail = 'PersonX feel ' + tail

    elif rel == 'oReact':
      if tail.startswith('PersonY be'):
        tail = 'PersonY feel ' + tail

    elif rel == 'xNeed':
      tail = tail.replace('have to ', '').replace('need to ', '')

    assertions_and_tails.append((templates[rel].format(head, tail).strip(), tail))

  return assertions_and_tails


def parse():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--prediction_file', type=str, required=False,
    default='results/prompting_dev/gpt3.5-turbo_template1.txt')
  parser.add_argument('-s', '--split', type=str, required=False,
    default='dev', choices=['tst', 'dev', 'trn'])
  parser.add_argument('-m', '--mode', type=str, required=False,
    default='evaluation', choices=['conversion', 'evaluation'])
  args = parser.parse_args()
  return args


def main():
  args = parse()

  if args.mode == 'conversion':
    data = pd.read_csv('prompting_data/ckbp2.0.csv')
    # data['pred'] = pd.read_csv('results/pseudoreasoner_and_human/human1.csv')['label1']
    # data['pred'] = pd.read_csv('prompting_data/ckbp2.0_with_gpt3.csv')['chatgpt']

    if args.split == 'dev':
      filename = 'dev_data_to_tune'
      data = data[data['split'] == 'dev']
      n_total = len(data)
      n_taken = 100 # since we sample by fraction, no. instances at the end may not be n_taken
      selected = pd.DataFrame()

      for rel in templates.keys():
        temp = data[data['relation'] == rel].sample(frac=n_taken/n_total, random_state=seed)
        temp['dim'] = rel2dim(rel)
        selected = selected.append(temp)

    elif args.split == 'tst':
      filename = 'tst_data_to_eval'
      df = data[data['split'] == 'tst']
      # df['pred'] = open('results/prompting_tst/full_tst_gpt3.5-turbo_template1.txt', 'r').read().splitlines()
      df['downsample'] = 0

      rel_idx, clss_idx, lb_idx = {}, {}, {}
      for rel in templates.keys():
        rel_idx[rel] = df['relation'] == rel
      for clss in ['test_set', 'cs_head', 'all_head', 'adv']:
        clss_idx[clss] = df['class'] == clss
      for l in [0,1]:
        lb_idx[l] = df['label'] == l

      # down-sample while keeping ratio
      for clss in ['test_set', 'cs_head', 'all_head', 'adv']:
        for rel in templates.keys():
          for l in [0,1]:
            idx = rel_idx[rel]*clss_idx[clss]*lb_idx[l]
            num = sum(idx)
            num_selected = int(num*0.25)
            selection = [1]*num_selected + [0]*(num-num_selected)
            random.shuffle(selection)
            df['downsample'][idx] = selection

      selected = df[df['downsample'] == 1]
      del selected['downsample']
      # selected['pred'].astype(int).to_csv(
      #   'results/prompting_tst/tst_human1.txt', index=False, header=False)
      # del selected['pred']
      selected['dim'] = selected['relation'].apply(rel2dim)

    else:
      filename = 'trn_w_assertion'
      selected = pd.read_csv('prompting_data/ckbp_v2_train.csv').fillna('')

    selected.reset_index(inplace=True, drop=True)
    assertions_and_tails = convert_triple_to_nl_assertion(selected)
    selected['assertion'] = [x[0] for x in assertions_and_tails]
    # selected['tail'] = [x[1] for x in assertions_and_tails]
    selected['assertion_tq'] = 'If '+ selected['head']+', '+\
      selected['relation'].apply(lambda r:templates_tq[r])+', '+selected['tail']

    # selected['naive1'] = naive_prompt_tq + selected['assertion_tq']
    # selected['naive2'] = naive_prompt0 + selected['assertion']
    # selected['naive3'] = naive_prompt1 + selected['assertion']
    selected.to_csv(f'prompting_data/{filename}.csv', index=False)

  else:
    filename = 'dev_data_to_tune' if args.split == 'dev' else 'tst_data_to_eval'
    data = pd.read_csv(f'prompting_data/{filename}.csv')
    labels = data['label']
    predictions = open(args.prediction_file).read().splitlines()
    predictions = [1 if ('true' in x.lower()[:12] or 'yes' in x.lower()[:8]) else 0 for x in predictions]
    # 'p, r, f1, support', 'auc', 'acc'
    p, r, f1, support = list(precision_recall_fscore_support(labels, predictions, average='binary'))
    auc = roc_auc_score(labels, predictions)
    acc = accuracy_score(labels, predictions)
    print(f'{p:.4f}|{r:.4f}|{f1:.4f}|{auc:.4f}|{acc:.4f}|')
    for i in range(len(labels)):
      if labels[i] != predictions[i]:
        print(i, labels[i], end=', ')

if __name__ == '__main__':
  main()


# python prompting/convert_triple_to_nl_assertion.py -s dev -f results/prompting_dev/gpt3.5-turbo_template1.txt
# python prompting/convert_triple_to_nl_assertion.py -s tst -f results/prompting_tst/tst_gpt3.5-turbo_template1.txt