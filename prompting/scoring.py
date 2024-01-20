import pandas as pd
from sklearn.metrics import (
  precision_recall_fscore_support, confusion_matrix,
  roc_auc_score, accuracy_score, f1_score
)


def parse():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', '--split', type=str, required=False,
    default='tst', choices=['dev', 'tst'])
  parser.add_argument('-q', '--question', type=str, required=False,
    default='0', choices=['moe', 'l2m', '0', '1', '2', '3'])
  parser.add_argument('-p', '--prediction_file', type=str, required=False,
    default='gpt3.5-turbo_moe_q1_prompt1.txt')
  parser.add_argument('-g', '--ground_truth', type=str, required=False,
    default='dev_moe_q1_label.txt')
  parser.add_argument('--q0', type=str, required=False,
    default='gpt3.5-turbo_naive_prompt3.txt')
  parser.add_argument('--q1', type=str, required=False,
    default='gpt3.5-turbo_moe_q1_prompt1.txt')
  parser.add_argument('--q2', type=str, required=False,
    default='gpt3.5-turbo_moe_q2_prompt1.txt')
  parser.add_argument('--q3', type=str, required=False,
    default='gpt3.5-turbo_moe_q3_prompt3.txt')
  parser.add_argument('--ablation', type=int, required=False, default=0)
  args = parser.parse_args()

  if args.split == 'dev':
    args.base_path = 'results/prompting_dev/'
    args.data_file = 'prompting_data/dev_data_to_tune.csv'
  elif args.split == 'tst':
    args.base_path = 'results/prompting_tst/'
    args.data_file = 'prompting_data/tst_data_to_eval.csv'

  args.prediction_file = args.base_path + args.prediction_file
  args.ground_truth = args.base_path + args.ground_truth
  args.q0 = args.base_path + args.q0
  args.q1 = args.base_path + args.q1
  args.q2 = args.base_path + args.q2
  args.q3 = args.base_path + args.q3
  return args

def dim2num(pred):
  pred = pred.lower()
  if 'event' in pred or '1' in pred:
    pred = 1
  elif 'mental state' in pred or '3' in pred:
    pred = 3
  else:
    pred = 2
  return pred

def check_temporal_order(pred, relation):
  check_tail_before_head_to_mark_wrong = 'xWant,oWant,xEffect,oEffect,xReact,oReact,Causes'.split(',')
  check_tail_after_head_to_mark_wrong = 'xIntent,xNeed,HinderedBy'.split(',')

  idx = relation.isin(check_tail_before_head_to_mark_wrong)
  temp = pred[idx]
  temp[temp != 0] = 1
  pred[idx] = 1 - temp

  idx = relation.isin(check_tail_after_head_to_mark_wrong)
  temp = pred[idx]
  temp[temp != 1] = 0
  pred[idx] = temp

  pred[~relation.isin(check_tail_before_head_to_mark_wrong+check_tail_after_head_to_mark_wrong)] = 0
  return pred

def post_process_zeroshot_cot(args):
  if '_full.txt' not in args.prediction_file:
    return

  predictions = open(args.prediction_file).read().split('\n--')
  if 'davinci' not in args.prediction_file:
    for i in range(len(predictions)):
      p = predictions[i].strip().lower()
      if '\n' not in p:
        p = 1 if 'true' in p else 0
      else:
        p = p.rsplit('\n', 1)[1]
        p = min(1, ('true' in p) + ('plausible' in p)*('not plausible' not in p))
      predictions[i] = p
  else:
    for i in range(len(predictions)):
      p = predictions[i].rsplit('--')[-1]
      p = min(1, ('True' in p) + ('Yes' in p))
      predictions[i] = p

  with open(args.prediction_file[:-9] + '.txt', 'w') as fout:
    fout.write('\n'.join([str(x) for x in predictions]))


def main(args):
  data_df = pd.read_csv(args.data_file)
  idx_q1 = data_df['relation'].isin('xReact,oReact,xAttr'.split(','))
  idx_q3 = data_df['relation'].isin('xIntent,xNeed'.split(','))
  # idx_q3 = data_df['relation'].isin('xIntent,xNeed,HinderedBy,Causes'.split(',')) # + oReact, consistently better for these 3 relations.
  n = len(data_df)
  
  if args.question not in ['moe', 'l2m']:
    predictions = open(args.prediction_file).read().splitlines()

    if args.question == '0':
      if predictions[0].isnumeric():
        predictions = [int(x) for x in predictions]
      else:
        predictions = [1 if ('true' in x.lower()[:12] or 'yes' in x.lower()[:8]) else 0 for x in predictions]
      predictions = pd.Series(predictions)
      labels = data_df['label']
      # print(labels.index[data_df['relation'].isin(['HinderedBy'])])
      # for i in labels.index[data_df['relation'].isin(['HinderedBy'])]:
      #   if labels[i] == 0 and predictions[i] == 1:
      #     print(i+2)
      # print(predictions[data_df['relation'] == 'xReact'].tolist())

    elif args.question == '1':
      labels = pd.Series([int(s[0]) for s in open(args.ground_truth).read().splitlines()])
      predictions = pd.Series([dim2num(s) for s in predictions])
      predictions[predictions > 1] = 0
      labels[labels > 1] = 0
      for i in labels.index[idx_q1]:
        if labels[i] == 1 and predictions[i] == 0:
          print(i)

    elif args.question == '2':
      labels = pd.Series([int(s[0]) for s in open(args.ground_truth).read().splitlines()])
      predictions = pd.Series([1 if 'yes' in s.lower() else 0 for s in predictions])
      labels = [1-s for s in labels]

  elif args.question == 'l2m':
    labels = data_df['label']
    l2m_answer = pd.read_csv(args.q0)['prediction']

    predictions = open(args.prediction_file).read().splitlines()
    if predictions[0].isnumeric():
      predictions = [int(x) for x in predictions]
    else:
      predictions = [1 if ('true' in x.lower()[:12] or 'yes' in x.lower()[:8]) else 0 for x in predictions]
    predictions = pd.Series(predictions)
    idx = data_df['relation'].isin('xReact,oReact,xAttr,xIntent,xNeed'.split(','))
    predictions[idx] = l2m_answer[idx]

  else:
    labels = data_df['label']
    predictions_q0_plau = open(args.q0).read().splitlines()
    if predictions_q0_plau[0].isnumeric():
      predictions_q0_plau = [int(x) for x in predictions_q0_plau]
    else:
      predictions_q0_plau = [1 if ('true' in x.lower()[:12] or 'yes' in x.lower()[:8]) else 0 for x in predictions_q0_plau]

    if args.ablation == 0:
      predictions_q1_mismatch = pd.Series([dim2num(s) for s in open(args.q1).read().splitlines()])
      predictions_q1_mismatch[~idx_q1] = 0
      temp = predictions_q1_mismatch[idx_q1]
      temp[temp > 1] = 0
      predictions_q1_mismatch[idx_q1] = temp

      # predictions_q2_not_amb = pd.Series([1 if 'yes' in s.lower() else 0 for s in open(args.q2).read().splitlines()])
      # predictions_q2_not_amb = 1 - pd.Series([int(s) for s in open(args.q2).read().splitlines()])

      predictions_q3_choices = pd.Series([int(s) if s.isnumeric() else -1 for s in open(args.q3).read().splitlines()])
      predictions_q3_wrong_temporal_order = check_temporal_order(predictions_q3_choices, data_df['relation'])
      predictions_q3_wrong_temporal_order[~idx_q3] = 0

    else:
      predictions_q1_mismatch = pd.Series([0]*n)
      predictions_q1_mismatch[idx_q1] = [1 if 'yes' in x.lower() else 0 for x in open(args.q1).read().splitlines()]
      # print(sum(predictions_q1_mismatch))
      predictions_q3_wrong_temporal_order = pd.Series([0]*n)
      predictions_q3_wrong_temporal_order[idx_q3] = [
        1 if 'yes' in x.lower() else 0 for x in open(args.q3).read().splitlines()]
      # print(sum(predictions_q3_wrong_temporal_order))

    predictions_q0 = pd.Series(predictions_q0_plau)
    predictions_q1 = 1 - predictions_q1_mismatch
    # predictions_q2 = predictions_q2_not_amb
    predictions_q3 = 1 - predictions_q3_wrong_temporal_order

    # for r in 'xWant,oWant,xEffect,oEffect,xReact,oReact,Causes,xIntent,xNeed,HinderedBy'.split(','):
    #   temp = predictions_q3_wrong_temporal_order.copy()
    #   temp[(data_df['relation'] != r)] = 0
    #   print(r, sum(temp))
    #   predictions_q3 = 1 - temp
    predictions = predictions_q0*predictions_q1*predictions_q3
    # predictions = predictions_q0*predictions_q2

  rel_breakdown_f1 = []
  for r in ('xWant|oWant|xEffect|oEffect|xReact|oReact|xAttr|xIntent|xNeed|' + \
    'Causes|isBefore|isAfter|HinderedBy|HasSubEvent').split('|'):
    idx = (data_df['relation'] == r)
    score = f1_score(labels[idx], predictions[idx], average='binary')
    # print(r, score)
    rel_breakdown_f1.append(score)

  p, r, f1, support = list(precision_recall_fscore_support(labels, predictions, average='binary'))
  auc = roc_auc_score(labels, predictions)
  acc = accuracy_score(labels, predictions)
  score_report_template = '{:.2f}|'*19
  score_list = [auc, acc, p, r, f1] + rel_breakdown_f1
  print(score_report_template.format(*[x*100 for x in score_list]))
  # print(confusion_matrix(labels, predictions))


def analyze_l2m_result():
  import re
  typing_l2m = open('results/prompting_tst/tst_gpt3.5-turbo_least2most_prompt3_full.txt'
    ).read().split('--294')[0].split('--')[1:]

  count_misjudge = 0
  for instance in typing_l2m:
    print(instance[:3])
    answers = [x for x in instance.splitlines() if 'A: ' in x]
    if re.findall(r'2.|persona|3.|mental state', answers[0]):
      if 'No' in answers[1]:
        count_misjudge += 1
    elif re.findall(r'1.|event|activity', answers[0]):
      if 'Yes' in answers[1]:
        count_misjudge += 1

  print(count_misjudge, '/', len(typing_l2m)) 


if __name__ == '__main__':
  args = parse()
  # main(args)
  analyze_l2m_result()

'''
python prompting/scoring.py -s dev -q 0 \
  -p gpt3.5-turbo_template1.txt


python prompting/scoring.py -s tst -q 0 \
  -p tst_gpt3.5-turbo_naive_prompt3.txt

python prompting/scoring.py -s tst -q l2m \
  -p tst_gpt3.5-turbo_naive_prompt3.txt \
  --q0 tst_gpt3.5-turbo_least2most_prompt3.csv

python prompting/scoring.py -s tst -q moe \
  --q1 tst_gpt3.5-turbo_moe_q1_prompt1.txt \
  --q3 tst_gpt3.5-turbo_moe_q3_prompt3.txt \
  --q0 tst_gpt3.5-turbo_naive_prompt3.txt


python prompting/scoring.py -s tst -q 0 \
  -p tst_davinci_kate_top1_prompt1.txt

python prompting/scoring.py -s tst -q l2m \
  -p tst_davinci_naive_prompt3.txt \
  --q0 tst_davinci_least2most_prompt3.csv

python prompting/scoring.py -s tst -q moe \
  --q1 tst_davinci_moe_q1_prompt1.txt \
  --q3 tst_davinci_moe_q3_prompt3.txt \
  --q0 tst_davinci_naive_prompt3.txt
'''