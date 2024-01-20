import os
import pandas as pd
from sklearn.metrics import (
  precision_recall_fscore_support, confusion_matrix,
  roc_auc_score, accuracy_score, f1_score
)


def parse():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--q0', type=str, required=False,
    default='tst_llama2-7b_naive_prompt3.txt')
  parser.add_argument('--q1', type=str, required=False,
    default='tst_llama2-7b_moe_q1_prompt1.txt')
  parser.add_argument('--q3', type=str, required=False,
    default='tst_llama2-7b_moe_q3_prompt3.txt')
  args = parser.parse_args()

  args.base_path = 'results/llama2/'
  args.data_file = 'prompting_data/tst_data_to_eval.csv'
  args.q0 = args.base_path + args.q0
  args.q1 = args.base_path + args.q1
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

def main(args):
  data_df = pd.read_csv(args.data_file)
  idx_q1 = data_df['relation'].isin('xReact,oReact,xAttr'.split(','))
  idx_q3 = data_df['relation'].isin('xIntent,xNeed'.split(','))
  n = len(data_df)

  predictions_q0_plau = open(args.q0).read().splitlines()
  if predictions_q0_plau[0].isnumeric():
    predictions_q0_plau = [int(x) for x in predictions_q0_plau]
  elif 'randomcot' in args.q0:
    yes_ = "likely to occur"
    no_ = "not likely to occur"
    predictions_q0_plau = [1 if (yes_ in x.lower() and no_ not in x.lower()) else 0 for x in predictions_q0_plau]
  elif 'zeroshot' in args.q0: # need to refine this rule!
    predictions_q0_plau = [1 if ('true' in x.lower().rsplit('--',1)[-1] or 'yes' in x.lower().rsplit('--',1)[-1]) else 0 for x in predictions_q0_plau]
  else:
    predictions_q0_plau = [1 if ('true' in x.lower()[:12] or 'yes' in x.lower()[:8]) else 0 for x in predictions_q0_plau]

  if os.path.exists(args.q1):
    predictions_q1_mismatch = pd.Series([0]*n)
    temp = pd.Series([dim2num(s) for s in open(args.q1).read().splitlines()])
    temp[temp > 1] = 0
    predictions_q1_mismatch[idx_q1] = temp.tolist()
    # print(predictions_q1_mismatch)

  if os.path.exists(args.q3):
    predictions_q3_wrong_temporal_order = pd.Series([0]*n)
    predictions_q3_wrong_temporal_order[idx_q3] = [int(s) if s.isnumeric() else 0 for s in open(args.q3).read().splitlines()]

  predictions_q0 = pd.Series(predictions_q0_plau)
  predictions_q1 = 1 #- predictions_q1_mismatch
  predictions_q3 = 1 - predictions_q3_wrong_temporal_order
  predictions = predictions_q0*predictions_q1*predictions_q3
  # for i in range(10):
  #   print(predictions_q0[i])
    # if predictions_q0[i] == 1 and predictions[i] == 0:
    #   print(i)

  labels = data_df['label']
  rel_breakdown_f1 = []
  for r in 'xReact|oReact|xAttr|xIntent|xNeed'.split('|'):
    idx = (data_df['relation'] == r)
    score = f1_score(labels[idx], predictions[idx], average='binary')
    # print(r, score)
    rel_breakdown_f1.append(score)

  p, r, f1, support = list(precision_recall_fscore_support(labels, predictions, average='binary'))
  auc = roc_auc_score(labels, predictions)
  acc = accuracy_score(labels, predictions)
  score_report_template = '{:.2f}|'*10
  score_list = [auc, acc, p, r, f1] + rel_breakdown_f1
  print(score_report_template.format(*[x*100 for x in score_list]))

if __name__ == '__main__':
  args = parse()
  main(args)

'''
# atomic_davinci_randomcot_prompt3 -> rule of thumbs, checked by hand: "not likely to occur" -> False, not have "likely to occur" -> also False

# process zeroshot cot output
lines = open(args.q0).read().split('\n----------')
fout = open('temp.txt', 'w')
fout.write('\n----------'.join([l.replace('\n', '--') for l in lines]))
fout.close()

lines = open(args.q0).read().splitlines()
fout = open('temp.txt', 'w')
fout.write('\n'.join([l.strip().rsplit('-', 1)[1] for l in lines]))
fout.close()

how to post-process moe output
- typing -> check '\([1-9]\)'
- temporal -> 

def conversion(i, x):
  output = 0
  x = x.strip()
  if x.startswith('The second statement'):
    output = 1
  elif x.startswith('Statement 0 is more plausible') or x.startswith('I cannot') or x.startswith('The more plausible statement is:--0') or x.startswith('The first statement') or 'statement 0 is more plausible' in x or 'The correct answer is (0)':
    output = 0
  else:
    print(f'>{i}.', x)
  return output

[conversion(i, m) for i, m in enumerate([x for x in temp.splitlines() if '24' not in x])]

- zeroshot cot


- random cot


python prompting/llama_scoring.py \
  --q0 tst_llama2-7b_randomcot_prompt3.txt

python prompting/llama_scoring.py \
  --q0 tst_llama2-7b_naive_prompt3.txt

'''