from nltk.stem import WordNetLemmatizer
import pandas as pd
import random
from tqdm import tqdm

lemmatizer = WordNetLemmatizer()
seed = 2023

naive_prompt0 = "Answer whether the following statement is plausible. Answer with only Yes or No:\n"
naive_prompt1 = "Judge the following statement if it's likely to occur, only answer 'True' or 'False':\n"
modal_verbs = ['will', 'would', 'can', 'could', 'shall', 'should', 'must', 'might', 'may']

# only test the generalizability of current setting
templates = {
  'xReact': '{}, thus as a result on PersonX\'s emotion, {}',
  'oReact': '{}, thus as a result on PersonY\'s emotion, {}',
  'xAttr': '{}, thus it can be seen about PersonX\'s attribute that {}',
  'xIntent': '{}, thus it can be seen about PersonX\'s intention that {}',
  'xNeed': 'The event {} will not happen unless {}', # or '{ will not} unless'
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
def fix_tail(data_df):
  for i in tqdm(range(len(data_df)), total=len(data_df)):
    rel, tail = data_df['relation'][i], data_df['tail'][i].lower().strip()
    if tail[-1] == '.':
      tail = tail[:-1]

    if rel == 'xReact':
      tail = 'PersonX feel ' + tail

    elif rel == 'oReact':
      tail = 'PersonY feel ' + tail

    elif rel == 'xAttr':
      tail = 'PersonX be ' + tail

    elif rel == 'xIntent':
      if tail.startswith('to '):
        tail = 'PersonX intended ' + tail
      else:
        v, _ = (tail + ' ').split(' ', 1)
        v = lemmatizer.lemmatize(v, pos='v')
        tail = 'PersonX intended to ' + v + ' ' + _

    elif rel == 'xNeed':
      if tail.startswith('to '):
        tail = 'PersonX' + tail[2:]
      else:
        tail = 'PersonX ' + tail

    data_df['tail'][i] = tail


def extract_and_negsample_atomic2020():
  # data = pd.read_csv("test.tsv", sep='\t')
  # data.fillna('none', inplace=True)
  # data.columns = ['head', 'relation', 'tail']
  # data = data[data['relation'].isin(templates.keys())]
  # data = data[data['tail'] != 'none'].reset_index(drop=True)
  # fix_tail(data)
  # data.to_csv('standardizing_tail.csv')
  data = pd.read_csv('standardizing_tail.csv')

  # random sample data
  n = 1000
  random.seed(seed)
  list_head = random.sample(list(data['head'].unique()), n)
  final_df = pd.DataFrame({'head': [], 'relation': [], 'tail': []})
  backup_tail, avoid_relation = [], []

  for i, head in enumerate(list_head):
    # print(i, head)
    temp = data[data['head'] == head].reset_index(drop=True)
    triple = temp.sample(1, random_state=seed, ignore_index=True)
    try:
      backup_tail.append(temp[temp['relation'] != triple['relation'][0]]['tail'].sample(1, random_state=seed))
    except:
      backup_tail.append(data[data['relation'] != triple['relation'][0]]['tail'].sample(1, random_state=seed))
    avoid_relation.append(temp[temp['tail'] == triple['tail'][0]]['relation'])
    final_df = final_df.append(triple, ignore_index=True)

  final_df['avoid_relation'] = avoid_relation
  final_df['backup_tail'] = backup_tail

  modification = [0]*(n//2) + [1]*(n//4) + [2]*(n//4)
  random.seed(seed)
  random.shuffle(modification)
  final_df['modification'] = modification
  final_df['label'] = (final_df['modification'] <= 0).astype(int)

  # modify data
  '''
  chance to apply modification 50%
  25%  O. replace relation
  25%  S. replace tail
  only these two categories when we fix head
  '''

  random.seed(seed)
  for i in range(n):
    j = final_df.iloc[i, -2]
    if j == 1: # modify relation
      final_df.iloc[i, 1] = random.sample(sorted(list(set(templates.keys()).difference(final_df['avoid_relation'][i]))), 1)
    elif j == 2: # modify tail
      final_df.iloc[i, 2] = final_df['backup_tail'][i]

  final_df[['head', 'relation', 'tail', 'label', 'modification']].to_csv('test_downsample_with_negative_sampling.csv', index=False)
  # final_df.to_csv('test_downsample_with_negative_sampling.csv', index=False)


def convert_data_format():
  selected = pd.read_csv('test_downsample_with_negative_sampling.csv')
  selected['assertion'] = [templates[rel].format(head, tail) for
    head, rel, tail in zip(selected['head'], selected['relation'], selected['tail'])]
  selected.to_csv(f'generalizability_test_atomic2020_w_negsample.csv', index=False)


if __name__ == '__main__':
  # extract_and_negsample_atomic2020()
  convert_data_format()