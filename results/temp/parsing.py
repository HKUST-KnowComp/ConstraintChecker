import os

def refine(pred, fname):
  sub = 'True' if 'prompt3' in fname else 'Yes'
  pred = pred.strip()
  if '--' in pred:
    pred = pred.rsplit('--', 1)[1]
  return sub if pred.lower() == 'plausible' else pred

for (root, dirs, files) in os.walk('.'):
  for fname in files:
    print(fname)
    # if fname.startswith('turbo'):
    #   os.rename(fname, fname.replace('turbo', 'tst_gpt3.5-turbo'))
    # if fname.startswith('davinci'):
    #   os.rename(fname, 'tst_'+fname)
    # continue

    if fname.startswith('tst_gpt3.5-turbo'):
      fin = open(fname, 'r').read().splitlines()
      predictions = [refine(p, fname) for x in fin if '196' not in x for p in x.split(',')]
      print(len(predictions))
      # open(fname, 'w').write('\n'.join(predictions))

    elif fname.startswith('tst_davinci'):
      fin = open(fname, 'r').read().splitlines()
      predictions = [refine(x, fname) for x in fin if '196' not in x]
      print(len(predictions))
      # open(fname, 'w').write('\n'.join(predictions))
