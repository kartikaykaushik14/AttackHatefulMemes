import pandas as pd

new = pd.read_json("/scratch/mmk9369/ml-cybersec/datasets/hateful_memes/defaults/annotations/train.jsonl", lines=True)

df = new.copy()
for i in range(len(df)):
  df.loc[i, 'text'] = df.loc[i, 'text'].replace(' ', '')
for i in range(len(df)):
  df.loc[i, 'img'] = df.loc[i, 'img'].replace('img', 'perturbed')

frames = [new, df]
result = pd.concat(frames)
result.to_json('/scratch/mmk9369/ml-cybersec/datasets/hateful_memes/defaults/annotations/adv_retrain.jsonl', orient='records', lines=True)

new = pd.read_json("/scratch/mmk9369/ml-cybersec/datasets/hateful_memes/defaults/annotations/adv_retrain.jsonl", lines=True)
print(new.head)