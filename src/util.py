import json
import datetime
import dateutil.relativedelta
from datasets import load_dataset

SEED = 42

def read_file(fname):
  with open(fname, 'r') as f:
    return f.read()
  

def write_data(fname, data):
  with open(fname, "w") as f:
    json.dump(data, f)


def load_example_data(fname, shuffle=True):
  dataset = load_dataset('json', data_files=fname, split='train')
  if shuffle:
    dataset = dataset.shuffle(seed=SEED)
  return dataset


def load_personachat_data(split='validation', f=100, shuffle=True):
  dataset = load_dataset("bavard/personachat_truecased", split=f'{split}[:{f}%]')
  if shuffle:
    dataset = dataset.shuffle(seed=SEED)
  return dataset


def load_data(fname, n=None, f=100, shuffle=True):
  if n:
    dataset = load_dataset('json', data_files=fname, split=f'train[:{n}]')
  else:
    dataset = load_dataset('json', data_files=fname, split=f'train[:{f}%]')
  if shuffle:
    dataset = dataset.shuffle(seed=SEED)
  return dataset


def print_readable_time_delta(time1, time2):
  dt1 = datetime.datetime.fromtimestamp(time1)
  dt2 = datetime.datetime.fromtimestamp(time2)
  rd = dateutil.relativedelta.relativedelta (dt2, dt1)
  print(rd)