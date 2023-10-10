import argparse
import time
import numpy as np

from util import *
from gpt_interface import *
  
MODEL = 'gpt-3.5-turbo'

MODEL_TYPES = {
  'text-davinci-003' : 'InstructGPT',
  'gpt-3.5-turbo' : 'ChatGPT'
}

PROMPT_PASSAGE = read_file('docs/passage_prompt.txt')
PROMPT_SCHEMA = read_file('docs/schema_prompt.txt')
PROMPT_EXAMPLE = ["My grandmother used to bake pies on Saturdays.", read_file('docs/passage_example.txt'), read_file('docs/schema_example.txt')]

SCHEMA_SECTIONS = [':header', ':preconditions', ':static-conditions', ':postconditions', ':goals', ':episodes']



def parse_generated_passages(resp):
  text = None
  if 'choices' in resp and resp['choices'] and 'message' in resp['choices'][0]:
    text = resp['choices'][0]['message']['content'].strip().split('\n')[-1]
  return text



def generate_passages(facts, examples=[PROMPT_EXAMPLE[0:2]], generate=False):
  passages = []
  cost = 0.0

  for idx, fact in enumerate(facts):
    avg_resp_len = np.mean([len(example[1]) for example in examples])
    cost += cost_chat_gpt(MODEL, fact, avg_resp_len, preamble=PROMPT_PASSAGE, examples=examples)
    
    passage = None
    if generate:
      passage = parse_generated_passages(generate_chat_gpt(MODEL, fact, preamble=PROMPT_PASSAGE, examples=examples))
    else:
      passage = examples[0][1]

    passages.append(passage)
      
  return cost, passages



def parse_generated_schemas(resp):
  text = None
  if 'choices' in resp and resp['choices'] and 'message' in resp['choices'][0]:
    text = resp['choices'][0]['message']['content'].replace(':event', ':header').replace('- ', '').strip()
    if not all([sec in text for sec in SCHEMA_SECTIONS]):
      return None
  return text



def generate_schemas(facts, passages, examples=[PROMPT_EXAMPLE[1:3]], n_restarts=5, generate=False):
  schemas = []
  cost = 0.0

  for idx, (fact, passage) in enumerate(zip(facts, passages)):
    avg_resp_len = np.mean([len(example[1]) for example in examples])
    prompt = (fact + ' ' + passage) if passage else fact
    cost += cost_chat_gpt(MODEL, prompt, avg_resp_len, preamble=PROMPT_SCHEMA, examples=examples)
    
    schema = None
    i = 0
    while schema is None and i < n_restarts:
      if generate:
        schema = parse_generated_schemas(generate_chat_gpt(MODEL, prompt, preamble=PROMPT_SCHEMA, examples=examples))
      else:
        schema = examples[0][1]
      if i >= 3:
        print('~ warning: needing to retry schema three times; this might indicate a bug')
      i += 1

    schemas.append(schema)
      
  return cost, schemas



def process_persona(examples, persona, k=2, generate=False):

  examples_subset = examples[:min(k, len(examples))]
  passage_examples = list(zip(examples_subset['fact'], examples_subset['passage']))

  c1, passages = generate_passages(persona, passage_examples, generate=generate)
  c2, schemas = generate_schemas(persona, passages, generate=generate)

  return passages, schemas, (c1+c2)



def main(args):
  passage_examples = load_example_data('examples/passage.json')

  if args.dataset == 'personachat':
    dataset = load_personachat_data(split='validation', f=args.percent)
  else:
    dataset = load_data(args.dataset, f=args.percent, shuffle=False)

  print('\n\n\n')
  print(f'Inducing passages and schemas for {args.percent}% of dataset (generate={args.generate})')

  cost = 0.
  cache = {}
  out = []

  for idx, d in enumerate(dataset):
    
    persona = d['personality']
    candidates = d['candidates']
    history = d['history']
    conv_id = d['conv_id']
    utterance_idx = d['utterance_idx']

    if conv_id in cache:
      passages, schemas = cache[conv_id]
    else:
      passages, schemas, c = process_persona(passage_examples, persona, generate=args.generate)
      cache[conv_id] = (passages, schemas)
      cost += c

    out.append({'personality':persona, 'passages':passages, 'schemas':schemas,
                'candidates':candidates, 'history':history,
                'conv_id':conv_id, 'utterance_idx':utterance_idx})
    
    if (1+idx) % args.verbosity == 0:
      print(f'Item {idx+1}/{len(dataset)}')
    
    if args.generate and (1+idx) % args.checkpoint_iter == 0:
      print(f'Saving checkpoint')
      write_data(args.output_filename, out)
  
  print(f'Total cost: ${round(cost,3)}')

  if args.generate:
    write_data(args.output_filename, out)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--generate', action='store_true', default=False, help='Whether to do a generation run')
  parser.add_argument('--dataset', type=str, default='personachat', help='The dataset to use (either "personachat" or a local filename)')
  parser.add_argument('--percent', type=int, default=1, help='The percent of the PersonaChat dataset to use for generation')
  parser.add_argument('--verbosity', type=int, default=10, help='Number of iterations to print status')
  parser.add_argument('--checkpoint-iter', type=int, default=50, help='Number of iterations to save current dataset checkpoint')
  parser.add_argument('--output-filename', type=str, default='data/personachat/personachat-schemas.json', help='JSON filename to store generated data')

  args = parser.parse_args()

  time1 = time.time()
  main(args)
  time2 = time.time()
  print_readable_time_delta(time1, time2)