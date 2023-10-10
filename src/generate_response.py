import argparse
import time

from util import *
from gpt_interface import *

from retrieval import retrieve_multilevel

MODEL = 'gpt-3.5-turbo'

PROMPT_RESPONSE_PARAPHRASE = read_file('docs/response_prompt_paraphrase.txt')
PROMPT_RESPONSE_UNCONSTRAINED = read_file('docs/response_prompt_unconstrained.txt')
PROMPT_RESPONSE_BASELINE = read_file('docs/response_prompt_baseline.txt')
PASSAGE_EXAMPLE = read_file('docs/passage_example.txt')

SCHEMA_SECTIONS = [':header', ':preconditions', ':static-conditions', ':postconditions', ':goals', ':episodes']



def subst_facts(prompt, dial_schema_facts, habitual_facts):
  dial_schema_user_str = '\n'.join(dial_schema_facts['user']) if dial_schema_facts['user'] else 'None'
  dial_schema_sys_str = '\n'.join(dial_schema_facts['sys']) if dial_schema_facts['sys'] else 'None'
  habitual_facts_str = '\n'.join(habitual_facts) if habitual_facts else 'None'
  prompt = prompt.replace('<dial-schema-facts-user>', dial_schema_user_str)
  prompt = prompt.replace('<dial-schema-facts-sys>', dial_schema_sys_str)
  prompt = prompt.replace('<habitual-facts>', habitual_facts_str)
  return prompt



def format_example(example):
  prompt = f'Relevant facts:\n<relevant-facts>\n\n<history-para>\nPerson B [ORIGINAL]: {example[2]}\n\n<history>\n<sys> [REWRITTEN]: {example[3]}'
  prompt = prompt.replace('<relevant-facts>', '\n'.join(example[0]))
  agents = ['<user>' if i%2==0 else '<sys>' for i in range(len(example[1]))]
  agents.reverse()
  agents_para = [agent.replace('<user>', 'Person A').replace('<sys>', 'Person B') for agent in agents]
  prompt = prompt.replace('<history>', '\n'.join([f'{a}: {h}' for a,h in zip(agents, example[1])]))
  prompt = prompt.replace('<history-para>', '\n'.join([f'{a}: {h}' for a,h in zip(agents_para, example[1])]))
  return prompt



def construct_prompt_paraphrase(sentence, user='User', sys='System', dial_schema_facts={'user':[], 'sys':[]}, habitual_facts=[], history=[], examples=[]):
  prompt = subst_facts(PROMPT_RESPONSE_PARAPHRASE, dial_schema_facts, habitual_facts)
  examples_str = [format_example(example) for example in examples]
  prompt = prompt.replace('<examples>', '\n\n'.join(examples_str))
  agents = ['<user>' if i%2==0 else '<sys>' for i in range(len(history))]
  agents.reverse()
  agents_para = [agent.replace('<user>', 'Person A').replace('<sys>', 'Person B') for agent in agents]
  prompt = prompt.replace('<history>', '\n'.join([f'{a}: {h}' for a,h in zip(agents, history)]))
  prompt = prompt.replace('<history-para>', '\n'.join([f'{a}: {h}' for a,h in zip(agents_para, history)]))
  prompt = prompt.replace('<sentence>', sentence)
  return prompt.replace('<user>', user).replace('<sys>', sys)



def construct_prompt_unconstrained(user='User', sys='System', dial_schema_facts={'user':[], 'sys':[]}, habitual_facts=[], history=[]):
  prompt = subst_facts(PROMPT_RESPONSE_UNCONSTRAINED, dial_schema_facts, habitual_facts)
  agents = ['<user>' if i%2==0 else '<sys>' for i in range(len(history))]
  prompt = prompt.replace('<history>', '\n'.join([f'{a}: {h}' for a,h in zip(agents, history)]))
  prompt = prompt + '\n<sys>:'
  return prompt.replace('<user>', user).replace('<sys>', sys)



def construct_prompt_baseline(user='User', sys='System', dial_schema_facts={'user':[], 'sys':[]}, history=[]):
  prompt = subst_facts(PROMPT_RESPONSE_BASELINE, dial_schema_facts, [])
  agents = ['<user>' if i%2==0 else '<sys>' for i in range(len(history))]
  prompt = prompt.replace('<history>', '\n'.join([f'{a}: {h}' for a,h in zip(agents, history)]))
  prompt = prompt + '\n<sys>:'
  return prompt.replace('<user>', user).replace('<sys>', sys)



def parse_generated_response(resp):
  text = None
  if 'choices' in resp and resp['choices'] and 'message' in resp['choices'][0]:
    text = resp['choices'][0]['message']['content'].strip()
    text = text.split('User:')[0].strip().split('System:')[-1].strip()
  return text



def generate_response(dial_schema_facts, habitual_facts, history, gold_response, examples, mode='unconstrained', generate=False):
  cost = 0.0

  if mode == 'paraphrase':
    prompt = construct_prompt_paraphrase(gold_response, dial_schema_facts=dial_schema_facts, habitual_facts=habitual_facts, history=history, examples=examples)
  elif mode == 'unconstrained':
    prompt = construct_prompt_unconstrained(dial_schema_facts=dial_schema_facts, habitual_facts=habitual_facts, history=history)
  else:
    prompt = construct_prompt_baseline(dial_schema_facts=dial_schema_facts, history=history)
  avg_resp_len = len(PASSAGE_EXAMPLE)
  cost += cost_chat_gpt(MODEL, prompt, avg_resp_len)
  
  response = None
  if generate:
    response = parse_generated_response(generate_chat_gpt(MODEL, prompt))
  else:
    response = PASSAGE_EXAMPLE

  return cost, response



def process_turn(persona, schemas, history, gold_response, examples, k=3, n=5, generate=False):

  examples_subset = examples[:min(k, len(examples))]
  paraphrase_examples = list(zip(examples_subset['facts'],examples_subset['context'], examples_subset['utterance'], examples_subset['paraphrase']))

  if generate:
    habitual_facts = retrieve_multilevel(history[-1], schemas, header=True, n=n)
  else:
    habitual_facts = schemas[0][0:(n+1)]

  dial_schema_facts = {'user':[], 'sys':persona}
  generation_args = [dial_schema_facts, habitual_facts, history, gold_response, paraphrase_examples]

  c1, response_paraphrase = generate_response(*generation_args, mode='paraphrase', generate=generate)
  c2, response_unconstrained = generate_response(*generation_args, mode='unconstrained', generate=generate)
  c3, response_baseline = generate_response(*generation_args, mode='baseline', generate=generate)

  return response_paraphrase, response_unconstrained, response_baseline, (c1+c2+c3)



def preprocess_schemas(schemas_raw):
  schemas = []
  for schema in schemas_raw:
    for section in SCHEMA_SECTIONS:
      schema = schema.replace(section, '')
    schemas.append([fact.replace('- ', '') for fact in schema.split('\n') if fact])
  return schemas



def main(args):
  paraphrase_examples = load_example_data('examples/paraphrase.json')
  validation = load_data(args.dataset, f=args.percent)

  print('\n\n\n')
  print(f'Generating responses for {args.percent}% of PersonaChat schema validation set (generate={args.generate})')

  cost = 0.
  cache = {}
  out = []

  for idx, d in enumerate(validation):
    
    persona = d['personality']
    passages = d['passages']
    schemas_raw = d['schemas']
    candidates = d['candidates']
    history = d['history']
    conv_id = d['conv_id']
    utterance_idx = d['utterance_idx']

    if conv_id in cache:
      schemas = cache[conv_id]
    else:
      schemas = preprocess_schemas(schemas_raw)
      
    response_paraphrase, response_unconstrained, response_baseline, c = process_turn(persona, schemas, history, candidates[-1], paraphrase_examples, generate=args.generate)
    cost += c

    out.append({'personality':persona, 'passages':passages, 'schemas':schemas_raw,
                'candidates':candidates, 'history':history, 'response_paraphrase':response_paraphrase,
                'response_unconstrained':response_unconstrained, 'response_baseline':response_baseline,
                'conv_id':conv_id, 'utterance_idx':utterance_idx})
    
    if (1+idx) % args.verbosity == 0:
      print(f'Item {idx+1}/{len(validation)}')
    
    if args.generate and (1+idx) % args.checkpoint_iter == 0:
      print(f'Saving checkpoint')
      write_data(args.output_filename, out)
  
  print(f'Total cost: ${round(cost,3)}')

  if args.generate:
    write_data(args.output_filename, out)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--generate', action='store_true', default=False, help='Whether to do a generation run')
  parser.add_argument('--dataset', type=str, default='data/personachat/personachat-schemas.json')
  parser.add_argument('--percent', type=int, default=100, help='The percent of the PersonaChat dataset to use for generation')
  parser.add_argument('--verbosity', type=int, default=10, help='Number of iterations to print status')
  parser.add_argument('--checkpoint-iter', type=int, default=50, help='Number of iterations to save current dataset checkpoint')
  parser.add_argument('--output-filename', type=str, default='data/personachat/personachat-schemas-responses.json', help='JSON filename to store generated data')

  args = parser.parse_args()

  time1 = time.time()
  main(args)
  time2 = time.time()
  print_readable_time_delta(time1, time2)