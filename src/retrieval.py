import os
import json
import requests
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer

from util import *
from gpt_interface import *

HF_EMBEDDING_PREFIX = "https://api-inference.huggingface.co/pipeline/feature-extraction/"
HF_SIMILARITY_PREFIX = "https://api-inference.huggingface.co/models/"

MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
MODEL = SentenceTransformer(MODEL_NAME)

HF_HEADER = { "Authorization" : f"Bearer {read_file('_keys/huggingface.txt')}" }

os.environ['TOKENIZERS_PARALLELISM'] = 'false'



def query_api(api_url, query):
  data = json.dumps(query)
  response = requests.request("POST", api_url, headers=HF_HEADER, data=data)
  return json.loads(response.content.decode("utf-8"))
    


def embedding_api(documents):
  query = {"inputs" : documents, "options" : {"wait_for_model" : True}}
  return query_api(HF_EMBEDDING_PREFIX+MODEL_NAME, query)



def similarity_api(text, documents):
  query = {"inputs" : {"source_sentence" : text, "sentences" : documents}, "options" : {"wait_for_model" : True}}
  return query_api(HF_SIMILARITY_PREFIX+MODEL_NAME, query)



def compute_embeddings(documents, use_api):
  if use_api:
    return embedding_api(documents)
  else:
    return MODEL.encode(documents)



def sim(x, y):
  return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))



def retrieve_multilevel(text, document_sets, n=5, header=False, use_api=False):
  document_agg = np.array([' '.join(ds) for ds in document_sets])
  document_sets = np.array([np.array(ds) for ds in document_sets], dtype=object)

  e_ds = compute_embeddings(document_agg, use_api=use_api)
  e_t = compute_embeddings(text, use_api=use_api)

  ds_r = document_sets[np.argmax(sim(e_ds, e_t))]
  if header:
    documents = ds_r[1:]
  else:
    documents = ds_r

  e_d = compute_embeddings(documents, use_api=use_api)

  scores = sim(e_d, e_t)
  scores_top = np.argsort(scores)[:-(min(n, len(scores))+1):-1]

  if header:
    return [ds_r[0]] + list(documents[scores_top])
  else:
    return list(documents[scores_top])