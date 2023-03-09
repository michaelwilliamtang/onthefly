# from COS597.data_util import load_data, get_dataloader, prepare_data
from datasets import load_dataset
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import pickle

from rank_bm25 import BM25Okapi

import argparse
import csv
import json
import math
import sys, os
import time

import torch
import torch.nn.functional as F
import torch.optim
import torch.optim as optim
import torch.utils.data as data
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torchtext import data as torchtext_data
from torchtext import datasets
from tqdm import tqdm, trange
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline

# import PPLM2
# from PPLM2.pplm_classification_head import ClassificationHead
# from PPLM2.run_pplm_discrim_train import *
# from PPLM2.run_pplm import *
# from run_pplm import *
import time

from evaluate import load
import logging

import nltk
# nltk.download('punkt')
import nltk.data
from nltk.corpus import stopwords

from SimCSE.simcse.tool import SimCSE

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


########################
# BASICALLY CONSTANTSs
########################

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EPSILON = 1e-10
TRUE_NEUTRAL_PROMPT = 0
NEUTRAL_PROMPT = 1
HELPFUL_PROMPT = 2
ADVERSARIAL_PROMPT = 3

pretrained_model = 'gpt2-medium'
no_cuda = False
device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
add_eos_token = pretrained_model.startswith("gpt2")
max_length_seq = 100

########################
# VARIABLES, CHANGE ME!
########################
DEBUG = False
if DEBUG:
    n_train = 10000 # number of retrieval candidates to load if DEBUG
    n_val = 10 # number of eval queries to run if DEBUG
    n_pos_refs = 8 # only applies to the bm25 atm
else:
    n_train = 10000
    n_val = 300
    n_pos_refs = 20 # only applies to the bm25 atm
# note that if pickle exists, will not be recomputed even if this is True
recompute_bm25 = False
recompute_simcse = True
retriever_desc = 'cc_news_text_descriptions_10k'
# desc = 'simcse_4_0p25_cc_news_text_descriptions_10k'
desc = 'title_simcse_4_0p25_cc_news_text_descriptions_10k'
n_sents = 1 # number of sentences per batch for on-the-fly training

MIN_REF_SIMILARITY = 0.4 # for simcse candidate retrieval
# RETRIEVER = 'simcse'
RETRIEVER = 'none'

FILTER_KEYWORDS = True # further filter extracted keywords by simcse similarity to query
MIN_WORD_SIMILARITY = 0.25 # for filtering keywords

MIN_BAG_SIZE = 4

# PROMPT_TYPE = 0
# n_helpful_prefix_chars = 200 # should basically be large enough to cover the chars in n_helpful_prefix_words
# n_helpful_prefix_words = 4 # number of words provided in helpful prompt

################################################################################################
# PREP DATA
################################################################################################

print('Data prep section! ------------')
print('DEBUG =', DEBUG)

# load, filter, shuffle
# filter modified from COS597/run_baseline.py
print('Preparing data...')
def filter_dataset(data):
    return data.filter(
            lambda d: d["url"].startswith(("https:","www"))
        ).filter(
            lambda d: len(d["title"]) > 30
        ).filter(
            lambda d: len(d["description"]) > 60
        )
cc_raw_data = load_dataset('cc_news', split='train')
n_train_before_filter = len(cc_raw_data)
cc_raw_data = filter_dataset(cc_raw_data)
cc_raw_data = cc_raw_data.shuffle(seed=SEED)
print('{} total examples ({} before filtering)'.format(len(cc_raw_data), n_train_before_filter))

# split
assert n_train + n_val <= len(cc_raw_data)
cc_data = cc_raw_data.select(range(n_train))
cc_data_val = cc_raw_data.select(range(len(cc_raw_data)-n_val,len(cc_raw_data)))

print('{} train; {} val'.format(n_train, n_val))
print('Done\n')

################################################################################################
# PREP MODELS
################################################################################################

print('Model prep section! ------------')

# sentence tokenizer
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# turn docs into a list of triples (or groups of n_sents) of sentences in those docs
def get_sent_groups(refs):
    ref_X = []
    for x in refs:
        sents = sent_tokenizer.tokenize(x)
        sent_groups = [' '.join(sents[i:i+n_sents]) for i in range(0, len(sents), n_sents)]
        ref_X.extend(sent_groups)
    return ref_X

# keyword classifier
# keyword_classifier = pipeline(task="feature-extraction", model="yanekyuk/bert-uncased-keyword-extractor")
keyword_classifier = pipeline(model="yanekyuk/bert-uncased-keyword-extractor")
stopwords = set(stopwords.words('english'))

# bm25 retriever, only needed for fetching when it is the RETRIEVER
if RETRIEVER == 'bm25':
    bm25_path = 'bm25_' + retriever_desc + '.pickle'
    if recompute_bm25 and not os.path.isfile(bm25_path):
        print('Preparing bm25...')
        corpus = [x['text'] for x in cc_data]
        tokenized_corpus = [nltk.word_tokenize(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        with open(bm25_path, 'wb') as f:
            pickle.dump([bm25, tokenized_corpus, corpus], f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved bm25 at', bm25_path)
    else:
        with open(bm25_path, 'rb') as f:
            bm25, tokenized_corpus, corpus = pickle.load(f)
        print('Read cached retriever at', retriever_desc)

# simcse retriever, always needed
simcse_path = 'simcse_' + retriever_desc + '.pickle'
if recompute_simcse and not os.path.isfile(simcse_path):
    print('Preparing simcse...')
    corpus = [x['text'] for x in cc_data]
    simcse = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    sents = get_sent_groups(corpus)
    simcse.build_index(sents)
    with open(simcse_path, 'wb') as f:
        pickle.dump([simcse, corpus], f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved simcse at', simcse_path)
else:
    with open(simcse_path, 'rb') as f:
        simcse, corpus = pickle.load(f)
    print('Read cached retriever at', retriever_desc)

################################################################################################
# UTILITY FUNCTIONS FOR ON-THE-FLY CLASSIFICATION
################################################################################################

def get_keywords(pos_refs):
    # with open('religion.txt', "r") as f:
    #     custom_bag = f.read().strip().split("\n")
    # return custom_bag

    # extract
    words = []
    for ref in pos_refs:
        res = keyword_classifier(ref)
        words.extend([x['word'] for x in res])

    # dedup
    words = [x for x in set(words)]

    # filter triviality
    custom_bag = []
    for word in words:
        if word not in stopwords and len(word) >= 2:
            custom_bag.append(word)

    return custom_bag

def filter_relevant_keywords(query, custom_bag):
    bag = []
    for word in custom_bag:
        blockPrint()
        if simcse.similarity(word, query) > MIN_WORD_SIMILARITY:
            bag.append(word)
        enablePrint()

    return bag

def fetch_refs_bm25(query):
    # TODO allow this without having actual text
    # retrieve candidates
    tokenized_query = nltk.word_tokenize(query)
    neighbors = bm25.get_top_n(tokenized_query, range(len(corpus)), n=2*n_pos_refs)
    # scores = bm25.get_scores(tokenized_query)

    # dedup candidates
    seen_titles = set()
    dedup_neighbors = []
    for i in neighbors:
        if cc_data[i]['title'] not in seen_titles:
            dedup_neighbors.append(i)
            seen_titles.add(cc_data[i]['title'])
    # print('{} neighbors ({} before dedup)'.format(len(dedup_neighbors), len(neighbors)))
    # print('Neighbor titles:')
    # print(cc_data[dedup_neighbors]['title'])

    # create on-the-fly train + test datasets using candidates
    pos_refs = cc_data[dedup_neighbors[:n_pos_refs]]['text']
    return pos_refs

def fetch_refs_simcse(query):
    blockPrint()
    neighbors = simcse.search(query)
    enablePrint()
    filtered_neighbors = []
    for pair in neighbors:
        if pair[1] > MIN_REF_SIMILARITY:
            filtered_neighbors.append(pair[0])
    print('Found {} neighbors'.format(len(filtered_neighbors)))
    return filtered_neighbors

def get_bag(query):
    print('GETTING BAG, query:', query)

    if RETRIEVER == 'simcse':
        pos_refs = fetch_refs_simcse(query)
    elif RETRIEVER == 'bm25':
        pos_refs = fetch_refs_bm25(query)
    elif RETRIEVER == 'none':
        pos_refs = []
    else:
        assert False # invalid input
    pos_refs.append(query)

    # create on-the-fly BoW
    custom_bag = get_keywords(pos_refs)
    if FILTER_KEYWORDS:
        custom_bag = filter_relevant_keywords(query, custom_bag)



    # generate using BoW
    # if PROMPT_TYPE == TRUE_NEUTRAL_PROMPT:
    #     prompt = 'BREAKING NEWS: Today'
    # else:
    #     if PROMPT_TYPE == NEUTRAL_PROMPT:
    #         prefix = 'BREAKING NEWS: Today'
    #     elif PROMPT_TYPE == HELPFUL_PROMPT:
    #         prefix = ' '.join(nltk.word_tokenize(actual_text[:n_helpful_prefix_chars])[:n_helpful_prefix_words])
    #     elif PROMPT_TYPE == ADVERSARIAL_PROMPT:
    #         prefix = 'Cinderella'
    #     else:
    #         print('Invalid prompt type')

    #     prompt = 'Generate a long article based on its title. Title: ' + query + '. Article Text: ' + prefix

    # generation = run_one_pplm_example(
    #         pretrained_model=pretrained_model,
    #         cond_text=prompt,
    #         uncond=False,
    #         bag_of_words='religion',
    #         discrim=None,
    #         discrim_weights=None,
    #         discrim_meta=None,
    #         # class_label=-1,
    #         length=100,
    #         stepsize=0.02,
    #         temperature=1.0,
    #         top_k=10,
    #         sample=True,
    #         num_iterations=1,
    #         grad_length=10000,
    #         horizon_length=1,
    #         window_length=0,
    #         decay=False,
    #         gamma=1, # 1.5
    #         gm_scale=0.6, # 0.9
    #         kl_scale=0.06, # 0.01
    #         seed=SEED,
    #         no_cuda=False,
    #         colorama=False,
    #         verbosity='quiet',
    #         tokenizer=None,
    #         model=None,
    #         # on_the_fly=True, # NEW PARAMS
    #         # classifier=discriminator.get_classifier(), # NEW PARAMS
    #         # class_id=1, # NEW PARAMS
    #         # ref_emb=None # NEW PARAMS
    #         bag_of_words_direct=custom_bag
    # )

    # return generation
    # print('PROMPT:')
    # print(prompt)
    print('BAG:', custom_bag)
    # for x in custom_bag:
    #     print(x)
    # print()

    return custom_bag

start_time = time.time()

examples = np.array(cc_data_val)
bags = np.array([get_bag(x['title']) for x in cc_data_val])
# X = [x['title'] for x in cc_data_val]
# y = [x['text'] for x in cc_data_val]
# y_hat = [generate_one(X_i, y_i) for X_i, y_i in zip(X, y)]

print("--- %s seconds ---" % (time.time() - start_time))

inds = []
for idx, x in enumerate(bags):
    if len(x) >= MIN_BAG_SIZE:
        inds.append(idx)
print('{} bags out of {} with at least {}'.format(len(inds), len(bags), MIN_BAG_SIZE))
# print(inds)

with open('bags_' + desc + '.pickle', 'wb') as f:
    pickle.dump([examples[inds], bags[inds], examples, bags, inds], f, protocol=pickle.HIGHEST_PROTOCOL)

print('Saved resulting bags at', desc)