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
import os
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

import PPLM
from PPLM.pplm_classification_head import ClassificationHead
from PPLM.run_pplm_discrim_train import *
from PPLM.run_pplm import *
import time

from evaluate import load
import logging

import nltk
# nltk.download('punkt')
import nltk.data

########################
# BASICALLY CONSTANTS
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
cached = False
no_cuda = False
save_model = False
log_interval = 10
device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
add_eos_token = pretrained_model.startswith("gpt2")
max_length_seq = 100

########################
# VARIABLES, CHANGE ME!
########################
DEBUG = True
if DEBUG:
    n_train = 3000 # number of retrieval candidates to load if DEBUG
    n_val = 1 # number of eval queries to run if DEBUG
    ref_epochs = 5
    n_pos_refs = 3
else:
    n_train = 30000
    n_val = 100
    ref_epochs = 10
    n_pos_refs = 8
ref_batch_size = 8
desc = 'cc_news_text_1000'
ref_lr = 0.001 # learning rate
n_sents = 1 # number of sentences per batch for on-the-fly training
n_neg_refs = 2 * n_pos_refs + 10 # actually will just use enough to match the pos refs, so we provide more here

PROMPT_TYPE = 0
n_helpful_prefix_chars = 200 # should basically be large enough to cover the chars in n_helpful_prefix_words
n_helpful_prefix_words = 4 # number of words provided in helpful prompt

################################################################################################
# PREP DATA
################################################################################################

print('Data prep section! ------------')
print('DEBUG =', DEBUG)

# load, filter, shuffle
# filter modified from COS597/run_baseline.py
def filter_dataset(data):
    return data.filter(
            lambda d: d["url"].startswith(("https:","www"))
        ).filter(
            lambda d: len(d["title"]) > 30
        )
        # .filter(
        #     lambda d: len(d["summary"]) > 60
        # )
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

################################################################################################
# PREP MODELS
################################################################################################

print('Model prep section! ------------')

# sentence tokenizer
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# retriever
if recompute_retriever:
    corpus = [x['text'] for x in cc_data]
    tokenized_corpus = [nltk.word_tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    with open('bm25_' + desc + '.pickle', 'wb') as f:
        pickle.dump([bm25, tokenized_corpus, corpus], f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open('bm25_' + desc + '.pickle', 'rb') as f:
        bm25, tokenized_corpus, corpus = pickle.load(f)

# pretrained LM
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
model = GPT2LMHeadModel.from_pretrained(pretrained_model)
model2 = GPT2LMHeadModel.from_pretrained(
    pretrained_model,
    output_hidden_states=True
)
model2.to(device)
model2.eval()

# freeze GPT-2 weights
for param in model2.parameters():
    param.requires_grad = False

# discriminator architecture for on-the-fly classifier
idx2class = ["general", "in_topic"]
class2idx = {c: i for i, c in enumerate(idx2class)}
discriminator = Discriminator(
    class_size=len(idx2class),
    pretrained_model=pretrained_model,
    cached_mode=cached,
    device=device,
#     tokenizer=tokenizer,
#     model=model,
).to(device)
discriminator_meta = {
    "class_size": len(idx2class),
    "embed_size": discriminator.embed_size,
    "pretrained_model": pretrained_model,
    "class_vocab": class2idx,
    "default_class": 0,
}


################################################################################################
# UTILITY FUNCTIONS FOR ON-THE-FLY CLASSIFICATION
################################################################################################

# turn docs into a list of triples (or groups of n_sents) of sentences in those docs
def get_sent_groups(refs):
    ref_X = []
    for x in refs:
        sents = sent_tokenizer.tokenize(x)
        sent_groups = [' '.join(sents[i:i+n_sents]) for i in range(0, len(sents), n_sents)]
        ref_X.extend(sent_groups)
    return ref_X

# creates dataset for on-the-fly classifier
def create_dataset(pos_refs, neg_refs):
    # turn positive and negative docs into a single list of groups of sentences
    ref_X_pos = get_sent_groups(pos_refs)
    ref_X_neg = get_sent_groups(neg_refs)
    ref_X = ref_X_pos + ref_X_neg
    ref_y = [1] * len(ref_X_pos) + [0] * len(ref_X_neg)
    n_pos = len(ref_X_pos)

    # TODO remove this flakiness
    assert len(ref_X) >= 2*n_pos

    ref_X = ref_X[:2*n_pos] # balance pos and neg
    ref_y = ref_y[:2*n_pos] # balance pos and neg

    # randomize
    perm = np.random.permutation(len(ref_X))
    ref_X = np.array(ref_X)[perm]
    ref_y = np.array(ref_y)[perm]

    # convert dataset to tensors
    n_examples = len(ref_X)
    ref_X_tensor = []
    ref_y_tensor = []
    for i in range(len(ref_X)):
        seq = discriminator.tokenizer.encode(ref_X[i])
        if len(seq) >= max_length_seq:
            print("Line {} is longer than maximum length {}, as it has length {}, truncating".format(
                i, max_length_seq, len(seq)
            ))
            seq = seq[:max_length_seq]
        if add_eos_token:
            seq = [50256] + seq
        seq = torch.tensor(
            seq, device=device, dtype=torch.long
        )
    #     else:
    #         print("Line {} is longer than maximum length {}".format(
    #             i, max_length_seq
    #         ))
    #         continue
        ref_X_tensor.append(seq)
        ref_y_tensor.append(ref_y[i])

    n_examples = len(ref_X_tensor) # may be less than before
    print('final number of examples =', n_examples)

    return Dataset(ref_X_tensor, ref_y_tensor)

def get_dataloaders(pos_refs, neg_refs, test_pos_refs, test_neg_refs, batch_size=ref_batch_size):
    train_dataset = create_dataset(pos_refs, neg_refs)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=collate_fn)

    test_dataset = create_dataset(test_pos_refs, test_neg_refs)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)
    return train_loader, test_loader

def train_discrim(discrim, train_loader, test_loader, lr=ref_lr, epochs=ref_epochs):
    start = time.time()
    optimizer = optim.Adam(discrim.parameters(), lr=lr)
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        start = time.time()
        print("\nEpoch", epoch + 1)

        train_epoch(
            discriminator=discrim,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            log_interval=log_interval,
            device=device
        )
        test_loss, test_accuracy = evaluate_performance(
            data_loader=test_loader,
            discriminator=discrim,
            device=device
        )

        end = time.time()
        print("Epoch took: {:.3f}s".format(end - start))

        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print("\nExample prediction")
        predict(example_sentence, discrim, idx2class,
                cached=cached, device=device)

    #     if save_model:
    #         # torch.save(discriminator.state_dict(),
    #         #           "{}_discriminator_{}.pt".format(
    #         #               args.dataset, epoch + 1
    #         #               ))
    #         torch.save(discriminator.get_classifier().state_dict(),
    #                    classifier_head_fp_pattern.format(epoch + 1))

    min_loss = float("inf")
    min_loss_epoch = 0
    max_acc = 0.0
    max_acc_epoch = 0
    print("Test performance per epoch")
    print("epoch\tloss\tacc")
    for e, (loss, acc) in enumerate(zip(test_losses, test_accuracies)):
        print("{}\t{}\t{}".format(e + 1, loss, acc))
        if loss < min_loss:
            min_loss = loss
            min_loss_epoch = e + 1
        if acc > max_acc:
            max_acc = acc
            max_acc_epoch = e + 1
    print("Min loss: {} - Epoch: {}".format(min_loss, min_loss_epoch))
    print("Max acc: {} - Epoch: {}".format(max_acc, max_acc_epoch))

def reset_weights(discrim):
    discrim.get_classifier().mlp.reset_parameters()

def eval_discrim(test_pos_sents, test_neg_sents, discrim):
    print('discriminator predictions for ground truth:')
    for x in test_pos_sents:
        print(x)
        print(discrim.predict(x))

    print('discriminator predictions for random:')
    for x in test_neg_sents:
        print(x)
        print(discrim.predict(x))

def generate_one(query, actual_text):
    print('Evaluating query:', query)

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
    print('{} neighbors ({} before dedup)'.format(len(dedup_neighbors), len(neighbors)))
    print('Neighbor titles:')
    print(cc_data[dedup_neighbors]['title'])

    # create on-the-fly train + test datasets using candidates
    pos_refs = cc_data[dedup_neighbors[:n_pos_refs]]['text']

    neg_refs = cc_data.select([random.randint(0,len(cc_data)-1) for i in range(n_neg_refs)])
    neg_refs = [x['text'] for x in neg_refs]

    test_pos_refs = [actual_text]
    test_neg_refs = cc_data.select([random.randint(0,len(cc_data)-1) for i in range(n_neg_refs)])
    test_neg_refs = [x['text'] for x in test_neg_refs]

    test_pos_sents = get_sent_groups(test_pos_refs)
    test_neg_sents = get_sent_groups(test_neg_refs)

    train_loader, test_loader = get_dataloaders(pos_refs, neg_refs, test_pos_refs, test_neg_refs, batch_size=ref_batch_size)
    reset_weights(discriminator)

    # train on on-the-fly dataset
    train_discrim(discriminator, train_loader, test_loader, lr=ref_lr, epochs=ref_epochs)

    # generate using trained classifier
    if PROMPT_TYPE == TRUE_NEUTRAL_PROMPT:
        prompt = 'BREAKING NEWS: Today'
    else:
        if PROMPT_TYPE == NEUTRAL_PROMPT:
            prefix = 'BREAKING NEWS: Today'
        elif PROMPT_TYPE == HELPFUL_PROMPT:
            prefix = ' '.join(nltk.word_tokenize(actual_text[:n_helpful_prefix_chars])[:n_helpful_prefix_words])
        elif PROMPT_TYPE == ADVERSARIAL_PROMPT:
            prefix = 'Cinderella'
        else:
            print('Invalid prompt type')

        prompt = 'Generate a long article based on its title. Title: ' + query + '. Article Text: ' + prefix

    generation = run_one_pplm_example(
            pretrained_model=pretrained_model,
            cond_text=prompt,
            uncond=False,
            bag_of_words=None,
            discrim=None,
            discrim_weights=None,
            discrim_meta=None,
            class_label=-1,
            length=60,
            stepsize=0.02,
            temperature=1.0,
            top_k=10,
            sample=True,
            num_iterations=1,
            grad_length=10000,
            horizon_length=1,
            window_length=0,
            decay=False,
            gamma=1.5,
            gm_scale=0.9,
            kl_scale=0.01,
            seed=SEED,
            no_cuda=False,
            colorama=False,
            verbosity='quiet',
            tokenizer=None,
            model=None,
            on_the_fly=True, # NEW PARAMS
            classifier=discriminator.get_classifier(), # NEW PARAMS
            class_id=1, # NEW PARAMS
    )

    return generation

################################################################################################
# EVALUATION
################################################################################################

print('Evaluation section! ------------')

# evals modified from COS597/eval.py
mauve = load("mauve")
bleu = load("bleu")

def eval_mauve(generations, labels):
    """
    Return mauve score of generations and references.
    """
    # if len(generations)<200:
    #     logger.warning("To run mauve score, need at least 200 data points.")

    print("calculating mauve scores...")
    mauve_score = mauve.compute(predictions=generations, references=labels)
    print(f"mauve score: {mauve_score.mauve}")
    return mauve_score

def eval_bleu(generations, labels):
    """
    Return bleu score of generations and references.
    """

    # print("calculating bleu scores...")
    bleu_scores = bleu.compute(predictions=[generations], references=[labels])
    print("bleu score: "+str(bleu_scores["bleu"]))

    return bleu_scores["bleu"]

def mean_bleu(y_hat, y):
    print("calculating bleu scores...")
    bleus = np.array([eval_bleu(y_hat_i, y_i) for y_hat_i, y_i in zip(y_hat, y)])
    print('mean: {}, std: {}'.format(np.mean(bleus), np.std(bleus)))
    return np.mean(bleus)

X = [x['title'] for x in cc_data_val]
y = [x['text'] for x in cc_data_val]
y_hat = [generate_one(X_i, y_i) for X_i, y_i in zip(X, y)]

with open('output_' + desc + '.pickle', 'wb') as f:
    pickle.dump([X, y, y_hat], f, protocol=pickle.HIGHEST_PROTOCOL)

mauve_score = eval_mauve(y_hat, y)
blue_score = mean_bleu(y_hat, y)
