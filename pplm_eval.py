# from COS597.data_util import load_data, get_dataloader, prepare_data
import torch
import random
import numpy as np
import pickle
import time
import datetime

from PPLM2.run_pplm import run_one_pplm_example

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

desc = 'simcse_debug_10_cc_news_text_descriptions_10k'
out_desc = 'first_words_simcse_debug_10_cc_news_text_descriptions_10k'

PROMPT_TYPE = NEUTRAL_PROMPT
n_helpful_prefix_chars = 200 # should basically be large enough to cover the chars in n_helpful_prefix_words
n_helpful_prefix_words = 4 # number of words provided in helpful prompt
stepsize = 0.02 # 0.03
gm_scale = 0.5  # 0.9
kl_scale = 0.07 # 0.02

with open('bags_' + desc + '.pickle', 'rb') as f:
    examples, bags, _, _, _ = pickle.load(f)

def generate_one(query, actual_text, custom_bag):
    print('GENERATING...')
    print('Query:', query)
    print('Bag:', custom_bag)

    # generate using BoW
    if PROMPT_TYPE == TRUE_NEUTRAL_PROMPT:
        prompt = 'Today'
    else:
        if PROMPT_TYPE == NEUTRAL_PROMPT:
            prefix = 'Today'
        elif PROMPT_TYPE == HELPFUL_PROMPT:
            prefix = ' '.join(nltk.word_tokenize(actual_text[:n_helpful_prefix_chars])[:n_helpful_prefix_words])
        elif PROMPT_TYPE == ADVERSARIAL_PROMPT:
            prefix = 'Cinderella'
        else:
            print('Invalid prompt type')

        prompt = 'Generate a long article based on its title. Title: ' + query + '. Article Text: ' + prefix

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
    custom_bag = ['google', 'microsoft']
    generation = run_one_pplm_example(
        cond_text=prompt,
        # num_samples=3,
        bag_of_words='religion',
        # discrim='sentiment',
        # class_label='very_positive',
        length=50,
        stepsize=stepsize, #0.02, # 0.03
        sample=True,
        num_iterations=10,
        grad_length=20,
        gamma=1,
        gm_scale=gm_scale, # 0.9
        kl_scale=kl_scale, #0.06, # 0.02
        verbosity='quiet',
        bag_of_words_direct=custom_bag,
    )

    return generation

print('NEW RUN-------------------------------------------------')
print(PROMPT_TYPE)
print(desc)
print(stepsize)
print(gm_scale)
print(kl_scale)

start_time = time.time()
examples = [examples[0]]
bags = [bags[0]]
print(len(examples), 'examples total')

queries = np.array([x['title'] for x in examples])
texts = np.array([x['text'] for x in examples])
y_hat = np.array([generate_one(query, text, bag) for query, text, bag in zip(queries, texts, bags)])

print("--- %s seconds ---" % (time.time() - start_time))

timecode = "{}".format(datetime.datetime.now().strftime("%m_%d_%H_%M_%S"))
output_desc = 'output_BoW_' + out_desc + timecode + '.pickle'
with open(output_desc, 'wb') as f:
    pickle.dump([examples, queries, texts, y_hat], f, protocol=pickle.HIGHEST_PROTOCOL)

print('Saved to', output_desc)