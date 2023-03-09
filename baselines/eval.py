#TODO: do not run, not debugged.
print("launched eval pipeline.")
import os
import argparse
import numpy as np
from evaluate import load
import logging
from tqdm import tqdm
from simCSE.simcse.tool import SimCSE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data_pair(logger, data_path, gen_only = True ):
    """
    Load generation results, convert reference to prompt format. 
        data_path should contain res.npy and ref.npy files.
    
    TODO: if prompting method changed, this also need to be chagned.
    """
    logger.info(f"loading data from {data_path}")
    assert os.path.exists(data_path+"-res.npy")
    raw_generations = np.load(data_path+"-res.npy")
    raw_references = np.load(data_path+"-ref.npy",allow_pickle=True)

    assert len(raw_generations) == len(raw_references)
    
    labels=[]
    generations = []
    if gen_only:
        for gen in raw_generations:
            generations.append(gen.split(sep="Generation:")[1])
        
        for lab in raw_references:
            labels.append(lab["text"])

    else: 
        for data in raw_references:
            if len(data["summary"]) > 1:
                summary_chunck = data["summary"].split(sep="\n")[0].split(sep=". ")[0]+". "
                prefix = "Generate long article based on title and summary. Title: "
                prompt = prefix+data["title"]+" Summary: "+ summary_chunck+"Generation: "
            
            else: 
                prefix = "Generate a long article based on title. Title: "
                prompt = prefix+data["title"]+" Generation: "
            
            label = prompt+data["text"]
            labels.append(label)
        generations = raw_generations 
    
    
    assert(len(generations) == len(labels))
    return generations, labels


def calc_stats(logger, generations, labels):
    """
    Calculate the average number of words and sentences.
        generations in the format produced by run_baseline 
        labels in the format of prompt+text 
    """
    logger.info("calculating generation stats...")
    # word level 
    sum_ref = 0
    sum_res = 0
    for i in range(len(generations)):
        sum_ref += len(labels[i].split(" "))
        sum_res += len(generations[i].split(" "))
    logger.info(f"generation has avg. word {sum_res/len(generations)}")
    logger.info(f"reference has avg. word {sum_ref/len(labels)}")

    # word level 
    sum_ref = 0
    sum_res = 0
    for i in range(len(generations)):
        sum_ref += len(labels[i].split(". "))
        sum_res += len(generations[i].split(". "))
    logger.info(f"generation has avg. sent {sum_res/len(generations)}")
    logger.info(f"reference has avg. sent {sum_ref/len(labels)}")

    return 
    
def eval_tfidf(logger, generations, labels):
    """
    Return tf-idf score.
    """
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    vectorizer = tfidf_vectorizer.fit(labels)
    logger.info("calculating tf-idf score...")
    scores = []
    for gen,ref in tqdm(zip(generations, labels)):
        score = cosine_similarity(vectorizer.transform([gen]), vectorizer.transform([ref]))
        scores.append(score[0])
    tfidf_score = np.average(scores)
    logger.info("tf-idf score: "+str(tfidf_score))
    return tfidf_score

def eval_mauve(logger,generations, labels):
    """
    Return mauve score of generations and references.
    """
    mauve = load("mauve")
    if len(generations)<200: 
        logger.warning("To run mauve score, need at least 200 data points.")

    logger.info("calculating mauve scores...")
    mauve_score = mauve.compute(predictions=generations, references=labels)
    logger.info(f"mauve score: {mauve_score.mauve}")
    return mauve_score    

def eval_bleu(logger,generations, labels):
    """
    Return bleu score of generations and references.
    """
    bleu = load("bleu")
    # scores = []

    logger.info("calculating bleu scores...")
    try:
        score = bleu.compute(predictions=generations, references=labels)
    except ZeroDivisionError:
        score = 0
    #     scores.append(score)
    # bleu_scores = np.average(scores)
    logger.info("bleu score: "+str(score))
    return score 

def eval_simcse(logger, generations, labels):
    simcse = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    scores = []
    logger.info("calculating simCSE scores...")
    for gen,ref in tqdm(zip(generations, labels)):
        scores.append(simcse.similarity([gen],[ref]))
    
    sim_score = np.average(scores)
    return sim_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str, default="gpt2-large")
    parser.add_argument("--res_dir",type=str, default="baseline1/cc_news-output/")
    parser.add_argument("--bleu",default=False, action="store_true")
    parser.add_argument("--mauve",default=False,action="store_true")
    parser.add_argument("--simcse", default=False, action="store_true")
    parser.add_argument("--tfidf",default=False,action="store_true")

    parser.add_argument("--log_folder", type=str, default="eval_log/")
    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    
    # path = args.model + "-top-p"
    # path = args.res_dir+path
    MODEL_NAME = args.model
    if MODEL_NAME in ["galactica-1.3b","galactica-6.7b"]:
        NORMAL_PATH = f"baseline1/cc_news-output/{MODEL_NAME}-top-p"
    else: 
        NORMAL_PATH = f"baseline1/cc_news-output/var_len/{MODEL_NAME}-300-FINAL"
        ADV_PATH = f"baseline1/cc_news-output/var_len/{MODEL_NAME}-300-adv"
        IN_CONTEXT_PATH = f"baseline1/cc_news-output/var_len/{MODEL_NAME}-300-INCONTEXT"

    generations, labels = load_data_pair(logger,data_path=NORMAL_PATH)
    calc_stats(logger, generations, labels)

    res_dict = {}

    if not isinstance(generations, list):
        generations = [generations]
        labels = [labels]

    if args.bleu:
        res = eval_bleu(logger, generations, labels)
        res_dict["bleu"] = res 

    if args.mauve:
        res = eval_mauve(logger, generations, labels)
        res_dict["mauve"] = res 
    
    if args.simcse: 
        res = eval_simcse(logger, generations, labels)
        res_dict["simcse"] = res
    
    if args.tfidf: 
        res = eval_tfidf(logger, generations, labels)
        res_dict["tfidf"] = res
    
    print(res_dict)
    

