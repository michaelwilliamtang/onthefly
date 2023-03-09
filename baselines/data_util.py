import datasets 
import logging
from datasets import load_dataset, Dataset
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import numpy as np
import pandas as pd 
import os 
from tqdm import tqdm 
import torch
import pickle 



def load_data(logger, name, split="validation", data_dir = "./data"):
    """
    Load data from newsroom dataset.
    
    Return
        data with title, summary, and text 
    """
    logger.info(f"Loading dataset...")
    if name == "newsroom":
        data = load_dataset(name, split=split, data_dir = data_dir).remove_columns(
            ['url', 'date', 'density_bin', 'coverage_bin', 'compression_bin', 'density', 'coverage', 'compression']
        )
        
    elif name == "cc_news":
        # cc-news only has train set 
        if split != "train":
            logger.warning(f"Asked for {split} split, but cc-news only has train split.")
        data = load_dataset("cc_news",split="train").remove_columns(
            ["date","domain","image_url"]
        ).rename_column("description","summary")
        
        data = data.filter()
        
        
    
    logger.info(f"loaded {len(data)} data from {name}.")
    return data


def get_dataloader(inputs, batch_size=64):
    """
    Return dataloader. for each batch, input_ids = batch[0] and attn_mask = batch[1]
        inputs: Dict[str, List] from prepared_data()
    """
    shape = inputs["input_ids"].shape 
    for v in inputs.values():
        assert v.shape == shape 
    
    dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
    # dataset = TensorDataset(inputs["input_ids"])
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader
    


def prepare_data(logger,tokenizer,data_list):
    """
    prepare data in tensor form.
        data_list: Dataset, keys {title, summary, text}
        title_prefix, content_prefix, summary_prefix: str 
    
    Prepare each data in the form title_prefix, title, (summary_prefix, summary), content_prefix.
    """

    logger.info("preparing data tensors...")
    
    """
    TODO @mwtang come up with better prompts please :)
    """
    prompts = []
    for data in tqdm(data_list):
        # fetch first sentence, may be fuzzy 
        summary_chunck = data["summary"].split(sep="\n")[0].split(sep=". ")[0]

        # need to truncate for long summaries.
        summary_chunck = summary_chunck[:200]+". "
        prefix = "Generate long article based on title and summary. Title: "
        prompt = prefix+data["title"]+" Summary: "+ summary_chunck+"Generation:"
        
        if len(prompt) > 1020:
            logger.warning("input sequence too long. disgarded data instance.")
        else:
            prompts.append(prompt)
    
    logger.info("Running tokenization...")
    input_tensors = tokenizer(prompts, return_tensors="pt", padding=True)

    return input_tensors

def prepare_adversarial_data(logger,tokenizer,data_list):
    """
    prepare data in tensor form, with random starter appended.
    Ablation goal: understand how much it can focus in the "guidance" rather than 
    following the random starter.
    """
    random_starters = ["The potato wakes up", "Cinderella wakes up", "The chincilla wakes up"]
    prompts = []
    for data in tqdm(data_list):
        summary_chunck = data["summary"].split(sep="\n")[0].split(sep=". ")[0]
        prefix = "Generate long article based on title and summary. Title: "
        prompt = prefix+data["title"]+" Summary: "+ summary_chunck+". Generation: "
        
        # random starters 
        starter = random_starters[np.random.randint(0,3)]
        prompt = prompt + starter 
        prompts.append(prompt)

    logger.info("Running tokenization...")
    input_tensors = tokenizer(prompts, return_tensors="pt", padding=True)


    return input_tensors

def prepare_in_context_starter_data(logger, tokenizer, data_list):
    """
    prepare data in tensor form, with first sentence of the actual news appended. 
    """
    prompts = []
    for data in tqdm(data_list):
        summary_chunck = data["summary"].split(sep="\n")[0].split(sep=". ")[0]
        starter_chunck = data["text"].split(sep="\n")[0].split(sep=". ")[0]
        prefix = "Generate long article based on title and summary. Title: "
        prompt = prefix+data["title"]+" Summary: "+ summary_chunck+". Generation: "+starter_chunck
        if len(prompt) > 1020:
            logger.warning("input sequence too long. disgarded data instance.")
        else:
            prompts.append(prompt)
    logger.info(len(prompts))
    logger.info("Running tokenization...")
    input_tensors = tokenizer(prompts, return_tensors="pt", padding=True)
    return input_tensors 