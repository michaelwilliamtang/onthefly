import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from data_util import get_dataloader

def load_model(logger, model_name):
    """
    Load transformers model, return tokenizer and model. 
    """
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-2}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    logger.info(f"Loading {model_name} with {free_in_GB-2} GB memory and {n_gpus} GPU.")

    if "opt" in model_name:
        assert model_name in ["opt-350m", "opt-1.3b", "opt-2.7b", "opt-6.7b","opt-13b", "opt-30b", "opt-66b"]
        model_name = "facebook/"+model_name
    elif "galactica" in model_name:
        assert model_name in ["galactica-125m","galactica-1.3b","galactica-6.7b",'galactica-30b']
        model_name = "facebook/"+model_name
    elif "gpt2" in model_name:
        assert model_name in ["gpt2", "gpt2-medium","gpt2-large", "gpt2-xl"]
    
    else:
        raise NotImplementedError()

    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        max_memory=max_memory,
        offload_folder = "~/.cache/huggingface/.offload"
        )
    if tokenizer.eos_token is None:
        logger.info("TOKENIZER: updated eos token")
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    if tokenizer.bos_token is None:
        logger.info("TOKENIZER: updated bos token")
        tokenizer.add_special_tokens({'bos_token': '<|endoftext|>'})
    # if tokenizer.cls_token is None:
    #     logger.info("TOKENIZER: updated cls token")
    #     tokenizer.add_special_tokens({'cls_token': '<|CLS|>'})
    # if tokenizer.mask_token is None:
    #     logger.info("TOKENIZER: updated mask token")
    #     tokenizer.add_special_tokens({'mask_token': '<|MASK|>'})
    # if tokenizer.sep_token is None:
    #     logger.info("TOKENIZER: updated sep token")
    #     tokenizer.add_special_tokens({'sep_token': '<|SEP|>'})

    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    return model, tokenizer
        

def generate(logger, 
            model, tokenizer, 
            input_tensors, batch_size = 64,
            method="top-p",
            min_len = 0, 
            max_len = 0,
            ):
    """
    Returns List[str] of model generation.
        input_tensors: Dict[str, List] from prepared_data (not dataloader).
    
    Please use top-p decoding. The other ones are really bad.
    """
    assert method in ["beam", "greedy", "top-p"]
    temperature = 1.0
    lp = 0.9
    
    
    dataloader = get_dataloader(input_tensors,batch_size=batch_size)
    generations = []
    
    logger.info("Running model generation...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch[0].cuda()
            attn_mask = batch[1].cuda()

            max_gen_len = 600
            min_gen_len = 128

            # logger.info(f"max gen len: {max_gen_len}")
            # logger.info(f"min gen len: {min_gen_len}")
            
            if method == "beam":
                outputs = model.generate(input_ids, attention_mask=attn_mask,
                                        max_length = max_gen_len,
                                        min_length = min_gen_len,
                                        temperature = temperature,
                                        num_beams = 5,
                                        length_penalty = lp,
                                        eos_token_id = tokenizer.eos_token_id,
                                        bos_token_id = tokenizer.bos_token_id,
                                        )
                generations += [_ for _ in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
            
            elif method == "greedy":
                outputs = model.generate(input_ids, attention_mask=attn_mask,
                                        max_length = len(input_ids)+max_len,
                                        min_length = len(input_ids)+min_len,
                                        temperature = temperature,
                                        do_sample = False,
                                        length_penalty = lp,
                                        eos_token_id = tokenizer.eos_token_id,
                                        bos_token_id = tokenizer.bos_token_id,
                                        )
                generations += [_ for _ in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
                

            elif method == "top-p":
                outputs = model.generate(input_ids, attention_mask=attn_mask,
                                        max_length = max_gen_len,
                                        min_length = min_gen_len,
                                        temperature = temperature,
                                        do_sample = True,
                                        top_k = 0,
                                        top_p = 0.9,
                                        repetition_penalty = 0.9,
                                        length_penalty = lp,
                                        eos_token_id = tokenizer.eos_token_id,
                                        bos_token_id = tokenizer.bos_token_id,
                                        )
                generations += [_ for _ in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
    
    # logger.info(f"SHAPE OF GENERATIONS: {len(generations)}")
    return generations 