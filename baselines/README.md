## Plug and Play Plus (PPP): Better In-Topic Generation With On-The-Fly Classifier
#### Data Preparation  
This project supports two dataset. cc_news contains news articles scrawled from languages (we filter out low-quality data first in this project). To use newsroom dataset, follows the steps below:  
1. Access newsroom dataset through https://lil.nlp.cornell.edu/newsroom/download/index.html  
2. unzip the compressed files and replace the data_dir parameter in data_util.py with your folder path  
### Baselines 
We first run experiments on pre-trained LLMs, including the GPT2, OPT, and Galactica families. Results can be found in baseline1 folder.  
### Evaluation Metrics  
From human evaluation, we observed that existing controlled generation metrics do not align with "in-topicness". Our proposed eval metric takes a BoW approach through a KNN retrieval of most similar embeddings from datastore, and perform a BLEU-like n-gram scanning to calculate generation overlap.  
### Ablations  
The first ablation is with adversarial input. Concretely we measure that, given a random, out-of-topic "generation starter", how well can a model stick with the instructions in the prompt. Results can be found in baseline1 folder under adv_input.
