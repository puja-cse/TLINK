import requests
import re
import json
import transformers
import argparse
import sys
import torch
from transformers import pipeline
from huggingface_hub import login
import pandas as pd
import random
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def get_few_shot_prompt(path):
    f = open(path, "r")
    base_prompt=""
    for line in f.readlines(): 
        base_prompt = base_prompt + line
    print(f"Base Prompt: \n{base_prompt}")
    return base_prompt

def get_answer(response):
    answer_list = ["BEFORE", "AFTER", "EQUAL", "VAGUE"]
    for item in reversed(response):
        if item in answer_list:
            return item
    return "BEFORE"

def prompt_model(pipe: pipeline, test_path: str, prompt_path: str): 
    
    pattern = 's1:|s2:|\t|\n'
    base_prompt = get_few_shot_prompt(prompt_path)
    results = []
    true_label = []
    file= open(test_path, "r")
    for line in file.readlines():
        s = re.split(pattern, line)
        s1 = s[1]
        s2 = s[2]
        loc1 = int (s[4])
        e1 = s[5]
        loc2 =int  (s[7])
        e2 = s[8]
        s1 = s1 +' </s> '
        true_label.append(s[9])
        #s1, s2, e1, e2 
        prompt = base_prompt + f""" Sentence 1: {s1}\n Event 1: {e1}\n Sentence 2: {s2}\nEvent 2: {e2}\n
        What is the relation between event 1 and event 2? 
        Generate the answer only. 
        """
        response = pipe(
                prompt,
                temperature=0.7,
                top_k=10,
                max_new_tokens=10,
                do_sample = True
            )[0]['generated_text']
        #print(f"----Generated Text: {response}\n--------\n")
        response = get_answer(re.split(" ", response))
        #print(f"----Response: {response}\n#################\n")
        results.append(response)
    return results, true_label
              
        

def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments"""

    parser = argparse.ArgumentParser(description="Prompt LLMs for TempRel Identification")

    parser.add_argument(
        '--model', 
        type=str, 
        default='google/flan-t5-small',
        help='Model to prompt'
    )

    parser.add_argument(
        '--test_dir',
        type=str,
        default='',
        help='Dir to test dataset'
    )
    parser.add_argument(
        '--prompt_dir',
        type=str,
        default='',
        help='Dir to test dataset'
    )
    parser.add_argument(
        '--n_shot',
        type=str, 
        default='4',
        help='4 shot, 8 shot or 12 shot'
    )
    args = parser.parse_args()
    return args 

def main():
    args = parse_args()
    print(args.model)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if args.model == "meta-llama/Meta-Llama-3-8B-Instruct":
        login(token="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")
        config = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        config.init_device = device
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
        pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    else:
        # Default pipeline setup for other models
        pipe = pipeline('text2text-generation', model=args.model, use_fast=True, torch_dtype=torch.bfloat16, repetition_penalty=1.1)



    root_dir = "../dataset/"
    results, true_label = prompt_model(pipe, root_dir+args.test_dir, root_dir+args.prompt_dir)
    output_dir = '../output_files/'
    df = pd.DataFrame({'predicted_label':results, 'true_label':true_label})
    file_name= output_dir+'LLM_response_'+args.n_shot+'_shot.csv'
    df.to_csv(file_name)
    

if __name__ == '__main__':
    main()