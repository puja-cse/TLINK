import torch
import torch.nn as nn
from customDataHandler import customDataset
from torch.utils.data import DataLoader
import argparse
from src.BertTempRel import BertTempRel
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np 

parser = argparse.ArgumentParser(description='PyTorch TLink Training')
parser.add_argument('--model_path', default="", type=str, help='Path to model')
parser.add_argument('--data_path', default='formatted_dataset/it_event_pairs_with_tab.txt', type=str, help='data path')
parser.add_argument('--path_to_write', default='Output_with_relation/sample.txt', type=str, help='where to write the relations')
parser.add_argument('--batch', default=16, type=int, help='Batch size')

args = parser.parse_args()

def get_labels(model_path, data_path, write_to_file, batch_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = "..'/dataset/"

    label2idx = {'BEFORE': 0, 'AFTER': 1, 'EQUAL':2, 'VAGUE': 3}
    idx2label = {0: 'BEFORE', 1: 'AFTER', 2: 'EQUAL', 3: 'VAGUE'}

    dataset= customDataset(path=root+data_path)
    custom_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    y_pred_all = np.empty(0)

    model = BertTempRel(labels=4)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch in custom_dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            loc1 = batch['location1'].to(device)
            loc2 = batch['location2'].to(device)
            y_logits = model(input_ids, attention_mask, loc1, loc2)
            #loss = loss_fn(y_logits, y)
            y_pred = y_logits.argmax(dim=1).detach().cpu().numpy()
            y_pred_all = np.append(y_pred_all, y_pred)
    input_file = open(root+data_path, "r")
    output_file = open(root+write_to_file, "a")
    i=0
    for line in input_file.readlines():
        output_file.write(line+"\t"+idx2label[y_pred_all[i]]+"\n")
    input_file.close()
    output_file.close()


if __name__ == "__main__":
    model_path = args.model_path
    data_path = args.data_path 
    write_to_file = args.path_to_write
    batch_size = args.batch
    get_labels(model_path, data_path, write_to_file, batch_size)
    
    
    