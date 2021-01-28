# -*- coding: utf-8 -*-

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader
from pandas import DataFrame as df
from sklearn.metrics import classification_report
from tqdm import tqdm
import copy
from kobert_transformers import get_kobert_model,get_tokenizer
from model import kobert_classifier
import argparse
parser = argparse.ArgumentParser(description = '필요한 변수')
# Input data
parser.add_argument('--max_len', default = 64)
parser.add_argument('--class_1_max_len', default = 512)
parser.add_argument('--stopword', default = ['재배포 금지','무단배포', '무단전재'])
parser.add_argument('--oversampling', default = True)
parser.add_argument('--train_file', default='./data/train_data_preprocessed')
parser.add_argument('--val_file', default='./data/val_data_preprocessed')
parser.add_argument('--test_file', default='./data/test_data_preprocessed_')
parser.add_argument('--over_train_file', default='./data/train_data_preprocessed_over')
parser.add_argument('--over_val_file', default='./data/val_data_preprocessed_over')
parser.add_argument('--batch_size', default=256)
parser.add_argument('--learning_rate', default=1e-6)
parser.add_argument('--eps', default=1e-8)
parser.add_argument('--weight_decay', default=1e-2)
parser.add_argument('--epochs', default=100)

def train():
    min_value = float('inf')
    min_epoch = None
    min_model = None
    count = 0
    for epoch in tqdm(range(1, args.epochs+1),desc='epoch',mininterval = 300):
        total_loss = 0
        Predicted=[]
        Actual=[]
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, length, longer, shorter, labels = batch
            outputs = model.forward(input_ids, attention_mask, length, longer, shorter)
            loss = F.cross_entropy(outputs, labels)
            predicted = outputs.argmax(-1).tolist()
            Predicted.extend(predicted)
            Actual.extend(labels.tolist())
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
                 
        
        if epoch%5 == 0: # 5 epoch마다 train, validation 계산
            # train
            with torch.no_grad():
                model.eval()
                total_loss = 0
                Predicted=[]
                Actual=[]
                for batch in train_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, attention_mask, length, longer, shorter, labels = batch
                    outputs = model.forward(input_ids, attention_mask, length, longer, shorter)
                    loss = F.cross_entropy(outputs, labels)
                    predicted = outputs.argmax(-1).tolist()
                    Predicted.extend(predicted)
                    Actual.extend(labels.tolist())
                    total_loss += loss.item()
                
                curr_loss = total_loss / len(train_dataloader)            
                print("")
                print("  Average train loss: {0:.5f}".format(curr_loss))
                print(classification_report(Actual,Predicted,digits=4))
                print("")
                
            # val    
            with torch.no_grad():
                model.eval()
                total_loss = 0
                Predicted=[]
                Actual=[]
                for batch in val_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, attention_mask, length, longer, shorter, labels = batch
                    outputs = model.forward(input_ids, attention_mask, length, longer, shorter)
                    loss = F.cross_entropy(outputs, labels)
                    predicted = outputs.argmax(-1).tolist()
                    Predicted.extend(predicted)
                    Actual.extend(labels.tolist())
                    total_loss += loss.item()
                
                curr_loss = total_loss / len(val_dataloader)            
                print("")
                print("  Average val loss: {0:.5f}".format(curr_loss))
                print(classification_report(Actual,Predicted,digits=4))
                print("")
                
                if min_epoch is None:
                    min_epoch = copy.copy(epoch)
                    min_value = copy.copy(curr_loss)
                    min_model = copy.deepcopy(model.state_dict())
                else:
                    if min_value < curr_loss:
                        count+=1
                        if count == 2: # epoch 10동안 해당 모델의 validation 값이 커지면 중단.
                            torch.save(min_model, './at_%d_model'%(min_epoch))
                            break
                    else:
                        count = 0 
                        min_value = curr_loss
                        min_epoch = copy.copy(epoch)
                        min_model = copy.deepcopy(model.state_dict())
    torch.save(min_model,'./at_%d_model'%(min_epoch))
    
    


if __name__ == '__main__':
    args=parser.parse_args()
    # SKT에서 개발한 BERT Tokenizer와 BERT model load 
    tokenizer = get_tokenizer()
    kobert = get_kobert_model()
    
    # data load
    train_data = pd.read_pickle(args.over_train_file if args.oversampling else args.train_file)
    val_data = pd.read_pickle(args.over_val_file if args.oversampling else args.val_file)
    test_data = pd.read_pickle(args.test_file)
    # tensor dataset으로 묶어주기
    train_data = TensorDataset(torch.LongTensor(train_data['ids'].tolist()), torch.LongTensor(train_data['mask'].tolist()), torch.LongTensor(train_data['length'].tolist()), torch.LongTensor(train_data['longer'].tolist()), torch.LongTensor(train_data['shorter'].tolist()), torch.LongTensor(train_data['info'].tolist()))
    val_data = TensorDataset(torch.LongTensor(val_data['ids'].tolist()), torch.LongTensor(val_data['mask'].tolist()), torch.LongTensor(val_data['length'].tolist()), torch.LongTensor(val_data['longer'].tolist()), torch.LongTensor(val_data['shorter'].tolist()), torch.LongTensor(val_data['info'].tolist()))
    test_data = TensorDataset(torch.LongTensor(test_data['ids'].tolist()), torch.LongTensor(test_data['mask'].tolist()), torch.LongTensor(test_data['length'].tolist()), torch.LongTensor(test_data['longer'].tolist()), torch.LongTensor(test_data['shorter'].tolist()), torch.LongTensor(test_data['info'].tolist()))

    # train loader    
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size,drop_last=False)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size,drop_last=False)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size,drop_last=False)    
    
    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model
    model = kobert_classifier(kobert).to(device)
    optimizer = AdamW(model.parameters(), lr = args.learning_rate, eps = args.eps, weight_decay = args.weight_decay)
    epochs = args.epochs
    train()   
