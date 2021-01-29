# -*- coding: utf-8 -*-
import pandas as pd
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
from pandas import DataFrame as df
from kobert_transformers import get_tokenizer,get_kobert_model
from model import kobert_classifier
import re

parser = argparse.ArgumentParser(description = '필요한 변수')
# Input data
parser.add_argument('--max_len', default = 64, type = int)
parser.add_argument('--class_1_max_len', default = 512, type = int)
parser.add_argument('--stopword', default = ['재배포 금지','무단배포', '무단전재'], type = list)
parser.add_argument('--batch_size', default = 256, type = int)
parser.add_argument('--input_data', type = str)
parser.add_argument('--model', type = str)
parser.add_argument('--result', type = str)

def inference():
    model.eval()
    Predicted=[]
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, length, longer, shorter, labels = batch
            outputs = model.forward(input_ids, attention_mask, length, longer, shorter)
            predicted = outputs.argmax(-1).tolist()
            Predicted.extend(predicted)
    return Predicted

def remove_stopword(data, stopword_list=args.stopword):
    pattern =''
    for i in stopword_list:
        pattern+='%s|'%i
    return re.sub(pattern[:-1],'',data)

def preprocess(data, stopword_list=args.stopword):
    # 대괄호 안의 문자 제거
    data['content'] = data['content'].apply(lambda i : re.sub(r'\[[^)]*\]', '', i)) 
    # 괄호 안의 문자 제거
    data['content'] = data['content'].apply(lambda i : re.sub(r'\([^)]*\)', '', i)) 
    # 불용어 제거
    data['content'] = data['content'].apply(lambda i : remove_stopword(i))
    # 길이 변수 추가
    data['length'] = data['content'].apply(lambda i : len(tokenizer.encode(i)))
    # BERT에 부합되는 꼴을 만들기 위해, max length = 64로 자르고, 모자란 부분은 패딩
    data['ids'] = data['content'].apply(lambda i : tokenizer.encode(i,add_special_tokens=True,truncation=True,padding='max_length',max_length=args.max_len))
    # attention mask - mask될 부분은 0, 아닌 부분은 1
    data['mask']=(torch.tensor(data['ids'].tolist()).eq(1)==0).long().tolist()
    # longer 길이가 class 1의 최대값보다 큰 경우 1, 아니면 0
    data['longer'] = data['length']>args.class_1_max_len
    # shorter 길이가 class 1의 최대값보다 작거나 같은 경우 1, 아니면 0
    data['shorter'] = data['length']<=args.class_1_max_len
    return data

if __name__ == '__main__':
    args=parser.parse_args()
    # SKT에서 개발한 BERT Tokenizer와 BERT model load 
    tokenizer = get_tokenizer()
    kobert = get_kobert_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = kobert_classifier(kobert).to(device)
    model.load_state_dict(torch.load(args.model))
    # data load
    data = pd.read_csv(args.input_data)
    data = preprocess(data)
    ids = data.n_id
    # tensor dataset으로 묶어주기
    data = TensorDataset(torch.LongTensor(data['ids'].tolist()), torch.LongTensor(data['mask'].tolist()), torch.LongTensor(data['length'].tolist()), torch.LongTensor(data['longer'].tolist()), torch.data(data['shorter'].tolist()), torch.LongTensor(data['info'].tolist()))
    # train loader    
    dataloader = DataLoader(data, batch_size=args.batch_size,drop_last=False)
    Predicted = inference()
    result = df(ids.tolist())
    result['id']=Predicted
    result.to_csv(args.result)