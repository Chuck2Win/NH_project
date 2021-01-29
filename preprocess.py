# -*- coding: utf-8 -*-

import re
import torch
import pandas as pd
from pandas import DataFrame as df
from kobert_transformers import get_tokenizer
import argparse
from sklearn.model_selection import train_test_split
from imblearn.over_sampling  import RandomOverSampler
parser = argparse.ArgumentParser(description = '필요한 변수')
# Input data
parser.add_argument('--train_file', default='./data/train_data.zip', type = str)
parser.add_argument('--val_size', default = 0.05, type = float)
parser.add_argument('--test_size', default = 0.2, type = float)                  
parser.add_argument('--test_file', default='./data/test_data.zip', type = str)
parser.add_argument('--max_len', default = 64, type = int)
parser.add_argument('--class_1_max_len', default = 512, type = int)
parser.add_argument('--stopword', default = ['재배포 금지','무단배포', '무단전재'], type = list)
parser.add_argument('--oversampling', default = True, type = bool)
if __name__ == '__main__':
    # save  
    args = parser.parse_args()
    tokenizer = get_tokenizer()

    # 데이터 읽어오기
    train_data=pd.read_csv(args.train_file,header=0)
    test_data=pd.read_csv(args.test_file,header=0)

    # stop word 제거 (현재는 재배포 금지, 무단배포, 무단전재만을 넣어둠)
    def remove_stopword(data, stopword_list=args.stopword):
        pattern =''
        for i in stopword_list:
            pattern+='%s|'%i
        return re.sub(pattern[:-1],'',data)

    # 대괄호 안의 문자 제거
    train_data['content'] = train_data['content'].apply(lambda i : re.sub(r'\[[^)]*\]', '', i)) 
    # 괄호 안의 문자 제거
    train_data['content'] = train_data['content'].apply(lambda i : re.sub(r'\([^)]*\)', '', i)) 
    # stop word 제거
    train_data['content'] = train_data['content'].apply(lambda i : remove_stopword(i))

    # 문장 길이 변수 추가
    test_data['length'] = test_data['content'].apply(lambda i : len(tokenizer.encode(i)))
    train_data['length'] = train_data['content'].apply(lambda i : len(tokenizer.encode(i)))

    # BERT에 부합되는 꼴을 만들기 위해, max length = 64로 자르고, 모자란 부분은 패딩
    train_data['ids'] = train_data['content'].apply(lambda i : tokenizer.encode(i,add_special_tokens=True,truncation=True,padding='max_length',max_length=args.max_len))
    test_data['ids'] = test_data['content'].apply(lambda i : tokenizer.encode(i,add_special_tokens=True,truncation=True,padding='max_length',max_length=args.max_len))

    # attention mask - mask될 부분은 0, 아닌 부분은 1
    train_data['mask']=(torch.tensor(train_data['ids'].tolist()).eq(1)==0).long().tolist()
    test_data['mask']=(torch.tensor(test_data['ids'].tolist()).eq(1)==0).long().tolist()
    
    # longer 길이가 class 1의 최대값보다 큰 경우 1, 아니면 0
    train_data['longer'] = train_data['length']>args.class_1_max_len
    train_data['longer'] = train_data['longer'].astype(int)
    
    # shorter 길이가 class 1의 최대값보다 작거나 같은 경우 1, 아니면 0
    train_data['shorter'] = train_data['length']<=args.class_1_max_len
    train_data['shorter'] = train_data['shorter'].astype(int)
    
    # train test split
    train_data,val_data = train_test_split(train_data,test_size = args.val_size,shuffle = True)
    val_data, test_data_ = train_test_split(val_data, test_size = args.test_size,shuffle = True)
    
    # oversampling
    if args.oversampling:
        train_X = train_data.loc[:,['ids','mask','length','longer','shorter']]
        train_y = train_data['info']
        val_X = val_data.loc[:,['ids','mask','length','longer','shorter']]
        val_y = val_data['info']
        X_samp, y_samp = RandomOverSampler().fit_sample(train_X.to_numpy(), train_y) # ids, mask, ord, length 순
        Val_X_samp,Val_y_samp = RandomOverSampler().fit_sample(val_X.to_numpy(), val_y) # ids, mask, length, longer, shorter 순
        sampled_train_y = df(y_samp.tolist(),columns=['info'])
        sampled_val_y = df(Val_y_samp.tolist(),columns=['info'])
        sampled_train = df(X_samp.tolist(),columns=train_X.columns)
        sampled_val = df(Val_X_samp.tolist(),columns=train_X.columns)
        Sampled_train = pd.concat([sampled_train,sampled_train_y],axis=1)
        Sampled_val = pd.concat([sampled_val,sampled_val_y],axis=1)
        Sampled_train.to_pickle('./data/train_data_preprocessed_over')
        Sampled_val.to_pickle('./data/val_data_preprocessed_over')
    else:
        ## 저장하기(pickle 형태로)
        train_data.to_pickle('./data/train_data_preprocessed')
        val_data.to_pickle('./data/val_data_preprocessed')
    
    
    test_data_.to_pickle('./data/test_data_preprocessed_')
    test_data.to_pickle('./data/test_data_preprocessed')
