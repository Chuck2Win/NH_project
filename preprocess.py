# -*- coding: utf-8 -*-

import re
import torch
import pandas as pd
from kobert_transformers import get_tokenizer
tokenizer = get_tokenizer()

# 데이터 읽어오기
train_data=pd.read_csv('./train_data.csv',header=0)
test_data=pd.read_csv('./test_data.csv',header=0)

# stop word 제거 (현재는 재배포 금지, 무단배포, 무단전재만을 넣어둠)
def remove_stopword(data, stopword_list=['재배포 금지','무단배포', '무단전재']):
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
train_data['ids'] = train_data['content'].apply(lambda i : tokenizer.encode(i,add_special_tokens=True,truncation=True,padding='max_length',max_length=64))
test_data['ids'] = test_data['content'].apply(lambda i : tokenizer.encode(i,add_special_tokens=True,truncation=True,padding='max_length',max_length=64))

# attention mask - mask될 부분은 0, 아닌 부분은 1
attention_masks_train=(torch.tensor(train_data['ids'].tolist()).eq(1)==0).long()
attention_masks_test=(torch.tensor(test_data['ids'].tolist()).eq(1)==0).long()

train_data['mask']=(torch.tensor(train_data['ids'].tolist()).eq(1)==0).long().tolist()
test_data['mask']=(torch.tensor(test_data['ids'].tolist()).eq(1)==0).long().tolist()

## 저장하기(pickle 형태로)
train_data.to_pickle('./train_data_preprocessed')
test_data.to_pickle('./test_data_preprocessed')
