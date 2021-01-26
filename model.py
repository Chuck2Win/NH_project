# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
class kobert_classifier(nn.Module):
    def __init__(self, kobert):
        super().__init__()
        self.bert = kobert
        self.classifier = nn.Linear(768+3,2)
    def forward(self,input_ids,attention_mask,length, longer, shorter):
        output = self.bert.forward(input_ids = input_ids, attention_mask = attention_mask)
        input = torch.cat([output.pooler_output, length.unsqueeze(-1),longer.unsqueeze(-1) ,shorter.unsqueeze(-1)],-1) # bs, 768
        predict = self.classifier.forward(input)
        return predict
