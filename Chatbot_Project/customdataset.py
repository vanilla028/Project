import math
import numpy as np
import pandas as pd
import random
import re
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

chatbot_data  = pd.read_cvs("Chatbot_data.csv")

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

# 허깅페이스 transformers 에 등록된 사전 학습된 koGTP2 토크나이저를 가져온다.
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", 
    bos_token=BOS, eos_token=EOS, unk_token="<unk>", pad_token=PAD, mask_token=MASK,)

class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=40):
        self._data = chats
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.sent_token = SENT
        self.eos = EOS
        self.mask = MASK
        self.tokenizer = koGPT2_TOKENIZER

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        q = turn["Q"]  # 질문 가져오기
        q = re.sub(r"([?.!,])", r" ", q)  # 구두점들을 제거

        a = turn["A"]  # 답변 가져오기
        a = re.sub(r"([?.!,])", r" ", a)  # 구두점들을 제거

        q_tokens = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        q_len = len(q_tokens)

        a_tokens = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_tokens)

        # 질문의 길이가 최대길이보다 크면
        if q_len > self.max_len:
            a_len = self.max_len - q_len        # 답변의 최대길이 - 질문길이
            if a_len <= 0:
                q_tokens = q_tokens[-(int(self.max_len / 2)) :]   # 질문 길이: 최대 길이의 반으로 
                q_len = len(q_tokens)
                a_len = self.max_len - q_len              # 답변의 길이: 최대길이 - 질문길이
            a_tokens = a_tokens[:a_len]
            a_len = len(a_tokens)

        #질문의 길이 + 답변의 길이가 최대길이보다 크면
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len        # 답변의 길이: 최대길이 - 질문길이
            if a_len <= 0:
                q_tokens = q_tokens[-(int(self.max_len / 2)) :]   # 질문길이: 최대길이의 반으로 
                q_len = len(q_tokens)
                a_len = self.max_len - q_len              # 답변의 길이: 최대길이 - 질문길이
            a_tokens = a_tokens[:a_len]
            a_len = len(a_tokens)

        # 답변 labels = [mask, mask, ...., mask, ..., <bos>,..답변.. <eos>, <pad>....]
        labels = [self.mask,] * q_len + a_tokens[1:]

        # mask = 질문길이 0 + 답변길이 1 + 나머지 0
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        
        # 답변 labels을 index 로 만든다.
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        
        # 최대길이만큼 PADDING
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        # 질문 + 답변을 index 로 만든다.    
        token_ids = self.tokenizer.convert_tokens_to_ids(q_tokens + a_tokens)
        
        # 최대길이만큼 PADDING
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        #질문+답변, 마스크, 답변
        return (token_ids, np.array(mask), labels_ids)

