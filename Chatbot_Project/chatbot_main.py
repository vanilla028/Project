import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from customdataset import ChatbotDataset
from utils import train, my_chatbot
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import re
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

# 배치 데이터 생성 함수
def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

from sklearn.model_selection import train_test_split

chatbot_data  = pd.read_csv("Chatbot_data.csv")

# 데이터를 훈련 및 테스트 데이터셋으로 분할
train_data, test_data = train_test_split(chatbot_data, test_size=0.2, random_state=42)


# 데이터셋과 데이터로더 정의
train_dataset = ChatbotDataset(train_data, max_len=40)
test_dataset = ChatbotDataset(test_data, max_len=40)


#윈도우 환경에서 num_workers 0으로 지정, 리눅스에서는 2
train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=2, shuffle=True, collate_fn=collate_batch,)
test_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=2, shuffle=False, collate_fn=collate_batch,)

model.to(device)
model.train()

criterion = torch.nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5) # 0.00003

epoch = 100
Sneg = -1e18 # 모델 손실 최소화를 위해 설정

train(model, train_dataloader, optimizer, criterion, epoch, Sneg, 'my_chatbot_model.pth')
my_chatbot(model, koGPT2_TOKENIZER)
