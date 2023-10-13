# pip install transformers
# pip install pytorch_lightning

import math
import numpy as np
import pandas as pd
import random
import re
import torch
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from pytorch_lightning import Trainer, LightningModule

# 시드 설정
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True  # GPU 사용 시 재현성을 위해 추가

data = pd.read_csv('Chatbot_Data.csv')
data

BOS = '</s>'
EOS = '</s>'
PAD = '<pad>'
UNK = '<unk>'
MASK = '<mask>'
SENT = '<unused0>' # 질문/답변 구분
Q_TKN = '<usr>' # 질문
A_TKN = '<sys>' # 답변

tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2",
        bos_token=BOS, eos_token=EOS, unk_token=UNK, pad_token=PAD, mask_token=MASK,)

tokenizer.tokenize("안녕하세요. 공감에 특화된 챗봇을 만드는 중입니다.")

class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=40):
        self._data = chats
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.sent_token = SENT
        self.eos = EOS
        self.mask = MASK
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        q = turn["Q"]  # 질문을 가져온다.
        q = re.sub(r"([?.!,])", r" ", q)  # 질문의 구두점 제거

        a = turn["A"]  # 답변을 가져온다.
        a = re.sub(r"([?.!,])", r" ", a)  # 답변의 구두점 제거

        q_tokenized = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        q_len = len(q_tokenized)

        a_tokenized = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_tokenized)

        # 질문 길이가 max_len보다 긴 경우
        if q_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_tokenized = q_tokenized[-(int(self.max_len / 2)) :]   # 질문 길이 = max_len의 반으로
                q_len = len(q_tokenized)
                a_len = self.max_len - q_len              # 답변 길이 = max_len - 질문 길이
            a_tokenized = a_tokenized[:a_len]
            a_len = len(a_tokenized)

        # (질문 길이 + 답변 길이)가 max_len보다 크면
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len        #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:       #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_tokenized = q_tokenized[-(int(self.max_len / 2)) :]   #질문길이를 최대길이의 반으로
                q_len = len(q_tokenized)
                a_len = self.max_len - q_len              #답변의 길이를 최대길이 - 질문길이
            a_tokenized = a_tokenized[:a_len]
            a_len = len(a_tokenized)

        # 답변 labels = [mask, mask, ...., mask, ..., <bos>,..답변.. <eos>, <pad>....]
        labels = [self.mask,] * q_len + a_tokenized[1:]

        # mask = 질문길이 0 + 답변길이 1 + 나머지 0
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)

        # 답변 labels를 index로 만든다.
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        # 최대길이만큼 PADDING
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        # (질문 + 답변)을 index로 만든다.
        token_ids = self.tokenizer.convert_tokens_to_ids(q_tokenized + a_tokenized)
        # 최대길이만큼 PADDING
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        # (질문 + 답변), 마스크, 답변
        return (token_ids, np.array(mask), labels_ids)


def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)


def train(model, train_dataloader, num_epoch, device):
    model.to(device)
    model.train()

    # 파라미터 설정
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)  # 0.00003
    Sneg = -1e18 # 모델 손실 최소화를 위해 설정

    print("Training start...")
    for epoch in range(num_epoch):
        total_loss = 0.0  # 각 에포크의 총 손실을 추적하기 위한 변수
        for batch_idx, samples in enumerate(train_dataloader):
            optimizer.zero_grad()
            token_ids, mask, label = samples
            token_ids = token_ids.to(device)
            mask = mask.to(device)
            label = label.to(device)
            out = model(token_ids)
            out = out.logits
            mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
            mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out))
            loss = criterion(mask_out.transpose(2, 1), label)
            avg_loss = loss.sum() / mask.sum()
            avg_loss.backward()
            optimizer.step()

            total_loss += avg_loss.item()

        print(f"Epoch [{epoch + 1}/{num_epoch}] - Average Loss: {total_loss / len(train_dataloader)}")

    print("Training end!")

    # 학습이 완료된 모델을 저장
    model_save_path = "./model(50epoch).pth"
    torch.save(model.state_dict(), model_save_path)

# 모델 생성 및 데이터 로드
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
train_dataset = ChatbotDataset(data, max_len=40)

# Window 환경에서 num_workers=0으로 지정, Linux에서는 2로 지정
train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=2, shuffle=True, collate_fn=collate_batch)

# 학습 실행
num_epoch = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train(model, train_dataloader, num_epoch, device)

print("Model saved to './model(50epoch).pth'")

"""### 학습 완료된 모델로 챗봇 테스트하기"""

def my_chatbot(model, tokenizer, device):
    print("안녕하세요, 저는 공감 특화 챗봇입니다. 대화를 시작합니다.")
    print("'q'를 입력하면 챗봇이 종료됩니다.")

    while True:
        q = input("User > ").strip()
        if q == "q":
            break
        a = ""
        while True:
            input_ids = torch.LongTensor(tokenizer.encode(Q_TKN + q + SENT + A_TKN + a)).unsqueeze(dim=0).to(device)
            with torch.no_grad():
                pred = model(input_ids)
                pred = pred.logits
                gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred.cpu(), dim=-1).squeeze().numpy().tolist())[-1]
                if gen == EOS:
                    break
                a += gen.replace("▁", " ")
        print("Chatbot > {}".format(a.strip()))

# 학습 완료된 챗봇 테스트

my_chatbot(model, tokenizer, device)

