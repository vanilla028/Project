import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
model.load_state_dict(torch.load('model(50epoch).pth'))

model.to(device)

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


def my_chatbot(model, tokenizer, device):
    print("안녕하세요, 저는 공감 특화 챗봇입니다. 대화를 시작합니다.")
    print("'q'를 입력하면 챗봇이 종료됩니다.")

    while True:
        q = input("User > ").strip()
        if q.lower() == "q":
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

if __name__ == "__main__":
    my_chatbot(model, tokenizer, device)
