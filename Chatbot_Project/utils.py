import torch

def train(model, train_dataloader, optimizer, criterion, epoch, Sneg, save_path):
    print("Start Training...")
    for current_epoch in range(epoch):
        for batch_idx, samples in enumerate(train_dataloader):
            optimizer.zero_grad()
            token_ids, mask, label = samples
            out = model(token_ids)
            out = out.logits  # Returns a new tensor with the logit of the elements of input
            mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
            mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out))
            loss = criterion(mask_out.transpose(2, 1), label)
            avg_loss = loss.sum() / mask.sum()
            avg_loss.backward()
            optimizer.step()
        
        print(f"Epoch [{current_epoch+1}/{epoch}] completed. Loss: {avg_loss.item()}")

    # 모델 저장
    torch.save(model.state_dict(), save_path)
    print("End Training!")



def my_chatbot(model, tokenizer):
    print("Hello, I'm your Empathetic Chatbot. How can I help you today?")
    print("Type 'quit' to exit.")
    
    while True:
        q = input("User > ").strip()
        if q == "quit":
            break
        a = ""
        
        while True:
            input_ids = torch.LongTensor(tokenizer.encode(Q_TKN + q + SENT + sent + A_TKN + a)).unsqueeze(dim=0)
            with torch.no_grad():
                pred = model(input_ids)
                pred = pred.logits
                gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            
            if gen == EOS:
                break
            a += gen.replace("▁", " ")
        
        print("Chatbot > {}".format(a.strip()))








