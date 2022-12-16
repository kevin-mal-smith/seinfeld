from transformers import GPT2Tokenizer,GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import tensorflow as tf
import pandas as pd
from tqdm import tqdm, trange
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import os


scripts = pd.read_csv('seinfeld_data/scripts.csv')
scripts['line'] = scripts.Character + ': ' + scripts.Dialogue

class seinfeld_lines(Dataset):  
    def __init__(self, control_code, truncate=False, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.lines = []

        for row in scripts['line']:
          self.lines.append(torch.tensor(
                self.tokenizer.encode(f"<|{control_code}|>{row}<|endoftext|>")
            ))               
        if truncate:
            self.lines = self.lines[:20000]
        self.lines_count = len(self.lines)
        
    def __len__(self):
        return self.lines_count

    def __getitem__(self, item):
        return self.lines[item]
dataset = seinfeld_lines(scripts['line'], truncate=True, gpt2_type="gpt2")   
scripts.line = scripts.line.astype(str)

#Get the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2',padding_side='left')
model = GPT2LMHeadModel.from_pretrained('gpt2')

#Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None


def train(
    dataset, model, tokenizer,
    batch_size=16, epochs=1, lr=2e-5,
    max_seq_len=400, warmup_steps=200,
    gpt2_type="gpt2", output_dir=".", output_prefix="wreckgar",
    test_mode=False,save_model_on_epoch=False,
):
    acc_steps = 100
    device=torch.device("cuda")
    model = model.cuda()
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss=0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        print(loss)
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    return model


modeler = train(dataset, model, tokenizer)