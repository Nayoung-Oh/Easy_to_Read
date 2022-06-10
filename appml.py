from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from typing import Iterable, List
from torch import Tensor
import torch
import torch.nn as nn
import io
from model import Seq2SeqTransformer
from dataset import WikiDataset
from timeit import default_timer as timer
from torch.nn.utils.rnn import pad_sequence
import gc, csv
from torch.utils.tensorboard import SummaryWriter
from nltk.corpus import stopwords as original_stopwords
import numpy as np
from model_baseline import Seq2SeqTransformerBaseline

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"

LANGUAGE = 'en'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


token_transform = None
vocab_transform = []

token_transform = get_tokenizer('spacy', language='en_core_web_sm')

def yield_tokens_from_file(file_path):
  with io.open(file_path, encoding = 'utf-8') as f:
    for line in f:
        line = line.rstrip()
        yield token_transform(line)

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
 
# Create torchtext's Vocab object 
file_paths = ["./wikilarge/wiki.full.aner.train.src", "./wikilarge/wiki.full.aner.train.dst"]
# file_paths = ["./wikismall/PWKP_108016.tag.new.aner.train.src", "./wikismall/PWKP_108016.tag.new.aner.train.dst"]
# for f in file_paths:
#   vocab_transform.append(build_vocab_from_iterator(yield_tokens_from_file(f),
#                                                 min_freq=3,
#                                                 specials=special_symbols,
#                                                 special_first=True))

print("pickle vocab_transform loading...")
import pickle
if not os.path.isfile('vocab_transform.pkl'):
    vocab_transform = []
    for f in file_paths:
      vocab_transform.append(build_vocab_from_iterator(yield_tokens_from_file(f),
                                                    min_freq=3,
                                                    specials=special_symbols,
                                                    special_first=True))
    with open('vocab_transform.pkl', 'wb') as f:
         pickle.dump(vocab_transform, f)
else:
    with open('vocab_transform.pkl', 'rb') as f:
         vocab_transform = pickle.load(f)

# Set UNK_IDX as the default index. This index is returned when the token is not found. 
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary. 
for i in range(2):
  vocab_transform[i].set_default_index(UNK_IDX)

stopwords = original_stopwords.words('english')
stop_idx = [a for a in vocab_transform[1](stopwords) if a != 0]
dot = ['.', '..', '...']
dot_idx = [a for a in vocab_transform[1](dot) if a != 0]
rest_idx = vocab_transform[1]([','])
word_path =  "./unigram_freq.csv"
with open(word_path, 'r') as f:
  word_data = csv.reader(f)
  word_freq = []
  count = 0
  for row in word_data:
    word_freq.append(row[0])
    count += 1
    if count > 10000:
      break
easy_idx = [a for a in vocab_transform[1](word_freq) if a != 0]

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[0])
TGT_VOCAB_SIZE = len(vocab_transform[1])
print(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
# exit()
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 32
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, 
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)


transformer_base = Seq2SeqTransformerBaseline(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, 
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

for p in transformer_base.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

transformer_base = transformer_base.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction='none')

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0002, betas=(0.9, 0.98), eps=1e-9)
optimizer_base = torch.optim.Adam(transformer_base.parameters(), lr=0.0002, betas=(0.9, 0.98), eps=1e-9)


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in range(2):
    text_transform[ln] = sequential_transforms(token_transform, #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fn(batch):
    info_batch, cls_batch, src_batch, tgt_batch = [], [], [], []
    for info_sample, cls_sample, src_sample, tgt_sample in batch:
        info_batch.append(torch.tensor(info_sample))
        cls_batch.append(torch.tensor(cls_sample))
        src_batch.append(text_transform[0](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[1](tgt_sample.rstrip("\n")))
    info_batch = torch.stack(info_batch)
    cls_batch = torch.stack(cls_batch)
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return info_batch, cls_batch, src_batch, tgt_batch

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len, sh = src.shape
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len + 1, src_seq_len + 1),device=DEVICE).type(torch.bool)

    src_padding_mask = torch.zeros(sh, src_seq_len + 1, device=DEVICE)
    src_tmp = (src == PAD_IDX).transpose(0, 1)
    src_padding_mask[:, :src_seq_len] = src_tmp
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# sumwriter = SummaryWriter("./logs/feature")

def feature_cal(tmp, info, cl):
  next_word = np.max(tmp, axis=2)
  next_word = next_word.transpose(1, 0)
  twos = np.ones((next_word.shape[0], 1)) * EOS_IDX
  new = np.append(next_word, twos, axis=1)
  length = []

  for i in range(next_word.shape[0]):
    l = np.where(new[i, :] == EOS_IDX)
    length.append(l[0][0] + 1)
  length = np.array(length).reshape(-1)
  
  # info # batch X 5
  
  dot_count = np.isin(next_word, dot_idx).sum(axis=1)
  rest_count = np.isin(next_word, rest_idx).sum(axis=1) + 1
  stop_count = np.isin(next_word, stop_idx).sum(axis=1) + 1
  easy_count = np.isin(next_word, easy_idx).sum(axis=1) 
  easy_ratio = easy_count / length + 0.1

  for val in range(1, 4):
    info[:, val] += 1
  info[:, 4] += 0.1

  dest = np.stack([dot_count, rest_count, stop_count, length, easy_ratio]).transpose(1, 0)
  feature = dest / info
  feature[:, 3] = 1 / feature[:, 3]
  weight = ((feature - cl)**2)
  weight = weight.mean(axis=1)
  scaled_weight = weight * 0.1 + 1.0
#   print(weight)
#   print(scaled_weight)
  scaled_weight = np.clip(scaled_weight, 1.0, 2.5)
  return scaled_weight

def train_epoch(model, optimizer, epoch):
    model.train()
    losses = 0
    train_iter = WikiDataset("./wikilarge/new_data_train.csv")
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    max_len = len(train_dataloader)
    start_time = timer()

    for i, (info, cl, src, tgt) in enumerate(train_dataloader):
        
        cl = cl.to(DEVICE)
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask, cl)

        optimizer.zero_grad()

        del src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
        gc.collect()
        torch.cuda.empty_cache()

        tgt_out = tgt[1:, :]
        tmp = logits.cpu().detach().numpy()
        b_size = logits.shape[1]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss = loss.reshape(-1, b_size).mean(axis=0)

        info = info.cpu().numpy()
        cl = cl.cpu().numpy()
        weight = feature_cal(tmp, info, cl)
        weight = torch.Tensor(weight).to(DEVICE)
        # exit()
        weighted_loss = (weight * loss).mean()
        weighted_loss.backward()
        
        optimizer.step()
        if i % 200 == 0 or i == (max_len - 1):
          end_time = timer()
          print(i, '/', max_len, end_time - start_time)
          start_time = timer()
          tmp = weighted_loss.item()
          sumwriter.add_scalar('training_loss', tmp, (epoch-1)*max_len + i)
        
        losses += weighted_loss.item()
        del logits, tgt, tgt_out, loss, cl, weight, weighted_loss
        gc.collect()
        torch.cuda.empty_cache()
    
    del train_dataloader, train_iter
    return losses / max_len


def evaluate(model, epoch):
    model.eval()
    losses = 0
    val_iter =  WikiDataset("./wikilarge/new_data_valid.csv")
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    max_len = len(val_dataloader)
    start_time = timer()
    with torch.no_grad():
        for i, (info, cl, src, tgt) in enumerate(val_dataloader):
            if i % 100 == 0:
                end_time = timer()
                print(i, '/', max_len, end_time - start_time)
                start_time = timer()
            cl = cl.to(DEVICE)
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask, cl)
            
            tgt_out = tgt[1:, :]
            tmp = logits.cpu().detach().numpy()
            b_size = logits.shape[1]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss = loss.reshape(-1, b_size).mean(axis=0)

            info = info.cpu().numpy()
            cl = cl.cpu().numpy()
            weight = feature_cal(tmp, info, cl)
            weight = torch.Tensor(weight).to(DEVICE)
            
            weighted_loss = (weight * loss).mean()
            losses += weighted_loss.item()
    
    sumwriter.add_scalar('validation_loss', losses/max_len, epoch * max_len)
    return losses / max_len

# function to generate output sequence using greedy algorithm 
def greedy_decode(model, src, src_mask, max_len, start_symbol, cl):
    src = src.to(DEVICE)
    cl = cl.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask, cl)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def simplify(model: torch.nn.Module, src_sentence: str, cl: Tensor):
    model.eval()
    src = text_transform[0](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens + 1, num_tokens + 1)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX, cl=cl).flatten()
    return " ".join(vocab_transform[1].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

transformer_base = Seq2SeqTransformerBaseline(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, 
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer_base.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer_base = transformer_base.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer_base = torch.optim.Adam(transformer_base.parameters(), lr=0.0002, betas=(0.9, 0.98), eps=1e-9)


# helper function to club together sequential operations
def sequential_transforms_base(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform_base(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in range(2):
    text_transform[ln] = sequential_transforms_base(token_transform, #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform_base) # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fnv(batch):
    info_batch, cls_batch, src_batch, tgt_batch = [], [], [], []
    for info_sample, cls_sample, src_sample, tgt_sample in batch:
        info_batch.append(torch.tensor(info_sample))
        cls_batch.append(torch.tensor(cls_sample))
        src_batch.append(text_transform[0](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[1](tgt_sample.rstrip("\n")))
    info_batch = torch.stack(info_batch)
    cls_batch = torch.stack(cls_batch)
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return info_batch, cls_batch, src_batch, tgt_batch

def generate_square_subsequent_mask_base(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask_base(src, tgt):
    src_seq_len, sh = src.shape
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask_base(_basetgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# sumwriter = SummaryWriter("./logs/baseline")


def train_epoch_base(model, optimizer, epoch):
    model.train()
    losses = 0
    train_iter = WikiDataset("./wikilarge/new_data_train.csv")
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    max_len = len(train_dataloader)
    start_time = timer()

    for i, (info, cl, src, tgt) in enumerate(train_dataloader):
        
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        del src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
        gc.collect()
        torch.cuda.empty_cache()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        loss.backward()
        
        optimizer.step()
        if i % 200 == 0 or i == (max_len - 1):
          end_time = timer()
          print(i, '/', max_len, end_time - start_time)
          start_time = timer()
          tmp = loss.item()
          sumwriter.add_scalar('training_loss', tmp, (epoch-1)*max_len + i)
        
        losses += loss.item()
        del logits, tgt, tgt_out, loss, cl, info
        gc.collect()
        torch.cuda.empty_cache()
    
    del train_dataloader, train_iter
    return losses / max_len


def evaluate_base(model, epoch):
    model.eval()
    losses = 0
    val_iter =  WikiDataset("./wikilarge/new_data_valid.csv")
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    max_len = len(val_dataloader)
    start_time = timer()
    with torch.no_grad():
        for i, (info, cl, src, tgt) in enumerate(val_dataloader):
            if i % 100 == 0:
                end_time = timer()
                print(i, '/', max_len, end_time - start_time)
                start_time = timer()
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask_base(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
            
            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            
            losses += loss.item()
    
    sumwriter.add_scalar('validation_loss', losses/max_len, epoch * max_len)
    return losses / max_len

# function to generate output sequence using greedy algorithm 
def greedy_decode_base(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask_base(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def simplify_base(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[0](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode_base(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[1].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


print("transformer parameter loading...")
transformer.load_state_dict(torch.load("new_large_6_val_1.046"))
transformer_base.load_state_dict(torch.load("baseline_3_val_2.484"))

#########################################################
#########################################################
#################### FLASK CODE #########################
#########################################################
#########################################################

from flask import Flask, render_template, request, send_file
import json
app = Flask(__name__)

@app.route("/")
def hello_world():
    # return render_template("hello.html")
    return send_file("templates/hello.html")


@app.route("/api/askmodel", methods = ['POST'])
def post():
    print(request.get_data())
    print(request.is_json)

    params = json.loads(request.get_data(), encoding='utf-8')
    print(params)

    arr = [
        float(params['feat1']),
        float(params['feat2']),
        float(params['feat3']),
        float(params['feat4']),
        float(params['feat5']),
    ]
    output = simplify(transformer, str(params['text']), torch.tensor([arr], dtype=torch.float))
    transformer_base.load_state_dict(torch.load("baseline_3_val_2.484"))
    output_baseline = simplify_base(transformer_base, str(params['text']))
    data = {"output": output, 
    "output-baseline": output_baseline}
    return data

    
if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8888)

