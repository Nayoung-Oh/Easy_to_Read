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
for f in file_paths:
  vocab_transform.append(build_vocab_from_iterator(yield_tokens_from_file(f),
                                                min_freq=1,
                                                specials=special_symbols,
                                                special_first=True))

# Set UNK_IDX as the default index. This index is returned when the token is not found. 
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary. 
for i in range(2):
  vocab_transform[i].set_default_index(UNK_IDX)

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[0])
TGT_VOCAB_SIZE = len(vocab_transform[1])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 16
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, 
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0002, betas=(0.9, 0.98), eps=1e-9)


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
    cls_batch, src_batch, tgt_batch = [], [], []
    for cls_sample, src_sample, tgt_sample in batch:
        cls_batch.append(torch.tensor(cls_sample))
        src_batch.append(text_transform[0](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[1](tgt_sample.rstrip("\n")))
    cls_batch = torch.stack(cls_batch)
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return cls_batch, src_batch, tgt_batch

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

# sumwriter = SummaryWriter("./logs")

def train_epoch(model, optimizer, epoch):
    model.train()
    losses = 0
    train_iter = WikiDataset("./wikilarge/filtered_data.csv")
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    max_len = len(train_dataloader)
    start_time = timer()

    for i, (cl, src, tgt) in enumerate(train_dataloader):
        
        cl = cl.to(DEVICE)
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask, cl)

        optimizer.zero_grad()

        del src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, cl
        gc.collect()
        torch.cuda.empty_cache()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        # if i % 500 == 0 or i == (max_len - 1):
        #   end_time = timer()
        #   print(i, '/', max_len, end_time - start_time)
        #   start_time = timer()
        #   tmp = loss.item()
        #   sumwriter.add_scalar('training_loss', tmp / BATCH_SIZE, epoch*max_len + i)
        
        losses += loss.item()
        del logits, tgt, tgt_out, loss
        gc.collect()
        torch.cuda.empty_cache()
    
    del train_dataloader, train_iter
    return losses / max_len


def evaluate(model, epoch):
    model.eval()
    losses = 0
    val_iter =  WikiDataset("./wikilarge/filtered_data_val.csv")
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    max_len = len(val_dataloader)
    # start_time = timer()
    with torch.no_grad():
        for i, (cl, src, tgt) in enumerate(val_dataloader):
            # if i % 100 == 0:
            #     end_time = timer()
            #     print(i, '/', max_len, end_time - start_time)
            #     start_time = timer()
            cl = cl.to(DEVICE)
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask, cl)
            
            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()
    
    # sumwriter.add_scalar('validation_loss', losses/max_len, epoch * max_len)
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
def translate(model: torch.nn.Module, src_sentence: str, cl: Tensor):
    model.eval()
    src = text_transform[0](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens + 1, num_tokens + 1)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX, cl=cl).flatten()
    return " ".join(vocab_transform[1].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

NUM_EPOCHS = 6
# transformer.load_state_dict(torch.load(""))
# val loss 3.95
min_val_loss = 10
with open("large_training_res.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, epoch)
        end_time = timer()
        val_loss = evaluate(transformer, epoch)
        print((f"Epoch: {epoch + 1}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        writer.writerow([epoch, train_loss, val_loss])
        if epoch % 10 == 0 or (epoch > 10 and val_loss < min_val_loss):
            print("save model")
            torch.save(transformer.state_dict(), f"large_{epoch}_val_{val_loss:.3f}")
        min_val_loss = min(val_loss, min_val_loss)

# transformer.load_state_dict(torch.load("layer3_ 20_val_3.336"))
# print(translate(transformer, "Charity Rice is known for his charity work .", torch.tensor([[1.0,1.0,0.9411764705882353,1.0,1.2777777777777777]])))
# He is known for his band for his father , for his name for his
# He is known for his own music music music for his own music music music
# Rice is also known for his charity work
# tgt: Rice is also known for his charity work . 