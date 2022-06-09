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
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from scipy import spatial


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
                                                min_freq=3,
                                                specials=special_symbols,
                                                special_first=True))

# Set UNK_IDX as the default index. This index is returned when the token is not found. 
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary. 
for i in range(2):
  vocab_transform[i].set_default_index(UNK_IDX)

stopwords = stopwords.words('english')
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

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction='none')

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

# sumwriter = SummaryWriter("./logs/feature_big")

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

NUM_EPOCHS = 3
# transformer.load_state_dict(torch.load("new_big_1_val_1.699"))
# # transformer.load_state_dict(torch.load(""))
# # val loss 3.95
# min_val_loss = 3
# with open("new_training_big.csv", "a", newline="") as f:
#     writer = csv.writer(f)
#     for epoch in range(2, NUM_EPOCHS+1):
#         start_time = timer()
#         # train_loss = 0
#         train_loss = train_epoch(transformer, optimizer, epoch)
#         end_time = timer()
#         val_loss = evaluate(transformer, epoch)
#         print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
#         writer.writerow([epoch, train_loss, val_loss])
#         if epoch % 1 == 0 or (epoch > 10 and val_loss < min_val_loss):
#             print("save model")
#             torch.save(transformer.state_dict(), f"new_big_{epoch}_val_{val_loss:.3f}")
#         min_val_loss = min(val_loss, min_val_loss)

# transformer.load_state_dict(torch.load("layer3_ 20_val_3.336"))
# print(simplify(transformer, "Charity Rice is known for his charity work .", torch.tensor([[1.0,1.0,0.9411764705882353,1.0,1.2777777777777777]])))

# He is known for his band for his father , for his name for his
# He is known for his own music music music for his own music music music
# Rice is also known for his charity work
# tgt: Rice is also known for his charity work . 

transformer.load_state_dict(torch.load("new_large_5_val_1.045"))

# transformer.load_state_dict(torch.load("large_3_val_2.338"))
# print(simplify(transformer, "The incident has been the subject of numerous reports as to ethics in scholarship .", torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.float)))
# print(simplify(transformer, "The incident has been the subject of numerous reports as to ethics in scholarship .", torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float)))
# print(simplify(transformer, "I really really love you more than anyone .", torch.tensor([[1.0,1.0,1.0,1.0,1.0]], dtype=torch.float)))
# print(simplify(transformer, "I really really love you more than anyone .", torch.tensor([[1.0,1.0,1.0,0.8,1.0]], dtype=torch.float)))
# print(simplify(transformer, "I really really love you more than anyone .", torch.tensor([[1.0,1.0,1.0,1.2,1.0]], dtype=torch.float)))


# The incident has been the subject of numerous reports regarding scholarship ethics .

# with open("./wikilarge/new_data_test.csv", encoding = 'utf-8') as f:
#     with open("./wikilarge/new_report.txt", "w", encoding='utf-8') as res:
#         reader = csv.reader(f)
#         for l in reader:
#             res.write(simplify(transformer, l[-2], torch.tensor([[float(i) for i in l[5:10]]])) + '\n')

# src_path = 'wikilarge/wiki.full.aner.test.src'
# dest_path = 'wikilarge/wiki.full.aner.test.dst'



# stopwords = nltk.corpus.stopwords.words('english')
# wordnet_lemmatizer = WordNetLemmatizer()


# word_path = './unigram_freq.csv'
# word_data = csv.reader(open(word_path, 'r'))
# word_freq = []
# with open(word_path) as word_file:
#   reader = csv.reader(word_file)
#   for row in reader:
#       word_freq.append(row[0])

# with open(dest_path, 'r') as f1:
#     test_dest = f1.readlines()
# with open(src_path, 'r') as f2:
#     test_src = f2.readlines()

# def count_rest(list):
#   count = 0
#   for w in list:
#     if w==',':
#       count = count+1
#   return count
# def count_dot(list):
#   count = 0
#   for w in list:
#     if w=='.':
#       count = count+1
#   if count==0:
#     count = 1
#   return count
# def count_freqword(sent, num):
#   res = 0
#   for word in sent:
#     if word in word_freq[1:int(num)]:
#       res = res + 1
#   return res
# def count_stopword(sent):
#   res = 0
#   for word in sent:
#     if word in stopwords:
#       res = res+1
#   return res
# def preprocess_freqword(sent):
#   for word in sent:
#     all_words[word] = 0
# def count_freqword(sent):
#   res = 0
#   for word in sent:
#     if not all_words.get(word):
#       continue
#     if all_words[word] == 1:
#       res = res + 1
#   return res

# temp_word_dest = [sent.strip('\n').split(' ') for sent in test_dest]
# temp_word_src = [sent.strip('\n').split(' ') for sent in test_src]
# prepro_dest = [[w.strip("."",") for w in sent] for sent in temp_word_dest]
# prepro_src = [[w.strip("."",") for w in sent] for sent in temp_word_src]
# prepro_dest = [[w for w in sent if w] for sent in prepro_dest]
# prepro_src = [[w for w in sent if w] for sent in prepro_src]
# prepro2_dest = [[w for w in sent if w not in stopwords] for sent in prepro_dest]
# prepro2_dest = [[wordnet_lemmatizer.lemmatize(w) for w in sent] for sent in prepro2_dest]
# prepro2_src = [[w for w in sent if w not in stopwords] for sent in prepro_src]
# prepro2_src = [[wordnet_lemmatizer.lemmatize(w) for w in sent] for sent in prepro2_src]

# all_words = {}
# easyNum = 10000 # top 10000 words

# for sent in prepro2_dest:
#   preprocess_freqword(sent)
# for sent in prepro2_src:
#   preprocess_freqword(sent)
# for word in word_freq[1:easyNum]:
#   all_words[word] = 1

# def make_features(src, dest):
#   # remove '\n' and split by ' '
#   temp_src = src.strip('\n').split(' ')
#   temp_dest = dest.strip('\n').split(' ')

#   # (1) count dot and (2) rest
#   dotCount_src = 0;  dotCount_dest = 0
#   restCount_src = 0;  restCount_dest = 0
#   for w in src:
#     if w in ['.', '..', '...']:
#       dotCount_src = dotCount_src+1
#     if w == ',':
#       restCount_src = restCount_src+1
#   for w in dest:
#     if w in ['.', '..', '...']:
#       dotCount_dest = dotCount_dest+1
#     if w == ',':
#       restCount_dest = restCount_dest+1
#   if dotCount_src == 0:
#     dotCount_src = 1
#   if dotCount_dest == 0:
#     dotCount_dest = 1

#   # remove '.'
#   temp_src = [w.strip("."",") for w in temp_src]
#   temp_src = [w for w in temp_src if w]
#   temp_dest = [w.strip("."",") for w in temp_dest]
#   temp_dest = [w for w in temp_dest if w]

#   # (3) count stopwords
#   stopCount_src = 0;  stopCount_dest = 0
#   for w in src:
#     if w in stopwords:
#       stopCount_src = stopCount_src+1
#   for w in dest:
#     if w in stopwords:
#       stopCount_dest = stopCount_dest+1

#   # remove stopwords and lemmatize
#   temp_src = [w for w in temp_src if w not in stopwords]
#   temp_dest = [w for w in temp_dest if w not in stopwords]
#   temp_src = [wordnet_lemmatizer.lemmatize(w) for w in temp_src]
#   temp_dest = [wordnet_lemmatizer.lemmatize(w) for w in temp_dest]

#   # (4) len
#   lenCount_src = len(temp_src)
#   lenCount_dest = len(temp_dest)

#   # (5) Count easywords
#   easywordCount_src = count_freqword(temp_src)
#   easywordCount_dest = count_freqword(temp_dest)
#   easywordRatio_src = (easywordCount_src)/(lenCount_src+1)
#   easywordRatio_dest = (easywordCount_dest)/(lenCount_dest+1)

#   # make all ratio
#   dotRatio = dotCount_dest/dotCount_src
#   restRatio = (restCount_dest+1)/(restCount_src+1)
#   stopRatio = (stopCount_dest+1)/(stopCount_src+1)
#   lenRatio = (lenCount_src+1)/(lenCount_dest+1)
#   easyRatio = (easywordRatio_dest+0.1)/(easywordRatio_src+0.1)

#   return [dotRatio, restRatio, stopRatio, lenRatio, easyRatio]

# # increment index score by inc
# def change_scores(scores, index, inc):
#   scores[index] = scores[index] + inc
#   return scores
# # compute difference b/w output and input (nonuse)
# def designated_change(index, input_feature, output_feature):
#   return output_feature[index]-input_feature[index]
# # compute other scores diff (nonuse)
# def other_change(index, input_feature, output_feature):
#   distance = []
#   for i in range(5):
#     if i==index:
#       distance.append(0.0)
#       continue
#     distance.append((input_feature[i]-output_feature[i])**2)
#   return distance

# # return 2 scores
# def total_output(src, dest, index, delta):
#   original_features = make_features(src, dest)
#   designated_features = change_scores(original_features.copy(), index, delta)
#   new_dest = simplify(transformer, src, torch.tensor([designated_features], dtype=torch.float))
#   output_features = make_features(src, new_dest)
#   #final_designated = designated_change(feature_index, designated_features, output_features)
#   #final_others = math.sqrt(sum(other_change(feature_index, designated_features, output_features)))
#   #return final_designated, final_others
#   return designated_features, output_features

# def sim(List1, List2):
#     result = 1 - spatial.distance.cosine(List1, List2)
#     return result

# total_len = len(test_src)

# f0_designates = []
# f0_outputs = []
# for i in range(total_len):
#   #if i%100==0:
#   #  print(i)
#   f0_designate, f0_output = total_output(test_src[i],test_dest[i],0,1.0)
#   f0_designates.append(f0_designate)
#   f0_outputs.append(f0_output)

# f1_designates = []
# f1_outputs = []
# for i in range(total_len):
#   f1_designate, f1_output = total_output(test_src[i],test_dest[i],1,1.0)
#   f1_designates.append(f1_designate)
#   f1_outputs.append(f1_output)

# f2_designates = []
# f2_outputs = []
# for i in range(total_len):
#   f2_designate, f2_output = total_output(test_src[i],test_dest[i],2,0.2)
#   f2_designates.append(f2_designate)
#   f2_outputs.append(f2_output)

# f3_designates = []
# f3_outputs = []
# for i in range(total_len):
#   f3_designate, f3_output = total_output(test_src[i],test_dest[i],3,0.2)
#   f3_designates.append(f3_designate)
#   f3_outputs.append(f3_output)

# f4_designates = []
# f4_outputs = []
# for i in range(total_len):
#   f4_designate, f4_output = total_output(test_src[i],test_dest[i],4,0.2)
#   f4_designates.append(f4_designate)
#   f4_outputs.append(f4_output)

# result0=sim(sum(f0_designates,[]),sum(f0_outputs,[])) # +1
# result1=sim(sum(f1_designates,[]),sum(f1_outputs,[])) # +1
# result2=sim(sum(f2_designates,[]),sum(f2_outputs,[])) # +0.2
# result3=sim(sum(f3_designates,[]),sum(f3_outputs,[])) # +0.2
# result4=sim(sum(f4_designates,[]),sum(f4_outputs,[])) # +0.2
# print(result0, result1, result2, result3, result4)