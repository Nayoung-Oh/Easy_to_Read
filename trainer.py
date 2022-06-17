'''
@ Contributor: Nayoung-Oh

Some parts are referred to https://pytorch.org/tutorials/beginner/translation_transformer.html
'''

from torch.utils.data import DataLoader
from torch import Tensor
import torch
import torch.nn as nn
import io
from model import Seq2SeqTransformer, Seq2SeqTransformerBaseline
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
from torchtext.data.utils import get_tokenizer


class Trainer():
    def __init__(self, data, model, loss):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.vocab_transform = []
        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.token_transform = get_tokenizer('spacy', language='en_core_web_sm')

        self.vocab_transform.append(torch.load("vocab_train_src.pth"))
        self.vocab_transform.append(torch.load("vocab_train_dst.pth"))

        for i in range(2):
            self.vocab_transform[i].set_default_index(self.UNK_IDX)
        
        stopwords_gen = stopwords.words('english')
        self.stop_idx = [a for a in self.vocab_transform[1](stopwords_gen) if a != 0]
        dot = ['.', '..', '...']
        self.dot_idx = [a for a in self.vocab_transform[1](dot) if a != 0]
        self.rest_idx = self.vocab_transform[1]([','])
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
        self.easy_idx = [a for a in self.vocab_transform[1](word_freq) if a != 0]

        torch.manual_seed(0)

        SRC_VOCAB_SIZE = len(self.vocab_transform[0])
        TGT_VOCAB_SIZE = len(self.vocab_transform[1])

        EMB_SIZE = 512
        NHEAD = 8
        FFN_HID_DIM = 512
        self.BATCH_SIZE = 32
        NUM_ENCODER_LAYERS = 3
        NUM_DECODER_LAYERS = 3

        self.model = model
        self.loss = loss
        self.data = data

        if self.model == "feature":
            transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, 
                                        NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
        else:
            transformer = Seq2SeqTransformerBaseline(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, 
                                        NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.transformer = transformer.to(self.DEVICE)

        if self.loss == "weighted":
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.PAD_IDX, reduction='none')
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.PAD_IDX)

        self.optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0002, betas=(0.9, 0.98), eps=1e-9)

        self.__generte_text_transform()

        self.sumwriter = SummaryWriter("./logs")

    def __generte_text_transform(self):
        # helper function to club together sequential operations
        def sequential_transforms(*transforms):
            def func(txt_input):
                for transform in transforms:
                    txt_input = transform(txt_input)
                return txt_input
            return func

        # function to add BOS/EOS and create tensor for input sequence indices
        def tensor_transform(token_ids):
            return torch.cat((torch.tensor([self.BOS_IDX]), 
                            torch.tensor(token_ids), 
                            torch.tensor([self.EOS_IDX])))

        # src and tgt language text transforms to convert raw strings into tensors indices
        self.text_transform = {}
        for ln in range(2):
            self.text_transform[ln] = sequential_transforms(self.token_transform, #Tokenization
                                                    self.vocab_transform[ln], #Numericalization
                                                    tensor_transform) # Add BOS/EOS and create tensor

    # function to collate data samples into batch tesors
    def __collate_fn(self, batch):
        info_batch, cls_batch, src_batch, tgt_batch = [], [], [], []
        for info_sample, cls_sample, src_sample, tgt_sample in batch:
            info_batch.append(torch.tensor(info_sample))
            cls_batch.append(torch.tensor(cls_sample))
            src_batch.append(self.text_transform[0](src_sample.rstrip("\n")))
            tgt_batch.append(self.text_transform[1](tgt_sample.rstrip("\n")))
        info_batch = torch.stack(info_batch)
        cls_batch = torch.stack(cls_batch)
        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.PAD_IDX)
        return info_batch, cls_batch, src_batch, tgt_batch

    def __generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def __create_mask(self, src, tgt):
        src_seq_len, sh = src.shape
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.__generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len + 1, src_seq_len + 1),device=self.DEVICE).type(torch.bool)

        if self.model == "feature":
            src_padding_mask = torch.zeros(sh, src_seq_len + 1, device=self.DEVICE)
            src_tmp = (src == self.PAD_IDX).transpose(0, 1)
            src_padding_mask[:, 1:src_seq_len+1] = src_tmp
        else:
            src_padding_mask = (src == self.PAD_IDX).transpose(0, 1)
        tgt_padding_mask = (tgt == self.PAD_IDX).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def __feature_cal(self, tmp, info, cl):
        next_word = np.max(tmp, axis=2)
        next_word = next_word.transpose(1, 0)
        twos = np.ones((next_word.shape[0], 1)) * self.EOS_IDX
        new = np.append(next_word, twos, axis=1)
        length = []

        for i in range(next_word.shape[0]):
            l = np.where(new[i, :] == self.EOS_IDX)
            length.append(l[0][0] + 1)
        length = np.array(length).reshape(-1)
        
        # info # batch X 5
        
        dot_count = np.isin(next_word, self.dot_idx).sum(axis=1)
        rest_count = np.isin(next_word, self.rest_idx).sum(axis=1) + 1
        stop_count = np.isin(next_word, self.stop_idx).sum(axis=1) + 1
        easy_count = np.isin(next_word, self.easy_idx).sum(axis=1) 
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
        scaled_weight = np.clip(scaled_weight, 1.0, 2.5)
        return scaled_weight

    def train_epoch(self, epoch):
        self.transformer.train()
        losses = 0
        train_iter = WikiDataset("./"+self.data+"/train.csv")
        train_dataloader = DataLoader(train_iter, batch_size=self.BATCH_SIZE, shuffle=False, collate_fn=self.__collate_fn, pin_memory=True)
        max_len = len(train_dataloader)
        start_time = timer()

        for i, (info, cl, src, tgt) in enumerate(train_dataloader):
            
            cl = cl.to(self.DEVICE)
            src = src.to(self.DEVICE)
            tgt = tgt.to(self.DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.__create_mask(src, tgt_input)

            if self.model == "feature":
                logits = self.transformer(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask, cl)
            else:
                logits = self.transformer(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            self.optimizer.zero_grad()

            del src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
            gc.collect()
            torch.cuda.empty_cache()

            tgt_out = tgt[1:, :]
            tmp = logits.cpu().detach().numpy()
            b_size = logits.shape[1]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            if self.loss == "weighted":
                loss = loss.reshape(-1, b_size).mean(axis=0)
                info = info.cpu().numpy()
                cl = cl.cpu().numpy()
                weight = self.__feature_cal(tmp, info, cl)
                weight = torch.Tensor(weight).to(self.DEVICE)
                weighted_loss = (weight * loss).mean()
                weighted_loss.backward()
                tmp = weighted_loss.item()
            else:
                loss.backward()
                tmp = loss.item()
            
            self.optimizer.step()

            if i % 200 == 0 or i == (max_len - 1):
                end_time = timer()
                print(i, '/', max_len, end_time - start_time)
                start_time = timer()
                self.sumwriter.add_scalar('training_loss', tmp, (epoch-1)*max_len + i)
            
            losses += tmp
            del logits, tgt, tgt_out, loss
            gc.collect()
            torch.cuda.empty_cache()
        
        del train_dataloader, train_iter
        return losses / max_len


    def evaluate(self, epoch):

        self.transformer.train()
        losses = 0
        val_iter = WikiDataset("./"+self.data+"/valid.csv")
        val_dataloader = DataLoader(val_iter, batch_size=self.BATCH_SIZE, shuffle=False, collate_fn=self.__collate_fn, pin_memory=True)
        max_len = len(val_dataloader)

        for i, (info, cl, src, tgt) in enumerate(val_dataloader):
            
            cl = cl.to(self.DEVICE)
            src = src.to(self.DEVICE)
            tgt = tgt.to(self.DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.__create_mask(src, tgt_input)

            if self.model == "feature":
                logits = self.transformer(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask, cl)
            else:
                logits = self.transformer(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            self.optimizer.zero_grad()

            del src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
            gc.collect()
            torch.cuda.empty_cache()

            tgt_out = tgt[1:, :]
            tmp = logits.cpu().detach().numpy()
            b_size = logits.shape[1]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            if self.loss == "weighted":
                loss = loss.reshape(-1, b_size).mean(axis=0)
                info = info.cpu().numpy()
                cl = cl.cpu().numpy()
                weight = self.__feature_cal(tmp, info, cl)
                weight = torch.Tensor(weight).to(self.DEVICE)
                weighted_loss = (weight * loss).mean()
                tmp = weighted_loss.item()
            else:
                tmp = loss.item()
            
            losses += tmp
            del logits, tgt, tgt_out, loss
            gc.collect()
            torch.cuda.empty_cache()
        
        del val_dataloader, val_iter
        self.sumwriter.add_scalar('validation_loss', losses/max_len, epoch * max_len)
        return losses / max_len

    def __greedy_decode(self, src, src_mask, max_len, start_symbol, cl):
        src = src.to(self.DEVICE)
        cl = cl.to(self.DEVICE)
        src_mask = src_mask.to(self.DEVICE)

        memory = self.transformer.encode(src, src_mask, cl)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.DEVICE)
        for i in range(max_len-1):
            memory = memory.to(self.DEVICE)
            tgt_mask = (self.__generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(self.DEVICE)
            out = self.transformer.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.transformer.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == self.EOS_IDX:
                break
        return ys

    def simplify(self, src_sentence, cl):
        self.transformer.eval()
        src = self.text_transform[0](src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens + 1, num_tokens + 1)).type(torch.bool)
        tgt_tokens = self.__greedy_decode(
            src, src_mask, max_len=num_tokens + 5, start_symbol=self.BOS_IDX, cl=cl).flatten()
        return " ".join(self.vocab_transform[1].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

