'''
@ Contributor: Nayoung-Oh, darae-lee

Calculate features and related information to generate preprocessed dataset files
'''

import csv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import argparse
from torchtext.vocab import build_vocab_from_iterator
import io
from torchtext.data.utils import get_tokenizer
from torch import save

class DataProcessor():
  def __init__(self, datatype):
    word_path = './unigram_freq.csv'
    self.word_freq = []
    with open(word_path) as word_file:
      reader = csv.reader(word_file)
      for row in reader:
        self.word_freq.append(row[0])
    self.wordnet_lemmatizer = WordNetLemmatizer()
    self.stopwords = stopwords.words('english')
    self.type = 'train'
    if datatype == "wikilarge":
      self.src_file = "./wikilarge/wiki.full.aner.train.src"
      self.dest_file = "./wikilarge/wiki.full.aner.train.dst"
      self.write_to = "./wikilarge/train.csv"
    else:
      self.src_file = "./wikismall/PWKP_108016.tag.new.aner.train.src"
      self.dest_file = "./wikismall/PWKP_108016.tag.new.aner.train.dst"
      self.write_to = "./wikismall/train.csv"

  def change_type(self, to_type):
    self.src_file = self.src_file.replace(self.type, to_type)
    self.dest_file = self.dest_file.replace(self.type, to_type)
    self.write_to = self.write_to.replace(self.type, to_type)
    self.type = to_type

  def generate_data(self):
    
    with open(self.dest_file, 'r', encoding = 'utf-8') as f1:
      train_dest = f1.readlines()
    with open(self.src_file, 'r', encoding = 'utf-8') as f2:
      train_src = f2.readlines()

    temp_word_dest = [sent.strip('\n').split(' ') for sent in train_dest]
    temp_word_src = [sent.strip('\n').split(' ') for sent in train_src]
    prepro_dest = [[w.strip("."",") for w in sent] for sent in temp_word_dest]
    prepro_src = [[w.strip("."",") for w in sent] for sent in temp_word_src]
    prepro_dest = [[w for w in sent if w] for sent in prepro_dest]
    prepro_src = [[w for w in sent if w] for sent in prepro_src]
    prepro2_dest = [[w for w in sent if w not in self.stopwords] for sent in prepro_dest]
    prepro2_dest = [[self.wordnet_lemmatizer.lemmatize(w) for w in sent] for sent in prepro2_dest]
    prepro2_src = [[w for w in sent if w not in self.stopwords] for sent in prepro_src]
    prepro2_src = [[self.wordnet_lemmatizer.lemmatize(w) for w in sent] for sent in prepro2_src]

    self.all_words = {}
    easyNum = 10000 # top 10000 words

    def preprocess_freqword(sent):
      for word in sent:
        self.all_words[word] = 0

    for sent in prepro2_dest:
      preprocess_freqword(sent)
    for sent in prepro2_src:
      preprocess_freqword(sent)
    for word in self.word_freq[1:easyNum]:
      self.all_words[word] = 1
    
  def __count_freqword(self, sent):
    res = 0
    for word in sent:
      if self.all_words.get(word, 0) == 1:
        res = res + 1
    return res

  def make_features(self, src, dest):
    # remove '\n' and split by ' '
    temp_src = src.strip('\n').split(' ')
    temp_dest = dest.strip('\n').split(' ')

    # (1) count dot and (2) rest
    dotCount_src = 0;  dotCount_dest = 0
    restCount_src = 0;  restCount_dest = 0
    for w in temp_src:
      if w in ['.', '..', '...']:
        dotCount_src = dotCount_src+1
      if w == ',':
        restCount_src = restCount_src+1
    for w in temp_dest:
      if w in ['.', '..', '...']:
        dotCount_dest = dotCount_dest+1
      if w == ',':
        restCount_dest = restCount_dest+1
    if dotCount_src == 0:
      dotCount_src = 1
    if dotCount_dest == 0:
      dotCount_dest = 1

    # remove '.'
    temp_src = [w.strip("."",") for w in temp_src]
    temp_src = [w for w in temp_src if w != '']
    temp_dest = [w.strip("."",") for w in temp_dest]
    temp_dest = [w for w in temp_dest if w != '']
    # (3) count stopwords
    stopCount_src = 0;  stopCount_dest = 0
    for w in temp_src:
      if w in self.stopwords:
        stopCount_src = stopCount_src+1
    for w in temp_dest:
      if w in self.stopwords:
        stopCount_dest = stopCount_dest+1

    # remove stopwords and lemmatize
    temp_src = [w for w in temp_src if w not in self.stopwords]
    temp_dest = [w for w in temp_dest if w not in self.stopwords]
    temp_src = [self.wordnet_lemmatizer.lemmatize(w) for w in temp_src]
    temp_dest = [self.wordnet_lemmatizer.lemmatize(w) for w in temp_dest]

    # (4) len
    lenCount_src = len(temp_src)
    lenCount_dest = len(temp_dest)

    # (5) Count easywords
    easywordCount_src = self.__count_freqword(temp_src)
    easywordCount_dest = self.__count_freqword(temp_dest)
    easywordRatio_src = (easywordCount_src)/(lenCount_src+1)
    easywordRatio_dest = (easywordCount_dest)/(lenCount_dest+1)

    # make all ratio
    dotRatio = dotCount_dest/dotCount_src
    restRatio = (restCount_dest+1)/(restCount_src+1)
    stopRatio = (stopCount_dest+1)/(stopCount_src+1)
    lenRatio = (lenCount_src+1)/(lenCount_dest+1)
    easyRatio = (easywordRatio_dest+0.1)/(easywordRatio_src+0.1)

    return [dotCount_src, restCount_src, stopCount_src, lenCount_src, easywordRatio_src, dotRatio, restRatio, stopRatio, lenRatio, easyRatio]

  def __save_vocab(self):
    def yield_tokens_from_file(file_path):
      with io.open(file_path, encoding = 'utf-8') as f:
        for line in f:
            line = line.rstrip()
            yield token_transform(line)
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
    token_transform = get_tokenizer('spacy', language='en_core_web_sm')

    vocab_transform = build_vocab_from_iterator(yield_tokens_from_file(self.src_file),
                                                min_freq=3,
                                                specials=special_symbols,
                                                special_first=True)
                                              
    save(vocab_transform, "vocab_"+self.type+"_src.pth")

    vocab_transform = build_vocab_from_iterator(yield_tokens_from_file(self.dest_file),
                                                min_freq=3,
                                                specials=special_symbols,
                                                special_first=True)
                                              
    save(vocab_transform, "vocab_"+self.type+"_dst.pth")

  def preprocess_data(self):
    files = ["train", "valid", "test"]

    for i in range(len(files)):
      self.generate_data()

      with open(self.write_to, "w", encoding = 'utf-8', newline='') as writecsv:
        with open(self.src_file, encoding = 'utf-8') as oris:
            with open(self.dest_file, encoding = 'utf-8') as orid:
                writer = csv.writer(writecsv)
                oris = oris.readlines()
                orid = orid.readlines()
                for s, d in zip(oris, orid):
                  feat = self.make_features(s, d)
                  feat.append(s.rstrip())
                  feat.append(d.rstrip())
                  writer.writerow(feat)
      if i == 0:
        self.__save_vocab()
      if i != len(files) - 1:
        self.change_type(files[i+1])
      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='wikilarge')

    args = parser.parse_args()

    processor = DataProcessor(vars(args)["data"])
    processor.preprocess_data()
