from operator import index
import random
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv

sources = []
dests = []


# with open("./wikismall/PWKP_108016.tag.80.aner.train.src", encoding = 'utf-8') as sf:
#     with open("./wikismall/PWKP_108016.tag.80.aner.train.dst", encoding = 'utf-8') as df:
#         for (s, d) in zip(sf, df):
#             ls = s.split(" ")
#             ds = d.split(" ")
#             if len(ls) > 5 and len(ls) < 35 and len(ds) > 5:
#                 sources.append(s)
#                 dests.append(d)

# with open("./wikismall/PWKP_108016.tag.new.aner.train.src", 'w', encoding = 'utf-8') as writes:
#     with open("./wikismall/PWKP_108016.tag.new.aner.train.dst", 'w', encoding = 'utf-8') as writed:
#         indexs = [i for i in range(len(sources))]
#         random.shuffle(indexs)
#         for i in indexs:
#             writes.write(sources[i])
#             writed.write(dests[i])

word_path = './unigram_freq.csv'
word_data = csv.reader(open(word_path, 'r'))
word_freq = []
with open(word_path) as word_file:
  reader = csv.reader(word_file)
  for row in reader:
    word_freq.append(row[0])
wordnet_lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')
# all_words[word] = 0(not in wordfreq) or 1(in wordfreq)

# dest_file = "./wikilarge/wiki.full.aner.train.dst"
dest_file = "./asset/dataset/asset.test.simp."
# src_file = "./wikilarge/wiki.full.aner.train.src"
src_file = "./asset/dataset/asset.test.orig"
write_to = "./asset/dataset/data_valid.csv"

train_dest = []
for i in range(10):
  with open(dest_file+str(i), 'r', encoding = 'utf-8') as f1:
    train_dest.extend(f1.readlines())
with open(src_file, 'r', encoding = 'utf-8') as f2:
  train_src = f2.readlines()

temp_word_dest = [sent.strip('\n').split(' ') for sent in train_dest]
temp_word_src = [sent.strip('\n').split(' ') for sent in train_src]
prepro_dest = [[w.strip("."",") for w in sent] for sent in temp_word_dest]
prepro_src = [[w.strip("."",") for w in sent] for sent in temp_word_src]
prepro_dest = [[w for w in sent if w] for sent in prepro_dest]
prepro_src = [[w for w in sent if w] for sent in prepro_src]
prepro2_dest = [[w for w in sent if w not in stopwords] for sent in prepro_dest]
prepro2_dest = [[wordnet_lemmatizer.lemmatize(w) for w in sent] for sent in prepro2_dest]
prepro2_src = [[w for w in sent if w not in stopwords] for sent in prepro_src]
prepro2_src = [[wordnet_lemmatizer.lemmatize(w) for w in sent] for sent in prepro2_src]


all_words = {}
easyNum = 10000 # top 10000 words

def preprocess_freqword(sent):
  for word in sent:
    all_words[word] = 0

for sent in prepro2_dest:
  preprocess_freqword(sent)
for sent in prepro2_src:
  # print(sent)
  preprocess_freqword(sent)
for word in word_freq[1:easyNum]:
  all_words[word] = 1


def count_freqword(sent):
  res = 0
  for word in sent:
    if all_words[word] == 1:
      res = res + 1
  return res

def make_features(src, dest):
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
    if w in stopwords:
      stopCount_src = stopCount_src+1
  for w in temp_dest:
    if w in stopwords:
      stopCount_dest = stopCount_dest+1

  # remove stopwords and lemmatize
  temp_src = [w for w in temp_src if w not in stopwords]
  temp_dest = [w for w in temp_dest if w not in stopwords]
  temp_src = [wordnet_lemmatizer.lemmatize(w) for w in temp_src]
  temp_dest = [wordnet_lemmatizer.lemmatize(w) for w in temp_dest]

  # (4) len
  lenCount_src = len(temp_src)
  lenCount_dest = len(temp_dest)

  # (5) Count easywords
  easywordCount_src = count_freqword(temp_src)
  easywordCount_dest = count_freqword(temp_dest)
  easywordRatio_src = (easywordCount_src)/(lenCount_src+1)
  easywordRatio_dest = (easywordCount_dest)/(lenCount_dest+1)

  # make all ratio
  dotRatio = dotCount_dest/dotCount_src
  restRatio = (restCount_dest+1)/(restCount_src+1)
  stopRatio = (stopCount_dest+1)/(stopCount_src+1)
  lenRatio = (lenCount_src+1)/(lenCount_dest+1)
  easyRatio = (easywordRatio_dest+0.1)/(easywordRatio_src+0.1)

  return [dotCount_src, restCount_src, stopCount_src, lenCount_src, easywordRatio_src, dotRatio, restRatio, stopRatio, lenRatio, easyRatio]

with open(write_to, "w", encoding = 'utf-8', newline='') as writecsv:
  for i in range(10):
    with open(src_file, encoding = 'utf-8') as oris:
        with open(dest_file+str(i), encoding = 'utf-8') as orid:
            writer = csv.writer(writecsv)
            oris = oris.readlines()
            orid = orid.readlines()
            for s, d in zip(oris, orid):
              feat = make_features(s, d)
              feat.append(s.rstrip())
              feat.append(d.rstrip())
              writer.writerow(feat)

# with open("./wikismall/filtered_data_val_new.csv", encoding = 'utf-8') as word_file:
#     reader = csv.reader(word_file)
#     for l in reader:
#       print(l[-1])
#       break
    # print(sum(1 for row in reader))

# with open("./wikismall/filtered_data_new.csv", "w", encoding = 'utf-8', newline='') as writecsv:
#   with open("./wikismall/filtered_data.csv", "r", encoding = 'utf-8', newline='') as readcsv:
#     writer = csv.writer(writecsv)
#     reader = csv.reader(readcsv)
#     for l in reader:
#       tmp = l[:-2]
#       tmp.append(l[-2].rstrip())
#       tmp.append(l[-1].rstrip())
      # writer.writerow(tmp)

# import torch

# print(torch.cuda.memory_summary())

# with open("./wikismall/PWKP_108016.tag.80.aner.ori.test.src", "r", encoding = 'utf-8') as oris:
#     with open("./wikismall/PWKP_108016.tag.80.aner.test.src", encoding = 'utf-8') as convs:
#         with open("./wikismall/report.txt", encoding = 'utf-8') as rep:
#           with open("./wikismall/report_new.txt", "w", encoding = 'utf-8') as finrep:
#             oris = oris.readlines()
#             convs = convs.readlines()
#             rep = rep.readlines()
#             for (o, c, r) in zip(oris, convs, rep):
#               olines = o.split()
#               clines = c.split()
#               info = {}
#               for (ol, cl) in zip(olines, clines):
#                 if '@' in cl:
#                   info[cl] = ol
#               for (c, o) in info.items():
#                 r = r.replace(c, o)
#               r = r.replace('@', '')
#               finrep.write(r)
