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

def make_features(src, dest):
      # remove '\n' and split by ' '
      temp_src = src.strip('\n').split(' ')
      temp_dest = dest.strip('\n').split(' ')

      # (1) count dot and (2) rest
      dotCount_src = 0;  dotCount_dest = 0
      restCount_src = 0;  restCount_dest = 0
      for w in src:
        if w in ['.', '..', '...']:
          dotCount_src = dotCount_src+1
        if w == ',':
          restCount_src = restCount_src+1
      for w in dest:
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
      temp_src = [w for w in temp_src if w]
      temp_dest = [w.strip("."",") for w in temp_dest]
      temp_dest = [w for w in temp_dest if w]

      # (3) count stopwords
      stopCount_src = 0;  stopCount_dest = 0
      for w in src:
        if w in stopwords:
          stopCount_src = stopCount_src+1
      for w in dest:
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
      easyNum = 10000
      easywordCount_src = 0;  easywordCount_dest = 0
      for w in temp_src:
        if w in word_freq[1:easyNum]:
          easywordCount_src = easywordCount_src+1
      for w in temp_dest:
        if w in word_freq[1:easyNum]:
          easywordCount_dest = easywordCount_dest+1
      easywordRatio_src = (easywordCount_src)/(lenCount_src+1)
      easywordRatio_dest = (easywordCount_dest)/(lenCount_dest+1)

      # make all ratio
      dotRatio = dotCount_dest/dotCount_src
      restRatio = (restCount_dest+1)/(restCount_src+1)
      stopRatio = (stopCount_dest+1)/(stopCount_src+1)
      lenRatio = (lenCount_src+1)/(lenCount_dest+1)
      easyRatio = (easywordRatio_dest+0.1)/(easywordRatio_src+0.1)

      return [dotRatio, restRatio, stopRatio, lenRatio, easyRatio]

with open("./wikismall/filtered_data.csv", "w", encoding = 'utf-8', newline='') as writecsv:
    with open("./wikismall/PWKP_108016.tag.new.aner.train.src", encoding = 'utf-8') as oris:
        with open("./wikismall/PWKP_108016.tag.new.aner.train.dst", encoding = 'utf-8') as orid:
            stopwords = stopwords.words('english')
            wordnet_lemmatizer = WordNetLemmatizer()
            word_path = './unigram_freq.csv'
            word_data = csv.reader(open(word_path, 'r'))
            word_freq = []
            writer = csv.writer(writecsv)
            with open(word_path) as word_file:
                reader = csv.reader(word_file)
                for row in reader:
                    word_freq.append(row[0])
            for s, d in zip(oris, orid):
                feat = make_features(s, d)
                feat.append(s)
                feat.append(d)
                writer.writerow(feat)

# with open("./wikismall/filtered_data_val.csv", encoding = 'utf-8') as word_file:
#     reader = csv.reader(word_file)
#     print(sum(1 for row in reader))

