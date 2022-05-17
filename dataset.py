import os, io
from torch.utils.data import Dataset
import csv
import itertools

class WikiDatasetRaw(Dataset):
    def __init__(self, src_file, dst_file):
        with io.open(src_file, encoding = 'utf-8') as sf:
          self.src_file = sf.readlines()
        with io.open(dst_file, encoding = 'utf-8') as df:
          self.dst_file = df.readlines()

    def __len__(self):
        return len(self.dst_file)

    def __getitem__(self, idx):
        return (self.make_features(self.src_file[idx], self.dst_file[idx]), self.src_file[idx], self.dst_file[idx])

class WikiDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.f = open(self.file_path, encoding = 'utf-8')
        reader = csv.reader(self.f)
        self.data = list(reader)

    def __len__(self):
        return len(self.data)

    def __del__(self):
        return self.f.close() 
    def __getitem__(self, idx):
        return ([float(i) for i in self.data[idx][:-2]], self.data[idx][-2], self.data[idx][-1])