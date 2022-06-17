'''
@ Contributor: Nayoung-Oh
'''
from torch.utils.data import Dataset
import csv

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
        return ([float(i) for i in self.data[idx][0:5]], [float(i) for i in self.data[idx][5:10]], self.data[idx][-2], self.data[idx][-1])