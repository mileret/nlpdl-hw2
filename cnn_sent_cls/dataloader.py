import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class MyDataset(Dataset):
    def __init__(self, sents, labels):
        '''
        sents: list[list[int]]
        labels: list[int]
        '''
        # to tensor
        self.sents = [torch.tensor(sent) for sent in sents]
        self.labels = torch.tensor(labels)

    def __getitem__(self, index):
        return self.sents[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

# use collate_fn to pad sentence
def collate_fn(data):
    '''
    data: list[(sent, label)]
    '''
    # sort by sentence length
    data.sort(key=lambda x: len(x[0]), reverse=True)
    sents, labels = zip(*data)
    # pad sentence and the minimum length is 5 (because max conv kernel size 5)
    sents = [torch.cat((sent, torch.zeros(5 - len(sent), dtype=torch.long))) if len(sent) < 5 else sent for sent in sents]
    sents = nn.utils.rnn.pad_sequence(sents, batch_first=True, padding_value=0)
    return sents, torch.tensor(labels)
