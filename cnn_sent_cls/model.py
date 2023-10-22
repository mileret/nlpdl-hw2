import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, n_filters, embedding_dim, filter_sizes, output_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # take the index of the word and return the embedding
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

        self.bn = nn.BatchNorm1d(embedding_dim)
        
    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text) # [batch size, sent len, emb dim]

        # add batch normalization
        embedded = embedded.permute(0, 2, 1)
        embedded = self.bn(embedded)
        embedded = embedded.permute(0, 2, 1)

        embedded = embedded.unsqueeze(1) # [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))
        
        return self.softmax(self.fc(cat))

    def cls_loss(self, pred, label):
        '''
        pred: [batch_size, n_classes]
        label: [batch_size]
        '''
        return F.cross_entropy(pred, label)
    
    def cls_acc(self, pred, label):
        '''
        pred: [batch_size, n_classes]
        label: [batch_size]
        '''
        pred = torch.argmax(pred, dim=1)
        return (pred == label).float().mean()
    
    def cls_pred(self, pred):
        '''
        pred: [batch_size, n_classes]
        '''
        return torch.argmax(pred, dim=1)
    