import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataloader import MyDataset, collate_fn
from model import CNN
from build_vocab import Vocab



def train(args):
    
    # load vocab
    vocab = Vocab()
    vocab_file = os.path.join(os.path.dirname(__file__), 'vocab.json')
    vocab.load_vocab(vocab_file)

    # load dataset
    train_data_file = os.path.join(os.path.dirname(__file__), 'train_data.json')
    with open(train_data_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    train_sentence = train_data['sentence']
    train_label = train_data['label']

    # load dev dataset
    dev_data_file = os.path.join(os.path.dirname(__file__), 'dev_data.json')
    with open(dev_data_file, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    dev_sentence = dev_data['sentence']
    dev_label = dev_data['label']

    # transform sentence to index
    train_sentence_idx = list()
    for sent in train_sentence:
        sent_idx = vocab.sent2idx(sent)
        train_sentence_idx.append(sent_idx)

    train_dataset = MyDataset(train_sentence_idx, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    dev_sentence_idx = list()
    for sent in dev_sentence:
        sent_idx = vocab.sent2idx(sent)
        dev_sentence_idx.append(sent_idx)
    
    dev_dataset = MyDataset(dev_sentence_idx, dev_label)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # build model
    model = CNN(vocab_size=len(vocab), n_filters=args.n_filters, embedding_dim=args.embedding_dim, filter_sizes=args.filter_sizes, output_dim=args.output_dim, dropout=args.dropout)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    val_loss = []
    # tensorboard
    log_dir = os.path.join(os.path.dirname(__file__), 'runs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(args.epochs):

        # train
        model.train()
        for i, (sents, labels) in tqdm(enumerate(train_dataloader), desc=f'Training Epoch {epoch}'):
            optimizer.zero_grad()
            pred = model(sents)
            loss = model.cls_loss(pred, labels)
            acc = model.cls_acc(pred, labels)
            loss.backward()
            optimizer.step()
            print('Epoch: {}, Iter: {}, Loss: {:.4f}, Acc: {:.4f}'.format(epoch, i, loss.item(), acc.item()))
            writer.add_scalar('train_loss', loss.item(), epoch * len(train_dataloader) + i)
            writer.add_scalar('train_acc', acc.item(), epoch * len(train_dataloader) + i)

        # validation
        model.eval()
        total_loss = 0
        for i, (sents, labels) in tqdm(enumerate(dev_dataloader), desc=f'Validation Epoch {epoch}'):
            pred = model(sents)
            loss = model.cls_loss(pred, labels)
            acc = model.cls_acc(pred, labels)
            total_loss += loss.item()
            print('Epoch: {}, Iter: {}, Loss: {:.4f}, Acc: {:.4f}'.format(epoch, i, loss.item(), acc.item()))
            writer.add_scalar('val_loss', loss.item(), epoch * len(dev_dataloader) + i)
            writer.add_scalar('val_acc', acc.item(), epoch * len(dev_dataloader) + i)
        total_loss /= len(dev_dataloader)
        val_loss.append(total_loss)
        
        # apply early stopping, if the validation loss is not decreasing for 5 epochs, stop training
        if len(val_loss) > 5 and val_loss[-1] > val_loss[-2] > val_loss[-3] > val_loss[-4] > val_loss[-5]:
            print('Early Stopping')
            print('Best Epoch: {}'.format(epoch - 5))
            break

        # save checkpoints
        save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'epoch_{epoch}.pth')
        torch.save(model.state_dict(), save_path)


def test(args):
    
    # load vocab
    vocab = Vocab()
    vocab_file = os.path.join(os.path.dirname(__file__), 'vocab.json')
    vocab.load_vocab(vocab_file)

    # load test dataset
    test_data_file = os.path.join(os.path.dirname(__file__), 'test_data.json')
    with open(test_data_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    test_sentence = test_data['sentence']
    test_label = test_data['label']

    # transform sentence to index
    test_sentence_idx = list()
    for sent in test_sentence:
        sent_idx = vocab.sent2idx(sent)
        test_sentence_idx.append(sent_idx)

    test_dataset = MyDataset(test_sentence_idx, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # build model
    model = CNN(vocab_size=len(vocab), n_filters=args.n_filters, embedding_dim=args.embedding_dim, filter_sizes=args.filter_sizes, output_dim=args.output_dim, dropout=args.dropout)
    
    # load checkpoints
    load_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    load_path = os.path.join(load_dir, args.load_path)
    model.load_state_dict(torch.load(load_path))

    # test
    model.eval()
    total_acc = 0
    for i, (sents, labels) in tqdm(enumerate(test_dataloader), desc=f'Test'):
        pred = model(sents)
        loss = model.cls_loss(pred, labels)
        acc = model.cls_acc(pred, labels)
        total_acc += acc.item()
        print('Iter: {}, Loss: {:.4f}, Acc: {:.4f}'.format(i, loss.item(), acc.item()))
    print('Test Acc: {:.4f}'.format(total_acc / len(test_dataloader)))

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(seed)
    # random.seed(seed)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # train args
    parser.add_argument('--seed', type=int, default=510)
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--n_filters', type=int, default=100)
    parser.add_argument('--filter_sizes', type=list, default=[3, 4, 5])
    parser.add_argument('--output_dim', type=int, default=4)

    # test args
    parser.add_argument('--load_path', type=str, default='epoch_29.pth')
    args = parser.parse_args()
    
    seed_everything(args.seed)

    if args.train:
        train(args)
    else:
        test(args)
