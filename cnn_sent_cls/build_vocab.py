import os
import jieba
import json

class Vocab(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        # add <pad> and <unk> token to vocab
        self.add_word('<pad>')
        self.add_word('<unk>')
        self.pad_idx = self.word2idx['<pad>']
        self.unk_idx = self.word2idx['<unk>']
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def update(self, tokens):
        for token in tokens:
            self.add_word(token)

    def __len__(self):
        return len(self.word2idx)
    
    def save_vocab(self, vocab_file):
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=4)
        print('vocab saved to {}'.format(vocab_file))
    
    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.word2idx = json.load(f)
        print('vocab loaded from {}'.format(vocab_file))
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.idx = len(self.word2idx)
        self.pad_idx = self.word2idx['<pad>']
        self.unk_idx = self.word2idx['<unk>']
    
    def sent2idx(self, sent) -> list:
        '''
        sent: list[String]

        return: list[int]
        '''
        return [self.word2idx.get(token, self.unk_idx) for token in sent]
        

def build_vocab():

    vocab = Vocab()
    trian_label = list()
    test_label = list()
    dev_label = list()
    train_sentence = list()
    test_sentence = list()
    dev_sentence = list()

    train_file = os.path.join(os.path.dirname(__file__), 'train.txt')
    test_file = os.path.join(os.path.dirname(__file__), 'test.txt')
    dev_file = os.path.join(os.path.dirname(__file__), 'dev.txt')

    # Read each line in the train dataset
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '')
            sentence, label = line.split('\t')
            label = int(label)
            tokens = jieba.cut(sentence)
            # transform tokens to list[String]
            tokens = list(tokens)
            vocab.update(tokens)
            # add tokenized sentence and label to list
            train_sentence.append(tokens)
            trian_label.append(label)

    # Read each line in the test dataset
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '')
            sentence, label = line.split('\t')
            label = int(label)
            tokens = jieba.cut(sentence)
            tokens = list(tokens)
            vocab.update(tokens)
            test_sentence.append(tokens)
            test_label.append(label)

    # Read each line in the dev dataset
    with open(dev_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '')
            sentence, label = line.split('\t')
            label = int(label)
            tokens = jieba.cut(sentence)
            tokens = list(tokens)
            vocab.update(tokens)
            dev_sentence.append(tokens)
            dev_label.append(label)

    # save sentences and labels to json file
    train_data = {'sentence': train_sentence, 'label': trian_label}
    test_data = {'sentence': test_sentence, 'label': test_label}
    dev_data = {'sentence': dev_sentence, 'label': dev_label}

    train_data_file = os.path.join(os.path.dirname(__file__), 'train_data.json')
    test_data_file = os.path.join(os.path.dirname(__file__), 'test_data.json')
    dev_data_file = os.path.join(os.path.dirname(__file__), 'dev_data.json')

    with open(train_data_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open(test_data_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    with open(dev_data_file, 'w', encoding='utf-8') as f:
        json.dump(dev_data, f, ensure_ascii=False, indent=4)

    # save vocab to json file
    vocab_file = os.path.join(os.path.dirname(__file__), 'vocab.json')
    vocab.save_vocab(vocab_file)


if __name__ == '__main__':
    build_vocab()
