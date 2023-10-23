# nlpdl-hw2
Code implementation of nlpdl hw2 2023

# Installation

```
conda install --yes --file requirements.txt
```

It's recommended to install python=3.8 and torch=2.0.1.

You can install PyTorch by following the instructions provided on the [PyTorch website](https://pytorch.org/get-started/locally/).

# Task2

For Task 2, you need to place the train.txt, test.txt, and dev.txt files under the `cnn_sent_cls/` directory.

To preprocess the dataset and build the vocabulary using Jieba, you can run the following command:

```
python cnn_sent_cls/build_vocab.py
```

For training, you can run the main script with the `-t` flag:

```
python cnn_sent_cls/main.py -t
```

There are additional options you can provide, such as setting the random seed, learning rate, batch size, number of epochs, dropout rate, embedding dimension, number of filters, filter sizes, and output dimension. These options can be specified when running the script.

The checkpoints generated during training will be saved in the `cnn_sent_cls/checkpoints/` directory.

For testing, you can run the main script and specify the path to the checkpoint file using the `--load_path` option:

```
python cnn_sent_cls/main.py --load_path=epoch_{num}
```
Replace {num} with the checkpoint number you want to load for testing.
