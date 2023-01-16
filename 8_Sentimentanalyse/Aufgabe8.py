'''
TODO: trainieren Sie damit einen Klassifizierer auf Basis von LSTMs.

Zur effizienteren Verarbeitung im LSTM transformieren Sie den gepaddeten Einabetensor mit pack padded sequence und
transformieren den Ausgabetensor des LSTMs mit pad packed sequence.

Im Training minimieren Sie die negative Loglikelihood mit dem CrossEntropyLoss von PyTorch. Nach jeder Epoche
evaluieren Sie das System auf den Entwicklungsdaten und geben die erzielte Genauigkeit aus.
Verwenden Sie keine vortrainierten Embeddings.

optional: GPU verwenden
'''

import torch
import torch.nn as nn
import torchtext
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def read_data(path):
    '''
    Trainingsdaten (oder Developmentdaten) aus einer Datei einlesen
    und eine Liste vom Paaren zurückgibt, wobei jedes Paar aus einem Klassen-Label (als Integer) und einer Liste von
    Tokens besteht.
    :return: eine Liste vom Paaren
    '''
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            l = line.split()
            label = int(l[0])
            tokens = l[1:]
            data.append((label, tokens))
    return data


def collate(batchdata):
    '''
    TODO: welche ein Batch von Datenpaaren als Argument erhält -> Liste von ein paar Datenpaaren
    und die Wortfolgen mit Hilfe von vocab auf Zahlen abbildet
    und drei Tensoren mit den Labels, den Wort-IDs und den Textlängen zurückgibt.
    Die Wort-IDs werden mit der PyTorch-Methode pad_sequence “gepaddet”.

    >>> v1 = vocab(ordered_dict)
    >>> print(v1['a']) #prints 1
    :return:
    '''
    pass


'''
TODO:Erzeugen Sie nun PyTorch-Objekte des Typs DataLoader für die Trainingsdaten und die Development-Daten, welche 
die Funktion collate verwenden, um ein Batch von Beispielen zusammenzufassen.
'''


def train():
    '''
    welche eine Trainingsepoche durchführt,
    und eine Funktion evaluate, welche die Genauigkeit auf den Developmentdaten berechnet.
    Die Funktionen erhalten jeweils einen DataLoader als Argument.
    :return:
    '''
    pass


class TextClassifier:
    '''
    TODO: welche das neuronale Netz implementiert 结构见板书
    '''

    def __init__(self):
        pass


if __name__ == "__main__":
    # Hauptprogramm
    '''
    TODO: Erzeugen Sie unter Verwendung der TorchText-Bibliothek mit den Befehlen ein Vokabular
    '''
    data = read_data('data/sentiment.train.tsv')
    texts = []
    for _, tokens in data:
        texts += tokens
    # wobei texts die Liste aller Wortfolgen in den Trainingsdaten ist.
    vocab = build_vocab_from_iterator(texts, specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])  # make default index same as index of unk_token
