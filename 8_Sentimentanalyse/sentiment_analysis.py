import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def read_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            label, *tokens = line.split()
            data.append((label, tokens))
    return data


def collate(batch):
    '''
    A custom collate function. Return 3 tensors: word-IDs, labels, text length
    '''
    inputs = []
    labels = []
    text_length = []
    for label, text in batch:
        inputs.append(torch.tensor(text))
        labels.append(int(label))
        text_length.append(len(text))

    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, torch.tensor(labels), torch.tensor(text_length)


def train(data):
    model.train()
    correct = 0
    total = 0
    for inputs, labels, input_length in data:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        # zero accumulated gradients
        model.zero_grad()

        # get the output from the model
        output = model(inputs, input_length)

        output_prob = output.softmax(dim=2)
        best_class = torch.argmax(output_prob, dim=2)
        predicted = best_class.squeeze().tolist()
        for i in range(len(predicted)):
            if predicted[i] == labels[i]:
                correct += 1
            total += 1

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(1), labels)
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return correct / total


def evaluate(data):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, input_length in data:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            output = model(inputs, input_length)

            output_prob = output.softmax(dim=2)
            best_class = torch.argmax(output_prob, dim=2)
            predicted = best_class.squeeze().tolist()
            for i in range(len(predicted)):
                if predicted[i] == labels[i]:
                    correct += 1
                total += 1

    return correct / total


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, hidden_dim):
        super(Net, self).__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, dropout=0.1, num_layers=2)
        self.linear = nn.Linear(hidden_dim * 2, num_classes)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x, input_length):
        embeds = self.embedding_layer(x)  # batch, input, embed
        embeds = self.drop(embeds)
        embeds = pack_padded_sequence(embeds, input_length, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(embeds)  # batch, input length, hidden*2

        lstm_out, out_len = pad_packed_sequence(lstm_out, batch_first=True)

        max_pool = lstm_out.max(dim=1)[0].unsqueeze(1)  # batch, 1, hidden*2

        # test max + mean pool (not better than only max pool)
        # mean_pool = lstm_out.mean(dim=1).unsqueeze(1) # batch, 1, hidden*2
        # max_mean_pool = torch.cat( [max_pool, mean_pool], dim=2) # batch, 1, hidden*2 *2]

        out = self.linear(max_pool)
        return out


if __name__ == "__main__":

    batch_size = 256
    embed_dim = 128
    hidden_dim = 128

    # Read train set and compute vocab
    train_data = read_data("data/sentiment.train.tsv")
    train_texts = []
    label_set = set()
    for label, token in train_data:
        train_texts.append(token)
        label_set.add(label)

    vocab = build_vocab_from_iterator(train_texts, specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])  # make default index same as index of unk_token
    vocab_size = len(vocab.get_itos())

    # create a data loader for train
    label_and_token_ids_train = [(label, vocab.lookup_indices(text)) for label, text in train_data]
    train_loader = DataLoader(label_and_token_ids_train, batch_size=batch_size, shuffle=True, collate_fn=collate)

    # create a data loader for dev
    dev_data = read_data("data/sentiment.dev.tsv")
    label_and_token_ids_dev = [(label, vocab.lookup_indices(text)) for label, text in dev_data]
    dev_loader = DataLoader(label_and_token_ids_dev, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # create a data loader for test
    test_data = read_data("data/sentiment.test.tsv")
    label_and_token_ids_test = [(label, vocab.lookup_indices(text)) for label, text in test_data]
    test_loader = DataLoader(label_and_token_ids_test, batch_size=batch_size, shuffle=False, collate_fn=collate)

    print("classes = ", label_set)
    model = Net(vocab_size=vocab_size, embedding_dim=embed_dim, num_classes=len(list(label_set)),
                hidden_dim=hidden_dim).to(DEVICE)

    # Training parameters
    epochs = 20
    lr = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    clip = 5  # gradient clipping

    for e in range(epochs):
        print("Epoch = ", e + 1)
        train_acc = train(train_loader)
        print("Train accuracy = ", train_acc)
        torch.save(model.state_dict(), "model.pt")

        dev_acc = evaluate(dev_loader)
        print("Dev accuracy   = ", dev_acc)

    test_acc = evaluate(test_loader)
    print("Test accuracy = ", test_acc)
