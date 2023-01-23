import torch
from torch import nn
from itertools import combinations
import torch.nn.functional as F


class Parser(nn.Module):
    def __init__(self, seq_len,
                 vocab_size,
                 embedding_dim,
                 char_lstm_hidden,
                 char_lstm_num_layers,
                 word_lstm_hidden,
                 word_lstm_num_layers,
                 num_classes,
                 linear_hidden):
        super(Parser, self).__init__()

        self.char_lstm_hidden = char_lstm_hidden
        self.word_lstm_hidden = word_lstm_hidden
        self.seq_len = seq_len
        self.embeddings_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.rnn_prefix = nn.LSTM(input_size=embedding_dim, hidden_size=char_lstm_hidden,
                                  num_layers=char_lstm_num_layers, batch_first=True)
        self.rnn_suffix = nn.LSTM(input_size=embedding_dim, hidden_size=char_lstm_hidden,
                                  num_layers=char_lstm_num_layers, batch_first=True)
        self.word_lstm = nn.LSTM(input_size=char_lstm_hidden * 2, hidden_size=word_lstm_hidden,
                                 num_layers=word_lstm_num_layers, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(in_features=word_lstm_hidden * 2, out_features=linear_hidden)
        self.linear2 = nn.Linear(in_features=linear_hidden, out_features=num_classes)

    def forward(self, prefix, suffix):
        prefix = self.embeddings(prefix)
        prefix = self.dropout(prefix)

        suffix = self.embeddings(suffix)
        suffix = self.dropout(suffix)

        # For each word, pass its suffix and prefix into a separate lstm
        # then concatenate the outputs, getting a sentence tensor (num_words, suffix + prefix)
        # then put a zero vector at the sentence beginning and ending.

        # prefix = (batch/num words, suffix_len, hidden)
        # h_prefix = (num_layer*num_direction, num words, hidden)
        _, (h_prefix, _) = self.rnn_prefix(prefix)
        _, (h_suffix, _) = self.rnn_suffix(suffix)

        # concatenate the last hidden states (of the last layer) of prefix and suffix
        # prefix_and_suffix = (num words, suffix+prefix)
        prefix_and_suffix = torch.cat((h_prefix[-1, :, :], h_suffix[-1, :, :]), 1)

        # pad(left, right, top, bottom)
        # sentence_tensor = ( 1+ num words +1, suffix+prefix)
        sentence_tensor = F.pad(prefix_and_suffix, pad=(0, 0, 1, 1), mode="constant", value=0.0)
        sentence_tensor = self.dropout(sentence_tensor)

        # out = (1, N+2, H2*2)
        out, _ = self.word_lstm(sentence_tensor.unsqueeze(dim=0))

        # split the output tensor into forward and backward hidden states.
        # for forward, remove the last word (zero padding).
        forward_hidden_states = out[-1, :-1, :self.word_lstm_hidden]  # (N,H2)

        # for backward, remove the first word (zero padding).
        backward_hidden_states = out[-1, 1:, self.word_lstm_hidden:]  # (N,H2)

        # compute all spans, and put them into a tensor
        spans = list(combinations(range(self.seq_len + 1), 2))
        # span_tensors = (num_spans, 2*H2)
        span_tensors = torch.stack([torch.cat((forward_hidden_states[end] - forward_hidden_states[start],
                                               backward_hidden_states[start] - backward_hidden_states[end]))
                                    for start, end in spans])
        span_tensors = self.dropout(span_tensors)

        out = self.linear1(span_tensors)  # out = (num span, hidden)
        out = self.dropout(out)
        out = self.relu(out)
        scores = self.linear2(out)  # score = (num span, num classes)

        # Die Bewertung des Labels \keine Konstituente" (mit Label-ID 0) sollte auf 0 gesetzt werden
        scores[:, 0].zero_()

        return scores


if __name__ == "__main__":
    vocab_size = 25
    sentence_length = 10
    prefix_length = 6
    num_classes = 15

    # tensor of prefix (character IDs)
    prefix = torch.randint(low=0, high=vocab_size, size=(sentence_length, prefix_length))

    # tensor of suffix (character IDs)
    # assume suffix is already reversed.
    suffix = torch.randint(low=0, high=vocab_size, size=(sentence_length, prefix_length))

    net = Parser(seq_len=sentence_length,
                 vocab_size=vocab_size,
                 embedding_dim=4,
                 char_lstm_hidden=8,
                 char_lstm_num_layers=3,
                 word_lstm_hidden=12,
                 word_lstm_num_layers=3,
                 num_classes=num_classes,
                 linear_hidden=24)
    out = net(prefix, suffix)

    # check if the output tensor is correct
    print(out.shape)
    assert out.shape[0] == (sentence_length * (1 + sentence_length)) / 2
    assert out.shape[1] == num_classes
    # score for label-id 0 should be 0
    assert torch.all(out[:, 0] == 0)
