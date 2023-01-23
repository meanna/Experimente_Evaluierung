import torch
from torch import nn
from itertools import combinations


class Parser(nn.Module):
    def __init__(self, seq_len,
                 vocab_size,
                 embedding_dim,
                 char_lstm_hidden,
                 char_lstm_num_layers,
                 word_lstm_hidden,
                 word_lstm_num_layers,
                 num_classes):
        super(Parser, self).__init__()

        self.char_lstm_hidden = char_lstm_hidden
        self.word_lstm_hidden = word_lstm_hidden
        self.seq_len = seq_len
        self.embeddings_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn_prefix = nn.LSTM(input_size=embedding_dim, hidden_size=char_lstm_hidden,
                                  num_layers=char_lstm_num_layers, batch_first=True)
        self.rnn_suffix = nn.LSTM(input_size=embedding_dim, hidden_size=char_lstm_hidden,
                                  num_layers=char_lstm_num_layers, batch_first=True)
        self.word_lstm = nn.LSTM(input_size=char_lstm_hidden * 2, hidden_size=word_lstm_hidden,
                                 num_layers=word_lstm_num_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(in_features=word_lstm_hidden * 2, out_features=num_classes)

    def forward(self, prefix, suffix):
        prefix = self.embeddings(prefix)
        suffix = self.embeddings(suffix)

        # create the tensor (N+2, 2*H1)
        sentence = []

        # for each word, pass its suffix and prefix into a separate lstm
        # and concatenate the outputs.
        # put zero vectors at the sentence beginning and ending of the sentence tensor.
        zero = torch.zeros((2 * self.char_lstm_hidden))
        sentence.append(zero)
        for i in range(len(prefix)):
            _, (h_prefix, _) = self.rnn_prefix(prefix[i].unsqueeze(0))
            _, (h_suffix, _) = self.rnn_suffix(suffix[i].unsqueeze(0))
            prefix_and_suffix = torch.cat((h_prefix.squeeze(), h_suffix.squeeze()), 0)
            sentence.append(prefix_and_suffix)
        sentence.append(zero)
        sentence_tensor = torch.stack(sentence)

        # out dim = (batch, N+2, H2*2)
        out, _ = self.word_lstm(sentence_tensor.unsqueeze(dim=0))

        # remove the last word
        forward_hidden_states = out[:, :-1, :self.word_lstm_hidden].squeeze()  # (N,H2)
        # remove the first word
        backward_hidden_states = out[:, 1:, self.word_lstm_hidden:].squeeze()  # (N,H2)

        # compute all spans, and put them into a tensor
        spans = list(combinations(range(self.seq_len + 1), 2))
        span_list = []
        for start, end in spans:
            span = torch.cat((forward_hidden_states[end] - forward_hidden_states[start],
                              backward_hidden_states[start] - backward_hidden_states[end]))
            span_list.append(span)

        span_tensors = torch.stack(span_list)  # (num_spans, 2*H2)
        scores = self.linear(span_tensors)  # (num span, num class)

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
                 char_lstm_num_layers=1,
                 word_lstm_hidden=12,
                 word_lstm_num_layers=1,
                 num_classes=num_classes)
    out = net(prefix, suffix)
    print(out.shape)
