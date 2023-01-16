import argparse, math, random, pickle
from collections import defaultdict


class CRF_POS_Tagger:
    def __init__(self):
        self.weights = None
        self.tag_set = None

    @staticmethod
    def read_data(path):
        """
        Read a dataset path (.txt) and return a list of samples
        in the form (word sequence, tag sequence) and a set of POS-tags.
        """
        samples = []
        tagset = set()
        with open(path, "r", encoding="utf-8") as file:
            words, tags = [], []
            for line in file:
                l = line.split()
                if l:
                    words.append(l[0])
                    tags.append(l[1])
                    tagset.add(l[1])
                else:  # end of sentence
                    samples.append((words, tags))
                    words, tags = [], []
        return samples, tagset

    @staticmethod
    def suffix(word, n):
        """Takes a word and the maximal length of suffix.
        Append 'U' to the word if the word is uppercase, and 'L' if lowercase.
        Returns the last n chars of the word"""

        suff = ''
        case_marker = 'U' if word[0].isupper() else 'L'
        length = len(word)
        if length < n:
            filler = '_' * (n - length)
            suff += filler + word
        else:
            suff = word[-n:]
        return suff + case_marker

    def features(self, ptag, tag, words, i):
        """Takes previous tag, current tag, word sequence and position.
        returns a list of features. Lexical features are tw and ts. tt is a context feature"""

        return [('tw', tag, words[i]),
                ('ts', tag, self.suffix(words[i], 4)),
                ('ts', tag, self.suffix(words[i], 3)),
                ('tt', ptag, tag)]

    @staticmethod
    def logsumexp(values):
        """
        Compute logsumexp operation of the given list of values.
        """
        max_value = max(values)
        result = max_value + math.log(sum(math.exp(value - max_value) for value in values))
        return result

    def score(self, ptag, tag, words, i):
        score = sum(self.weights[feat] for feat in self.features(ptag, tag, words, i))
        return score

    def compute_log_forward(self, words):

        forward = [defaultdict(float) for _ in words]
        forward[0]["<s>"] = 1.0
        for i in range(1, len(words)):
            tags = {"</s>"} if i == len(words) - 1 else self.tag_set
            for t in tags:
                logsumexp_args = [forward[i - 1][prev_t] + self.score(prev_t, t, words, i) for prev_t in
                                  forward[i - 1]]
                forward[i][t] = self.logsumexp(logsumexp_args)
        return forward

    def compute_log_backward(self, words):

        backward = [defaultdict(float) for _ in words]
        backward[-1]["</s>"] = 0.0
        for i in range(len(words) - 1, 0, -1):
            tags = {"<s>"} if i == 1 else self.tag_set
            for t in tags:
                logsumexp_args = [backward[i][next_t] + self.score(t, next_t, words, i) for next_t in backward[i]]
                backward[i - 1][t] = self.logsumexp(logsumexp_args)
        return backward

    def train(self, train_path, num_epoch=1, lr=0.001):
        train_samples, self.tag_set = self.read_data(train_path)
        self.weights = defaultdict(float)

        for epoch in range(num_epoch):
            random.shuffle(train_samples)
            for words, tags in train_samples:
                expected = defaultdict(float)
                observed = defaultdict(float)

                words = [" "] + words + [" "]
                tags = ["<s>"] + tags + ["</s>"]

                log_forward = self.compute_log_forward(words)
                log_backward = self.compute_log_backward(words)
                for i in range(1, len(log_forward)):
                    for tag in log_forward[i]:
                        for prev_tag, prev_prob in log_forward[i - 1].items():
                            p = math.exp(prev_prob +
                                         self.score(prev_tag, tag, words, i) +
                                         log_backward[i][tag] -
                                         log_forward[-1]["</s>"])

                            for f in self.features(prev_tag, tag, words, i):
                                expected[f] += p

                    for f in self.features(tags[i - 1], tags[i], words, i):
                        observed[f] += 1

                for f in expected:
                    self.weights[f] += lr * (observed[f] - expected[f])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', type=str, help="Path of the training data in txt.")
    parser.add_argument('param_file', type=str, help="File name under which the trained weights should be saved.")
    args = parser.parse_args()

    tagger = CRF_POS_Tagger()
    tagger.train(train_path=args.train_path, num_epoch=5, lr=0.01)

    with open(args.param_file, 'wb') as f:
        pickle.dump(tagger.weights, f)
