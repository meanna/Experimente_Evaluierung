# crf-train.py train.txt param-file
# python crf_train.py ../Tiger/train.txt param-file

import argparse, math, random, pickle, time
from collections import defaultdict
from tqdm import tqdm
from itertools import groupby


class CRF_POS_Tagger:
    def __init__(self):
        self.weights = None
        self.tag_set = None
        self.prune_lex_tag_threshold = None
        self.prune_forward_threshold = None
        self.mu = None
        self.lr = None
        self.cache = defaultdict(float)

    @staticmethod
    def read_data(path, num_samples=None):
        """
        Read a dataset path (.txt) and return a list of samples
        in the form (word sequence, tag sequence) and a set of POS-tags.
        If num_samples is given, read only num_samples sentences.
        """
        samples = []
        tagset = set()
        c = 0
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
                    c += 1
                    if num_samples and c >= num_samples:
                        break

        tagset = list(tagset)
        print("Num samples", len(samples))
        print("Num tags", len(tagset))
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

    @staticmethod
    def sign(num):
        if num > 0:
            return 1.0
        elif num < 0:
            return -1.0
        else:
            return 0

    @staticmethod
    def shape(word, n=5):
        """Maps all characters to a character type.
        Upper case to "X", lower case to 'x' (e.g. 'Geräte' -> 'Xxxxxx'),
        and digits to 'd' (e.g., 'ID-s123' -> 'XX-xddd'). All other characters remain the same.
        The number of consecutive characters of the same type is limited to n.
        (e.g. if n = 5, 'Algorithmus' -> 'Xxxxxx').
        """

        def transform(c):
            if c.isalpha():
                return 'X' if c.isupper() else 'x'
            if c.isdigit():
                return 'd'
            return c

        return ''.join(c * min(n, len(list(g))) for c, g in groupby(word, transform))

    def features(self, ptag, tag, word):
        """Takes previous tag, current tag, word sequence and position.
        returns a list of features. Lexical features are tw and ts. tt is a context feature"""
        context_features = [('tt', ptag, tag)]
        return context_features + self.lexical_feature(tag, word)

    def lexical_feature(self, tag, word):
        return [('tw', tag, word),
                ('shape', tag, self.shape(word)),
                ('ts', tag, self.suffix(word, 4)),
                ('ts', tag, self.suffix(word, 3)),
                ]

    @staticmethod
    def logsumexp(values):
        """
        Compute logsumexp operation of the given list of values.
        """
        max_value = max(values)
        result = max_value + math.log(sum(math.exp(value - max_value) for value in values))
        return result

    def best_tags_wrt_lexical_features(self, word, i):

        scores = {}
        tag_set = {"</s>"} if word == " " else self.tag_set
        for tag in tag_set:
            scores[tag] = self.lex_score(tag, word, i)

        return self.prune(scores, self.prune_lex_tag_threshold)

    def prune(self, tag_score_dict, threshold):
        best_tag_to_score = dict()
        t = max(tag_score_dict.values()) + math.log(threshold)
        for tag, score in tag_score_dict.items():
            if score > t:
                best_tag_to_score[tag] = score
            else:
                self.count_prune += 1  # for debugging
        return best_tag_to_score

    def score(self, ptag, tag, words, i):
        """Compute a score for the given arguments based on lexical and context features"""
        arg = (ptag, tag, words[i], i)
        if arg not in self.cache:
            score = sum(self.weights[feat] for feat in self.features(ptag, tag, words[i]))
            self.cache[arg] = score
        return self.cache.get(arg)

    def lex_score(self, tag, word, i):
        """Compute a score for the given arguments based on lexical features"""
        arg = (tag, word, i)
        if arg not in self.cache:
            score = sum(self.weights[feat] for feat in self.lexical_feature(tag, word))
            self.cache[arg] = score
        return self.cache.get(arg)

    def compute_log_forward_prune_L1(self, words):

        forward = [dict() for _ in words]
        forward[0]["<s>"] = 0.0

        for i in range(1, len(words)):
            tag_to_forward = dict()
            # prune tags based on lexical features
            for t in self.best_tags_wrt_lexical_features(words[i], i):
                logsumexp_args = []
                for prev_t in forward[i - 1]:
                    # update L1
                    self.regularize_with_L1(self.features(prev_t, t, words[i]))
                    logsumexp_args.append(forward[i - 1][prev_t] + self.score(prev_t, t, words, i))
                tag_to_forward[t] = self.logsumexp(logsumexp_args)
            # prune forward based on log(forward)
            forward[i] = self.prune(tag_to_forward, self.prune_forward_threshold)

        return forward

    def regularize_with_L1(self, features, remove_zero_weight=False):
        """ Update L1 for the given features (dictionary)"""

        for f in features.copy():
            if remove_zero_weight and self.weights[f] == 0.0:
                del self.weights[f]
            num_steps = self.current_step - self.last_updated[f]
            if num_steps > 0:
                total_decay = self.decay * num_steps
                if abs(total_decay) > abs(self.weights[f]):
                    self.weights[f] = 0.0
                else:
                    # weight[f] -= eta * mu * sign(weight[f])
                    self.weights[f] -= total_decay * self.sign(self.weights[f])
            self.last_updated[f] = self.current_step

    def compute_log_backward_prune_L1(self, words, log_forward):

        backward = [defaultdict(float) for _ in words]
        backward[-1]["</s>"] = 0.0
        for i in range(len(words) - 1, 0, -1):
            for t in log_forward[i - 1]:
                logsumexp_args = [backward[i][next_t] + self.score(t, next_t, words, i) for next_t in backward[i]]
                backward[i - 1][t] = self.logsumexp(logsumexp_args)
        return backward

    def train_L1_lazy(self, train_path,
                      param_file,
                      mu=0.01,
                      num_epoch=1,
                      lr=0.001,
                      num_train_samples=None,
                      dev_path=None,
                      num_dev_samples=None,
                      prune_forward_threshold=0.001,
                      prune_lex_tag_threshold=0.001):
        self.train_samples, self.tag_set = self.read_data(train_path, num_train_samples)
        self.weights = defaultdict(float)
        self.decay = mu * lr
        self.prune_lex_tag_threshold = prune_lex_tag_threshold
        self.prune_forward_threshold = prune_forward_threshold

        print(f"lr {lr}, mu {mu}")
        print("Prune threshold for tags", self.prune_lex_tag_threshold)
        print("Prune threshold for forward", self.prune_forward_threshold)
        print("Train set", train_path)

        self.last_updated = defaultdict(int)
        self.current_step = 0
        best_dev_acc = 0.0

        start_time_total = time.perf_counter()
        for epoch in range(num_epoch):
            self.count_prune = 0  # for debugging
            random.shuffle(self.train_samples)
            start_time = time.perf_counter()
            for words, tags in tqdm(self.train_samples):
                self.current_step += 1

                self.expected = defaultdict(float)
                self.observed = defaultdict(float)

                words = [" "] + words + [" "]
                tags = ["<s>"] + tags + ["</s>"]

                log_forward = self.compute_log_forward_prune_L1(words)
                log_backward = self.compute_log_backward_prune_L1(words, log_forward)
                # compute gradient
                for i in range(1, len(log_forward)):
                    for tag in log_forward[i]:
                        for prev_tag, prev_prob in log_forward[i - 1].items():
                            p = math.exp(prev_prob +
                                         self.score(prev_tag, tag, words, i) +
                                         log_backward[i][tag] -
                                         log_forward[-1]["</s>"])

                            for f in self.features(prev_tag, tag, words[i]):
                                self.expected[f] += p

                    for f in self.features(tags[i - 1], tags[i], words[i]):
                        self.observed[f] += 1

                # update gradient without L1
                for f in self.expected:
                    self.weights[f] += lr * (self.observed[f] - self.expected[f])

                self.cache = defaultdict(float)

            # update L1 for features that have not been updated yet
            self.regularize_with_L1(self.weights, remove_zero_weight=True)

            # compute accuracy on dev set and save model parameters
            print("## Compute Dev Accuracy ##")
            dev_acc = self.compute_accuracy(dev_path, num_dev_samples)
            print("Dev acc", dev_acc)

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                print("Save model as", param_file)
                with open(param_file, 'wb') as f:
                    pickle.dump((self.weights, self.tag_set), f)
            print("Num pruned tags", self.count_prune)

            exec_time = time.perf_counter() - start_time
            print(f"epoch {epoch + 1} takes {exec_time} secs, {exec_time / 60.0} mins")
            print("---" * 20)

        total_time = time.perf_counter() - start_time_total
        print(f"Time total {total_time} secs, {total_time / 60.0} mins")
        print("Best dev accuracy", best_dev_acc)

    def viterbi(self, words):
        """Compute the best tag sequence."""
        vitscore = [dict() for _ in words]
        vitscore[0]["<s>"] = 0.0
        bestprev = [dict() for _ in words]
        for i in range(1, len(words)):
            tags = {"</s>"} if i == len(words) - 1 else self.tag_set
            for tag in tags:
                for prevtag in vitscore[i - 1]:
                    s = vitscore[i - 1][prevtag] + (self.score(prevtag, tag, words, i))
                    if tag not in vitscore[i] or vitscore[i][tag] < s:
                        vitscore[i][tag] = s
                        bestprev[i][tag] = prevtag
        best_tags = []
        tag = '</s>'
        for i in range(len(words) - 1, 1, -1):
            tag = bestprev[i][tag]
            best_tags.append(tag)
        best_tags.reverse()
        return best_tags

    def compute_accuracy(self, test_path, num_test_samples=None):
        self.test_samples, _ = self.read_data(path=test_path, num_samples=num_test_samples)
        print("Compute accuracy for", test_path)
        correct = 0
        total = 0

        for words, tags in self.test_samples:
            words = [" "] + words + [" "]
            best_tags = self.viterbi(words)
            for i in range(len(best_tags)):
                if best_tags[i] == tags[i]:
                    correct += 1
                total += 1
        return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', type=str, help="Path of the training data in txt.")
    parser.add_argument('param_file', type=str, help="File name under which the trained weights should be saved.")
    args = parser.parse_args()

    num_epoch = 10
    lr = 0.01
    mu = 0.01
    prune_lex = 0.001
    prune_forward = 0.001

    # allow limiting the number of samples for debugging purpose
    # if None, the complete dataset will be used
    num_samples = 10
    num_dev = 5
    num_test = 5

    dev = "../Tiger/develop.txt"
    test = "../Tiger/test.txt"

    print("train_L1_lazy")
    tagger = CRF_POS_Tagger()
    tagger.train_L1_lazy(train_path=args.train_path,
                         param_file=args.param_file,
                         mu=mu,
                         num_epoch=num_epoch,
                         lr=lr,
                         num_train_samples=num_samples,
                         dev_path=dev,
                         num_dev_samples=num_dev,
                         prune_lex_tag_threshold=prune_lex,
                         prune_forward_threshold=prune_forward
                         )
    test_acc = tagger.compute_accuracy(test, num_test_samples=num_test)
    print("Test acc", test_acc)
