import os
from utils import tokenize_to_words
from collections import defaultdict
import math
import sys
import json



class DataSample:
    def __init__(self, filename, label):
        """a sample is a doc (file) with a label (category/ class)"""

        self.filename = filename
        self.label = label

    def count_features(self):
        word_count = defaultdict(int)
        with open(self.filename, 'r', encoding='latin-1') as text:  # 'ISO-8859-1'
            tokens = tokenize_to_words(text.read())
            for tok in tokens:
                word_count[tok] = word_count.get(tok,0) + 1

        return word_count  #{christmas: 139}


class DataSet:
    def __init__(self, folder):
        """a dataset is a collection of samples with a feature set (vocab)"""

        self.folder = folder
        self.sample_list = self.collect_samples()
        self.labels = set()
        self.vocab = set()
        self.label_sample_count = defaultdict(int) #{spam: 150002}
        self.word_count_per_class = defaultdict(int)  # {(christmas, ham): 139}, {(christmas, spam): 54}
        self.word_count = defaultdict(int)  # {christmas: 193}
        self.total_words_per_class = defaultdict(int) #{ham: 3840664}

        for sample in self.sample_list:
            self.labels.add(sample.label)
            self.label_sample_count[sample.label] += 1
            for word, count in sample.count_features().items():
                self.vocab.add(word)
                self.word_count_per_class[(word, sample.label)] += count

        for (word, _), count in self.word_count_per_class.items():
            self.word_count[word] += count

        for lab in self.labels:
            for sample in self.sample_list:
                if lab == sample.label:
                    for w, c in sample.count_features().items():
                        self.total_words_per_class[lab] += c

    def collect_samples(self):
        samples = []
        for dir in os.listdir(self.folder):
            label = dir  # the directory name is the label/class
            path = self.folder + '/' + label
            for f in os.listdir(path):
                if f.endswith(".txt"):
                    file = os.path.join(path, f)
                    sample = DataSample(file, label)
                    samples.append(sample)
        return samples


class NB_Classifier:
    def __init__(self, dataset):
        """
        A classifier using Naive Bayes method.
        """
        self.dataset = dataset

        # p(spam) = f(spam)/sum(f(spam + ham))
        self.label_apriori = {l: n / len(self.dataset.sample_list) for l, n in self.dataset.label_sample_count.items()}

        # sum_w' f(w')
        self.total_words = sum(self.dataset.word_count.values())

        # p(w) = f(w)/sum_w' f(w') where f(w) = sum_c f(w,c)
        self.word_apriori = {w: count/self.total_words for w, count in self.dataset.word_count.items()}

    def calc_KN_disc(self, label):
        """"calculates Kneser Ney discount = N1/(N1+2N2) for each class"""

        N1 = 0
        N2 = 0
        for (_, lab), count in self.dataset.word_count_per_class.items():
            if lab == label:
                if count == 1:
                    N1 += 1
                elif count == 2:
                    N2 += 1

        discount = N1 / (N1 + 2 * N2)

        return discount

    def calc_prob(self, word, label):
        """calculates relative frequency by deducting a discount from positive frequencies"""

        # r(w|c) = max(0,f(w,c) - disc) / sum_w' f(w',c)
        rel_prob = defaultdict(float)
        total_relfreq = defaultdict(float)
        alpha = defaultdict(float)
        prob = defaultdict(float)

        for (w, lab), count in self.dataset.word_count_per_class.items():
            if w == word and lab == label:
                rel_prob[(w, lab)] = max(0, count - self.calc_KN_disc(lab))/self.dataset.total_words_per_class[lab]
                total_relfreq[lab] += rel_prob[(w, lab)]
                alpha[lab] = 1 - total_relfreq[lab]

                #p(w|c) = r(w|c) + alpha(c) p(w)
                prob[(w, lab)] = rel_prob[(w,lab)] + alpha[lab] * self.word_apriori[w]

        return prob


def main(train_folder, paramfile):
    dataset = DataSet(train_folder)
    classifier = NB_Classifier(dataset)

    labels = ['spam', 'ham']
    word = 'christmas'

    for l in labels:
        word_count_per_class = dataset.word_count_per_class[(word,l)]
        print('wordcount per class (' + word + '|' + l + '): ' + str(word_count_per_class))
        print('total wordcount for ' + word + ' -> ' + str(dataset.word_count[word]))
        print('total word per class: ' +l + ' -> ' + str(dataset.total_words_per_class[l]))
        print('discount for ' + l + ' -> ' + str(classifier.calc_KN_disc(l)))
        prob = classifier.calc_prob(word, l)
        print('probability for '+ word + '|' + l + ' -> ' + str(prob))

    #with open(paramfile, 'wt') as file:
       #json.dump(classifier.calc_word_label_prob(), file)


traindir = "/home/michelangelolinu/Dokumente/Enron/train"
paramfile = "paramfile"

main(traindir, paramfile)

# if __name__ == "__main__":
#   main(sys.argv[1], sys.argv[2])
