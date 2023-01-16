# python3 train.py data/train paramfile

import argparse
from collections import defaultdict
import os
import pickle
from math import log

parser = argparse.ArgumentParser()
parser.add_argument('train_dir', type=str, help="E.g. train")
parser.add_argument('paramfile', type=str, help="E.g. parameters")
args = parser.parse_args()

# read training data and compute f(w) per class
freq_class_word = defaultdict(lambda: defaultdict(int))
f_c = {}  # f(c)
for c in os.listdir(args.train_dir):
    data_path = os.path.join(args.train_dir, c)
    file_paths = os.listdir(data_path)
    f_c[c] = len(file_paths)
    print(f"Num {c} = {len(file_paths)}")
    for file_path in file_paths:
        file_path = os.path.join(data_path, file_path)
        with open(file_path, "r", encoding="ISO-8859-1") as f:
            for line in f:
                for token in line.split():
                    freq_class_word[c][token] += 1

r_w_c = defaultdict(lambda: defaultdict(float))  # r(w|c)
p_w = defaultdict(float)  # p(w)
p_c_log = defaultdict(float)  # p(c)

N1, N2 = 0, 0  # for computing discount
total_w_freq = 0  # sum_w' f(w') for computing p(w)

# compute log p(c), discount, and sum_w' f(w') for p(w)
for c in freq_class_word:
    p_c_log[c] = log(f_c[c] / sum(f_c.values()))
    total_w_freq += sum(freq_class_word[c].values())
    for w in freq_class_word[c]:
        if freq_class_word[c][w] == 1:
            N1 += 1
        elif freq_class_word[c][w] == 2:
            N2 += 1

discount = N1 / (N1 + (2 * N2))

# compute r(w|c) and p(w)
for c in freq_class_word:
    freq_class = sum(freq_class_word[c].values())
    for w in freq_class_word[c]:
        # r(w|c) = max(0, f(w,c) –d) / sum_w' f(w',c)
        r_w_c[c][w] = (freq_class_word[c][w] - discount) / freq_class

        # p(w) = f(w) / sum_w' f(w')
        p_w[w] += freq_class_word[c][w] / total_w_freq

# compute log p(w|c)
p_w_c_log = defaultdict(lambda: defaultdict(float))  # p(w|c)
for c in freq_class_word:
    alpha = 1.0 - sum(r_w_c[c].values())
    for w in p_w:
        p_w_c_log[c][w] = log((r_w_c[c][w]) + (alpha * p_w[w]))

# save parameters
parameters = [dict(p_w_c_log), dict(p_c_log)]
with open(args.paramfile + '.pickle', 'wb') as f:
    pickle.dump(parameters, f)
    print("Save parameters as", args.paramfile + '.pickle')
