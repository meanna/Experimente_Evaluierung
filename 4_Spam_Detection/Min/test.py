# python3 test.py paramfile.pickle data/test
# Accuracy on Enron6 = 0.9796666666666667

import pickle
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('paramfile', type=str, help="E.g. paramfile.pickle")
parser.add_argument('mail_dir', type=str, help="E.g. test")
args = parser.parse_args()

with open(args.paramfile, 'rb') as f:
    p_w_c_log, p_c_log = pickle.load(f)


def classify(input_file_path, true_class):
    # c^ = argmax_c p(c) p(d|c)
    # c^ = argmax_c p(c) product_i=1..n p(wi | c)
    pred_classes_and_probs = []

    with open(input_file_path, "r", encoding="ISO-8859-1") as f:
        doc = f.readlines()
        for c in p_c_log:
            total_p_w_c_log = 0
            for line in doc:
                for w in line.split():
                    total_p_w_c_log += p_w_c_log[c][w]

            # p(c) p(d|c) for class c
            prob_pred_class = p_c_log[c] + total_p_w_c_log
            pred_classes_and_probs.append((c, prob_pred_class))

    # c^ = argmax_c p(c) p(d|c)
    pred_classes_and_probs.sort(key=lambda x: x[1], reverse=True)
    pred_class = pred_classes_and_probs[0][0]

    # Das Programm test.py soll fur jede Datei ihren Namen und die zugewiesene Klasse
    # mit einem Tabulator als Trennzeichen auf einer Zeile ausgeben.
    print(f"{os.path.basename(input_file_path)}\t{pred_class}")
    return pred_class == true_class


accuracy = 0.0
num_correctly_predicted = 0
num_total_doc = 0

# read the test set, classify, and compute the accuracy.
for class_dir in os.listdir(args.mail_dir):
    file_paths = os.listdir(os.path.join(args.mail_dir, class_dir))
    num_total_doc += len(file_paths)
    for file_name in file_paths:
        file_path = os.path.join(args.mail_dir, class_dir, file_name)

        correctly_predicted = classify(file_path, class_dir)
        if correctly_predicted:
            num_correctly_predicted += 1

accuracy = num_correctly_predicted / num_total_doc
# print("accuracy", accuracy)
