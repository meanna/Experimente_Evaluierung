'''
Aufruf: crf-annotate.py param-file text.txt
2) Implementierung eines Taggerprogrammes
    Außerdem sollen Sie ein Programm schreiben, welches die gespeicherten Parameter einliest und dann Eingabesätze
    mit dem Viterbi-Algorithmus annotiert.
    Die Eingabesätze werden aus einer Datei eingelesen, die ein Wort pro Zeile enthält, wobei auf jeden Satz eine
    Leerzeile folgt.
    Die Ausgabe erfolgt im gleichen Format wie die Trainingsdaten.
'''
import argparse
import pickle
from crf_train import CRF_POS_Tagger


def read_data(path):
    """
    Read the input file(.txt) that we want to annotate.
    Each line consists of a word, an empty line marks the sentence ending.
    Return a list of samples in the form [[word sequence 1], [word sequence 2],...]
    """
    samples = []
    with open(path, "r", encoding="utf-8") as file:
        words = []
        for line in file:
            if line != '\n':
                words.append(line.strip())
            else:  # end of sentence
                samples.append(words)
                words = []

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('param_file', type=str, help="File name under which the trained weights should be saved.")
    parser.add_argument('input_sentences', type=str, help="Path of input sentences in txt.")
    args = parser.parse_args()

    tagger = CRF_POS_Tagger()

    # die gespeicherten Parameter einlesen
    tagger.weights, tagger.tag_set = pickle.load(open(args.param_file, 'rb'))

    # Eingabesätze mit dem Viterbi-Algorithmus annotieren
    input_sentences = read_data(args.input_sentences)

    for words in input_sentences:  # This is one sentence
        words = [" "] + words + [" "]
        best_tags = tagger.viterbi(words)

        for i in range(len(best_tags)):
            print(words[i + 1] + '\t' + best_tags[i])

        # after write one sentence
        print()
