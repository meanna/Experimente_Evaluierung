'''
# python load_weight_and_test.py param-file_2000
'''
import argparse
import pickle
from crf_train import CRF_POS_Tagger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('param_file', type=str, help="File name under which the trained weights should be saved.")
    args = parser.parse_args()

    tagger = CRF_POS_Tagger()

    # die gespeicherten Parameter einlesen
    try:
        tagger.weights, tagger.tag_set = pickle.load(open(args.param_file, 'rb'))
    except:
        # load the old format of weights
        _, tagger.tag_set = tagger.read_data("../Tiger/train.txt", None)
        tagger.weights = pickle.load(open(args.param_file, 'rb'))

    acc = tagger.compute_accuracy(test_path="../Tiger/test.txt", num_test_samples=None)
    print(acc)
