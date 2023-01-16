# command examples
# python3 test.py paramfile mail-dir
# e.g. python3 test.py paramfile ../data/test

import argparse
from train import LogLinearModelClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('paramfile', type=str, help="path to model weights (pickle)")
    parser.add_argument('mail_dir', type=str, help="Path to test dir")
    args = parser.parse_args()

    model = LogLinearModelClassifier()
    test_acc = model.test(test_dir=args.mail_dir, paramfile=args.paramfile)
