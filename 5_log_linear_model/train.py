# command examples
# python3 train.py train-dir paramfile
# e.g. python3 train.py ../data/train paramfile

# If using dev set
# python3 train.py train-dir paramfile --dev_dir dev-dir
# e.g. python3 train.py ../data/train paramfile --dev_dir ../data/dev

from math import log, exp, pow
from collections import defaultdict
import argparse, os, pickle, random


class LogLinearModelClassifier:
    def __init__(self):
        self.train_dir = None
        self.dev_dir = None
        self.paramfile = None
        self.classes = None
        self.train_samples = None
        self.dev_samples = None
        self.weights = None
        self.gradients = None
        self.learning_rate = None
        self.mu = None
        self.epochs = None

    def create_samples(self, data_dir):
        """
        Return a list of sample in the form (label, doc) for a given data directory,
        and a list of document names.
        data_dir can be a directory containing the train/dev/test dataset.
        """
        samples = []

        for c in os.listdir(data_dir):
            data_path = os.path.join(data_dir, c)
            file_names = os.listdir(data_path)
            for file_name in file_names:
                file_path = os.path.join(data_path, file_name)
                with open(file_path, "r", encoding="ISO-8859-1") as f:
                    doc = []
                    for line in f:
                        doc += line.split()
                    if doc:  # prevent issue with empty doc
                        samples.append((c, doc))

        return samples

    def compute_scores(self, features):
        """
        Compute a score for each class given a feature dictionary.
        Return a dictionary in the form class:score
        """
        scores = {}
        for c in features:
            scores[c] = sum(features[c][f] * self.weights[c][f] for f in features[c])
        return scores

    def compute_probs(self, features):
        """
        Compute the probability given feature dictionary.
        Return a dictionary in the form class:prob
        """
        c_to_score = self.compute_scores(features)
        z = self.logsumexp(c_to_score.values())
        probs = {c: exp(score - z) for c, score in c_to_score.items()}
        return probs

    def create_features(self, doc):
        """
        Given a list of tokens (doc), create a feature vector represented by
        a dictionary in dictionary in the form [class][(class,word)]: word frequency.
        """
        features = defaultdict(lambda: defaultdict(float))
        for c in self.classes:
            for w in doc:
                features[c][(c, w)] += 1.0
        return features

    def logsumexp(self, scores):
        """
        Compute Logsumexp given a list of scores.
        """
        max_score = max(scores)
        result = max_score + log(sum([exp(score - max_score) for score in scores]))
        return result

    def train(self, train_dir, paramfile, learning_rate, mu, num_epoch, dev_dir=None):
        """
        Train the model on a given training set. If dev set(dev_dir) is given, compute
        the accuracy on train and dev set and print them out every epoch.
        E.g. Epoch 5: Train Accuracy 0.68, Dev Accuracy 0.664
        Save the parameters every epoch.
        """
        self.train_dir = train_dir
        self.paramfile = paramfile
        self.classes = os.listdir(self.train_dir)
        self.learning_rate = learning_rate
        self.mu = mu
        self.epochs = num_epoch

        if not self.train_samples:
            self.train_samples = self.create_samples(self.train_dir)
            print(f"Number of train samples {len(self.train_samples)}")
        if dev_dir and not self.dev_samples:
            self.dev_dir = dev_dir
            self.dev_samples = self.create_samples(self.dev_dir)
            print(f"Number of dev samples {len(self.dev_samples)}")

        self.weights = defaultdict(lambda: defaultdict(float))

        for epoch in range(1, self.epochs + 1):

            last_updated = {}
            random.shuffle(self.train_samples)
            for i, (true_c, doc) in enumerate(self.train_samples, start=1):
                unique_tokens = set(doc)

                self.gradients = defaultdict(lambda: defaultdict(float))
                expected_freq = defaultdict(lambda: defaultdict(float))

                # compute feature vector and the p(c|d) for each class
                features = self.create_features(doc)
                c_to_probs = self.compute_probs(features)

                # compute the expected frequency
                for c in self.classes:
                    for w in unique_tokens:
                        expected_freq[c][(c, w)] = c_to_probs[c] * features[c][(c, w)]

                # compute the gradients
                for c in self.classes:
                    for w in unique_tokens:
                        self.gradients[c][(c, w)] = features[true_c][(c, w)] - expected_freq[c][(c, w)]

                # efficient weight update
                for c in self.classes:
                    for w in unique_tokens:
                        if w not in last_updated:
                            last_updated[w] = 1

                        # add the gradient part to the weights
                        self.weights[c][(c, w)] += self.learning_rate * (self.gradients[c][(c, w)])

                        # add L2 regularization part to the weights
                        factor = (i + 1) - last_updated[w]

                        self.weights[c][(c, w)] *= pow(1.0 - (self.learning_rate * self.mu), float(factor))
                        last_updated[w] = i

                if i % 1000 == 0:
                    print(f"Process {i} samples")

            # if using dev set, output the accuracy of train and dev set
            # and save the parameters every epoch with a unique name.
            # train acc is useful when we want to apply early stopping, but with L2,
            # it does not work well.
            if dev_dir:
                train_acc, _ = self.compute_accuracy(self.train_samples)
                dev_acc, _ = self.compute_accuracy(self.dev_samples)
                print(f"Epoch {epoch}: Train Accuracy {train_acc}, Dev Accuracy {dev_acc}")
                # save parameters
                paramfile = self.paramfile + f"_lr{learning_rate}_mu{mu}_epoch{epoch}_dev_acc{round(dev_acc, 2)}"

            # save parameters every epoch
            with open(paramfile, 'wb') as f:
                pickle.dump(dict(self.weights), f)
                print("Save parameters as", paramfile)

    def update_weights(self):
        """
        To update the weights with the regular formula. Not used in code.
        """
        for c in self.classes:
            for f in self.weights[c]:
                self.weights[c][f] += self.learning_rate * (self.gradients[c][f] - (self.mu * self.weights[c][f]))

    def compute_accuracy(self, samples):
        """
        Return the accuracy of a given dataset and the predictions.
        """
        correct = 0.0
        preds = []
        for (true_class, doc) in samples:
            features = self.create_features(doc)
            probs = self.compute_probs(features)
            probs_sorted = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            pred_class = probs_sorted[0][0]
            preds.append(pred_class)
            if pred_class == true_class:
                correct += 1
        acc = correct / len(samples)

        return acc, preds

    def test(self, test_dir, paramfile):
        """
        Classify the given test set.
        Print out file name and the predicted class.
        Return the accuracy.
        """
        self.classes = os.listdir(test_dir)
        with open(paramfile, 'rb') as f:
            self.weights = pickle.load(f)
        test_samples = self.create_samples(test_dir)
        test_acc, preds = self.compute_accuracy(test_samples)

        test_file_names = []
        for c in os.listdir(test_dir):
            test_file_names += os.listdir(os.path.join(test_dir, c))

        for i in range(len(test_file_names)):
            print(f"{test_file_names[i]}\t{preds[i]}")

        return test_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', type=str, help="A directory containing the training dataset")
    parser.add_argument('paramfile', type=str, help="Parameters output file name")
    parser.add_argument('--dev_dir', required=False, type=str, help="A directory containing the development dataset")
    args = parser.parse_args()

    model = LogLinearModelClassifier()
    # best epochs are around epoch7,8,9 and higher, it gives dev acc 0.98
    model.train(args.train_dir, args.paramfile, learning_rate=0.05, mu=0.01, num_epoch=10, dev_dir=args.dev_dir)

    # Helper function
    def hyperparam_search():
        learning_rate = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.05, 0.005, 0.005]
        mus = learning_rate
        epochs = 10

        # search for optimal parameters
        for lr in learning_rate:
            for mu in mus:
                model.train(args.train_dir, args.paramfile, lr, mu, epochs, dev_dir=args.dev_dir)
                print("---" * 14)
