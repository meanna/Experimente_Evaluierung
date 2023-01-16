# !wget -O abbr.txt https://www.cis.lmu.de/~schmid/lehre/Experimente/data/abbreviations
import sys
import re


class Tokenizer:
    def __init__(self, abbreviations):
        self.abbr_list = []
        with open(abbreviations, "r", encoding='utf-8') as f:
            for line in f:
                self.abbr_list.append(line.strip())

    def tokenize_to_words(self, text):
        """
        Tokenize a text string into words. Return a list of words.
        Steps
        1. split by spaces
        2. detect abbreviations
        3. split by special symbols in the token ["'", '"', "(", ")"]
        4. split by symbols that are at the end of tokens [",", ".", "!", "?", ";", ":"]
        """
        words = []
        tokens = text.split()
        for token in tokens:
            if token in self.abbr_list:
                words.append(token)
            else:
                split_result = self.split_by_symbols(token)
                words += split_result
        return words

    @staticmethod
    def tokenize_to_sentences(tokens, separator=" ", print_output=False):
        """
        Tokenize a list of tokens into a list of sentences.
        :param tokens: list of tokens
        :param separator: sentence separator that will be printed out such as " " or "|"
        :return: A list of sentences
        """
        sentences = []
        sentence = []
        num_quotation_marks = 0

        for i in range(len(tokens)):
            sentence.append(tokens[i])
            # dealing with ""
            # e.g. "Blabla", sagt er.
            # e.g. Die "schnittpolizei" denken.
            # e.g. Er sagt: "blablabla. Blablabla."
            if tokens[i] == '"':
                num_quotation_marks += 1
                if num_quotation_marks % 2 == 0:
                    num_quotation_marks = 0
                    if (i < len(tokens) - 1 and tokens[i - 1] in [".", "!", "?", ")"] and tokens[i + 1] != ',') or (
                            i == len(tokens) - 1):
                        sentences.append(sentence)
                        sentence = []

            # end of sentence
            elif tokens[i] in [".", "!", "?", ")"]:
                if i == len(tokens) - 1 or (num_quotation_marks % 2 == 0 and tokens[i + 1] not in [")", ".", ","]):
                    sentences.append(sentence)
                    sentence = []

        if sentence:
            sentences.append(sentence)
        if print_output:
            for sentence in sentences:
                print(separator.join(sentence))
        return sentences

    @staticmethod
    def split_by_symbols(token):
        """
        Given a string token, split into subtokens by "'", '"', "(", ")"
        e.g. "(Hello.)" -> "(", "Hello.", ")"
        Then split each subtoken by ",", ".", "!", "?", ";", ":"
        e.g. "Hello." -> "Hello", "."
        :param token: a token
        :return: list of tokens
        """
        delimiters = ("'", '"', "(", ")")
        regex_pattern = "(" + '|'.join(map(re.escape, delimiters)) + ")"
        # e.g. "(Hello.)" -> "(", "Hello.", ")"
        split_result = list(filter(None, re.split(regex_pattern, token)))

        # e.g. "Hello." -> "Hello", "."
        split_result_ = []
        for token in split_result:
            if len(token) > 1 and token[-1] in [",", ".", "!", "?", ";", ":"]:
                split_result_.append(token[:-1])
                split_result_.append(token[-1])
            else:
                split_result_.append(token)

        return split_result_


def test_tokenize(tokenizer):
    def run_test(input, expected_result):
        tokens = tokenizer.tokenize_to_words(input)
        sentences = tokenizer.tokenize_to_sentences(tokens, separator=" ", print_output=False)
        assert sentences == expected_result

    text = "word word (word word.). word word? (word word's?) word."
    expected = [['word', 'word', '(', 'word', 'word', '.', ')', '.'], ['word', 'word', '?'],
                ['(', 'word', 'word', "'", 's', '?', ')'], ['word', '.']]

    run_test(text, expected)

    text = "word word. (word word word.) word word word?"
    expected = [['word', 'word', '.'], ['(', 'word', 'word', 'word', '.', ')'], ['word', 'word', 'word', '?']]
    run_test(text, expected)

    text = 'Er sagt: "blablabla. Blablabla."'
    expected = [['Er', 'sagt', ':', '"', 'blablabla', '.', 'Blablabla', '.', '"']]
    run_test(text, expected)
    text = 'Die "schnittpolizei" denken.'
    expected = [['Die', '"', 'schnittpolizei', '"', 'denken', '.']]
    run_test(text, expected)
    text = '"Blabla", sagt er.'
    expected = [['"', 'Blabla', '"', ',', 'sagt', 'er', '.']]
    run_test(text, expected)

    text = "word word , word word. word word?"

    expected = [['word', 'word', ',', 'word', 'word', '.'], ['word', 'word', '?']]
    run_test(text, expected)
    text = 'Das sei Aufgabe der Polizei:"Ich habe schon früher gesagt, Polizei." Mit "anderen Stellen" meint ' \
           'er etwa den Kulturminister.'
    expected = [['Das', 'sei', 'Aufgabe', 'der', 'Polizei', ':', '"', 'Ich', 'habe', 'schon', 'früher', 'gesagt', ',',
                 'Polizei', '.', '"'], ['Mit', '"', 'anderen', 'Stellen', '"'
                    , 'meint', 'er', 'etwa', 'den', 'Kulturminister', '.']]

    run_test(text, expected)
    text = "tagesschau.de Haber-Bosch-Verfahren 10%-ige Nutzer*innen 12,345. word word! "
    expected = [['tagesschau.de', 'Haber-Bosch-Verfahren', '10%-ige', 'Nutzer*innen', '12,345', '.'],
                ['word', 'word', '!']]

    run_test(text, expected)
    print("All tests passed")


def main(abbreviations, text_file):
    tokenizer = Tokenizer(abbreviations)
    test_tokenize(tokenizer)
    tokens = []
    with open(text_file, "r", encoding='utf-8') as f:
        for line in f:
            tokens += tokenizer.tokenize_to_words(line)

    tokenizer.tokenize_to_sentences(tokens, separator=" ", print_output=True)


# set PYTHONIOENCODING=utf8 or export PYTHONIOENCODING=utf8
# python tokenize.py abbr.txt text.txt > tokens.txt
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
