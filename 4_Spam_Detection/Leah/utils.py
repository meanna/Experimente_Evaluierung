import re
import string

def split_by_symbols(token):
    """
    Given a string token, split into subtokens by "'", '"', "(", ")"
    e.g. "(Hello.)" -> "Hello."
    Then split each subtoken by ",", ".", "!", "?", ";", ":"
    e.g. "Hello." -> "Hello"
    :param token: a token
    :return: list of tokens without the punctuations
    """
    split_result = filter(None, re.split('[\'"()]', token))

    split_result_ = []
    for token in split_result:
        if len(token) > 1 and token[-1] in string.punctuation:
            split_result_.append(token[:-1])
        else:
            split_result_.append(token)
    return split_result_


def tokenize_to_words(text):
    """
    Tokenize a text string into words. Return a list of words.
    Steps
    1. split by spaces
    2. split by special symbols in the token ["'", '"', "(", ")"]
    3. split by symbols that are at the end of punctuations
    Return a list of normalized words
    """
    words = []
    tokens = text.split()
    for token in tokens:
        # do not consider punctuations
        if token not in string.punctuation:
            split_result = split_by_symbols(token.lower()) #normalize words
            words += split_result
    return words


sent = "Schicken  Sie  das  fertige  Programm,  die  Liste  der  ausgegebenen  Klassen  " \
       "und  die erzielte Genauigkeit (Accuracy) an schmid@cis.lmu.de. Die Genauigkeit können Sie von Hand ausrechnen."

sent2 = "villarreal / hou / ect @ ect , kimberly rizzi / hou / ect @ ect , fran l mayes / hou / ect @ ect ,"

sent3 = "Subject: congratulations congratulations on your expanded role . i hope this means you get lots more money " \
        "and fewer hours !"

#print(tokenize_to_words(sent3))