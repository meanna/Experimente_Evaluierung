class TreeReader:

    def read_from_file(self, filepath):

        with open(filepath, 'r') as file:
            for line in file:
                tree = line.strip()
                constituents, words = self.parse_a_tree(tree)
                yield tree, constituents, words

    def parse_a_tree(self, tree):
        """
        Given a parse tree, return a list of sorted constituents (depth first traveral),
        and a list of words.
        input parse = "(TOP(S(NP(NNP Ms.)(NNP Haag))(VP(VBZ plays)(NP(NNP Elianti)))(. .)))"

        output constituents: [('TOP=S', 0, 5), ('NP', 0, 2), ...]
        output words: ['Ms.', 'Haag', ...]

        """
        constituents, words, _ = self.process_parse(tree, constituents=[], words=[], i=0)

        constituents = self.remove_chain_rules(constituents)
        return constituents, words

    def reconstruct_a_tree(self, constituents, words):
        """
        Given a constituent list and a word list, reconstruct the original tree.

        input constituents: [('TOP=S', 0, 5), ('NP', 0, 2), ...]
        input words: ['Ms.', 'Haag', ...]

        outout tree: "(TOP(S(NP(NNP Ms.)(NNP Haag))(VP(VBZ plays)(NP(NNP Elianti)))(. .)))"
        """
        constituents = self.expand_to_chain_rules(constituents)
        tree, _, _ = self.get_parse(constituents, words)
        return tree

    @staticmethod
    def read_token(parse, pos):
        token = ""
        i = 0
        while parse[pos + i] not in ["(", ")", " "]:
            token += parse[pos + i]
            i += 1
        return token, pos + i

    def process_parse(self, parse, constituents, words, i):
        """
        Helper function.
        Given a parse tree, return a list of unsorted constituents, list of words,
        and i is the current position in the parse string.
        > parse = "(TOP(S(NP(NNP Ms.)(NNP Haag))(VP(VBZ plays)(NP(NNP Elianti)))(. .)))"

        output constituents: [('DT', 0, 1), ('NN', 1, 2), ('NN', 2, 3) .. ]
        output words: ['Ms.', 'Haag', 'plays', 'Elianti', '.']

        """
        start = len(words)
        while i < len(parse):
            # if see (, call the function recursively
            # if not, read the token until seeing a space or )
            if parse[i] == "(":
                i += 1
                result_list, _, i = self.process_parse(parse, constituents, words, i)
                constituents += result_list

            # if see a space, it means, we just read a label
            elif parse[i] == " ":
                label = token
                i += 1

            elif parse[i] == ")":
                # if the previous position is not ), we just read a word
                # otherwise, it is a label e.g. the last ) belongs to VP in (VP(N car))
                if parse[i - 1] != ")":
                    word = token
                    words.append(word)
                else:
                    label = token
                end = len(words)
                i += 1
                return [(label, start, end)], words, i
            else:
                token, i = self.read_token(parse, i)

        return constituents, words, i

    @staticmethod
    def expand_to_chain_rules(constituents):
        """
        Given a constituent list without chain rules,
        e.g. [('TOP=S', 0, 5), ..]
        Return a new list where the modified rules are expanded back to chain rules
        e.g. [('TOP', 0, 5), ('S', 0, 5), ..]
        """
        new_list = []
        for label, start, end in constituents:
            if "=" in label:
                chained_labels = label.split("=")
                for chained_label in chained_labels:
                    new_list.append((chained_label, start, end))

            else:
                new_list.append((label, start, end))
        return new_list

    def get_parse(self, constituents, words):
        """
        Given constituent list (in depth first traversal order, and with chain rules)
        and a word list, return the parse tree as string.
        > constituents= [('TOP', 0, 5), ('S', 0, 5), ('NP', 0, 2), ('NNP', 0, 1) ..]
        > words= ['Ms.', 'Haag', 'plays', 'Elianti', '.']

        output = "(TOP(S(NP(NNP Ms.)(NNP Haag))(VP(VBZ plays)(NP(NNP Elianti)))(. .)))"

        """
        if constituents:
            out = ""
            # handle chain rule e.g.
            if_chain_rule = False
            label, start, end = constituents[0]
            if constituents[1:]:
                label1, start1, end1 = constituents[1]
                if start == start1 and end == end1:
                    if_chain_rule = True
            if if_chain_rule:
                out += "(" + label
                constituents = constituents[1:]
                child, ind, constituents = self.get_parse(constituents, words)
                out += child + ")"
                return out, end, constituents
            else:
                # if the node has children
                if end != start + 1:
                    out += "(" + label
                    ind = start
                    constituents = constituents[1:]
                    while ind < end:
                        child, ind, constituents = self.get_parse(constituents, words)
                        out += child
                    out += ")"
                    return out, end, constituents
                # if the node does not have any children
                else:
                    out += "(" + label + " " + words[start] + ")"
                    return out, end, constituents[1:]

    @staticmethod
    def remove_chain_rules(constituents):
        """
        Sort the given constituent list to have an order of depth first traversal.
        Remove chain rules from the given constituent list.
        input e.g. [('TOP', 0, 5), ('S', 0, 5), ('NP', 0, 2), ('NNP', 0, 1) .. ]
        output e.g. [('TOP=S', 0, 5), ('NP', 0, 2), ('NNP', 0, 1), ..]

        """
        constituents = sorted(constituents, key=lambda x: (x[1], -x[2]))
        final_constituents = []
        i = 0
        while i <= len(constituents) - 1:

            label, start, end = constituents[i]
            chain_labels = [label]
            for j in range(i + 1, len(constituents)):
                next_label, next_start, next_end = constituents[j]
                if next_start == start and next_end == end:
                    chain_labels.append(next_label)
                else:
                    break
            chain_labels.reverse()

            merged_label = "=".join(chain_labels)

            con = (merged_label, start, end)
            final_constituents.append(con)
            i += len(chain_labels)
        return final_constituents


if __name__ == "__main__":
    p = TreeReader()


    def test_read_a_tree():
        tree = "(TOP(S(NP(NNP Ms.)(NNP Haag))(VP(VBZ plays)(NP(NNP Elianti)))(. .)))"

        constituents, words = p.parse_a_tree(tree)
        recon_tree = p.reconstruct_a_tree(constituents, words)
        print("tree :")
        print(tree)
        print("reconstructed tree :")
        print(recon_tree)
        assert tree == recon_tree


    def test_read_from_file():
        # To test if the parse works properly
        # It can handle trees with chain rules that are longer than 2 rules
        outputs = p.read_from_file(filepath="./PennTreebank/data/train.txt")

        for tree, constituents, words in outputs:
            recon_tree = p.reconstruct_a_tree(constituents, words)
            # print("tree")
            # print(tree)
            # print("recon")
            # print(recon_tree)
            # print("..." * 10)
            assert tree == recon_tree


    test_read_a_tree()
    test_read_from_file()
