def read_data(path):
    """
    Return a list of sample in the form (word sequence, tag sequence)
    and a set of tags
    """
    samples = []
    tagset = set()
 
    with open(path, "r") as f:
        words, tags = [],[]
        for line in f: 
            l = line.split()
            if l:
                words.append(l[0])
                tags.append(l[1])
                tagset.add(l[1])
            else:  # end of sentence
                samples.append((words, tags))
                words, tags = [],[]

    return samples, tagset
