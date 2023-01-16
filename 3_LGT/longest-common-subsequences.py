import sys


def viterbi(word1: str, word2: str):
    m = len(word1)
    n = len(word2)

    # initialize (m+1)*(n+1) matrix
    matrix = [([''] * (n + 1)) for i in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1] + word1[i - 1]
            else:
                if len(matrix[i][j - 1]) > len(matrix[i - 1][j]):
                    matrix[i][j] = matrix[i][j - 1]
                else:
                    matrix[i][j] = matrix[i - 1][j]
    return matrix[-1][-1]


print(viterbi(sys.argv[1], sys.argv[2]))
