import re


def levenshtein_distance(word1, word2):
    len1, len2 = len(word1), len(word2)
    # Initialize a matrix to store the edit distances, operations, and positions
    dp = [[(0, "", []) for _ in range(len2 + 1)] for _ in range(len1 + 1)]

    # Initialize the first row and column
    for i in range(len1 + 1):
        dp[i][0] = (i, "d" * i)
    for j in range(len2 + 1):
        dp[0][j] = (j, "i" * j)

    # Fill in the rest of the matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if word1[i - 1] == word2[j - 1] else 1
            # Minimum of deletion, insertion, or substitution
            deletion = dp[i - 1][j][0] + 1
            insertion = dp[i][j - 1][0] + 1
            substitution = dp[i - 1][j - 1][0] + cost
            min_dist = min(deletion, insertion, substitution)

            # which operation led to the minimum distance
            if min_dist == deletion:
                operation = dp[i - 1][j][1] + "d"
            elif min_dist == insertion:
                operation = dp[i][j - 1][1] + "i"
            else:
                operation = dp[i - 1][j - 1][1] + ("s" if cost else "=")

            dp[i][j] = (min_dist, operation)

    # min edit distance, list of operations, positions of operations
    return dp[len1][len2][0], dp[len1][len2][1]


def extract_words(sentence):
    words = re.findall(r"\b[\w']+\b", sentence)
    return words


# edge cases for spans of deletion, insertion, substitution
def handle_delete(start, end, orig, new):
    orig.append([start, end - 1])
    new.append([start - 1, start])


def handle_insert(start, end, orig, new):
    temp_new = [start - 1, start]
    orig.append(temp_new)
    new.append(orig[-1])
    orig[-1], new[-1] = new[-1], temp_new


def handle_substitute(start, end, orig, new):
    orig.append([start, end - 1])
    new.append([start, end - 1])


# editing the last index of the sentence is another edge case
def handle_last_operation(prev_op, start, end, orig, new):
    if prev_op == "d":
        handle_delete(start, end, orig, new)
    elif prev_op == "i":
        handle_insert(start, end, orig, new)
    elif prev_op == "s":
        handle_substitute(start, end, orig, new)


# adjust spans according to edge case expected output
def adjust_last_span(operations, orig, new):
    if operations[-1] == "d":
        new[-1] = [new[-1][0] - 1, new[-1][1] - 1]
        orig[-1] = [orig[-1][0] - 1, orig[-1][0] - 1]
    elif operations[-1] == "i":
        new[-1] = [new[-1][0] - 1, new[-1][1] - 1]
        orig[-1] = [orig[-1][0] - 1, orig[-1][0]]


def get_spans(operations):
    orig = []
    new = []
    prev_op = None
    start = 0
    end = 0
    for i, op in enumerate(operations):
        # prevent span duplication of sequential edits of the same type
        if op != "=":
            if op != prev_op:
                if prev_op:
                    handle_last_operation(prev_op, start, end, orig, new)
                prev_op = op
                start = i
            end = i + 1
        else:
            if prev_op:
                handle_last_operation(prev_op, start, end, orig, new)
                prev_op = None
            start = end
    # edge case of last operation
    if prev_op:
        handle_last_operation(prev_op, start, end, orig, new)
    adjust_last_span(operations, orig, new)
    return orig, new


def get_edits(operations):
    used_edits = []
    prev_op = ""
    for op in operations:
        if op == "i" and prev_op != "i":
            used_edits.append("insertion")
        elif op == "d" and prev_op != "d":
            used_edits.append("deletion")
        elif op == "s" and prev_op != "s":
            used_edits.append("substitution")
        prev_op = op
    return used_edits


def parse_edit(orig_transcript, trgt_transcript):
    word1 = extract_words(orig_transcript)
    word2 = extract_words(trgt_transcript)
    distance, operations = levenshtein_distance(word1, word2)
    orig_span, new_span = get_spans(operations)
    return operations, orig_span, new_span
