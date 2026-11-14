def extract_words():
    words = []
    with open("durhack_word_list.csv") as doc:
        for line in doc.readlines():
            words.append(line.strip())
    return words

word_exhaust = extract_words()
def rec_tree(depth):
    if depth == 0:
        global words_exhaust
        if len(words_exhaust) > 0:
            word = words_exhaust[0]
            words_exhaust = words_exhaust[1:]
            return [word]
        return []
    return [rec_tree(depth-1), rec_tree(depth-1), rec_tree(depth-1)]
word_tree = rec_tree(10)

def index_to_path(index, depth=10):
    path = []

    for _ in range(depth):
        path.append(index % 3)
        index //= 3

    return path[::-1]

def path_to_turns(path):
    turns = ""
    for i in path:
        if i == 0:
            turns += "L"
        elif i == 1:
            turns += "F"
        else:
            turns += "R"
    return turns

def words_path(words, w1, w2, w3):
    i1 = words.index(w1)
    i2 = words.index(w2)
    i3 = words.index(w3)
    p1 = index_to_path(i1, 10)
    p2 = index_to_path(i2, 10)
    p3 = index_to_path(i3, 10)
    turns = path_to_turns(p1) + path_to_turns(p2) + path_to_turns(p3)
    return turns

path = words_path(extract_words, "zygodactyl", "zygomorphous", "zygotene")
