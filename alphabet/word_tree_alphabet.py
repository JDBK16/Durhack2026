# Extract the words from the durchack_word_list.csv into an array
def extract_words(filepath="durhack_word_list.csv"):
    words = []
    try:
        with open(filepath) as doc:
            for line in doc.readlines():
                words.append(line.replace("\n", ""))
    except FileNotFoundError:
        print(f"File {filepath} not found. Please ensure the file exists.")
    return words

# Create a ternary tree of depth 10 with the first 30k leafs as words
word_exhaust = extract_words("alphabet.csv")
def rec_tree(depth):
    if depth == 0:
        global word_exhaust
        if len(word_exhaust) > 0:
            word = word_exhaust[0]
            word_exhaust = word_exhaust[1:]
            return [word]
        return []
    return [rec_tree(depth-1), rec_tree(depth-1), rec_tree(depth-1)]
word_tree = rec_tree(10)

# Conversts index in words array to path in word tree
def index_to_path(index, depth=10):
    path = []

    for _ in range(depth):
        path.append(index % 3)
        index //= 3

    return path[::-1]

# Converts path in word tree to turns in protein folding
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

# Turns a word into turns
def word_to_turns(words, word):
    i = words.index(word)
    p = index_to_path(i, 10)
    turns = path_to_turns(p)
    return turns

# Converts protein folding turns into a path in the word tree
def turns_to_path(turns):
    indices = ""
    for i in turns:
        if i == "L":
            indices += "0"
        elif i == "F":
            indices += "1"
        else:
            indices += "2"
    return indices

# Returns the word at the path in the word tree
def traverse_word_tree(word_tree, i):
    for index in i:
        word_tree = word_tree[int(index)]
    return word_tree[0]
