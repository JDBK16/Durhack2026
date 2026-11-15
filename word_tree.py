# Extract the words from the durchack_word_list.csv into an array
def extract_words(filepath="durhack_word_list.csv"):
    words = []
    with open(filepath) as doc:
        for line in doc.readlines():
            words.append(line.replace("\n", ""))
    return words

# Create a ternary tree of depth 10 with the first 30k leafs as words
def rec_tree(depth, words=extract_words(), index=0):
    if depth == 0:
        if index < len(words):
            return [words[index]]
        return []

    size = 3 ** (depth - 1)

    return [
        rec_tree(depth - 1, words, index),
        rec_tree(depth - 1, words, index + size),
        rec_tree(depth - 1, words, index + 2 * size)
    ]

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
    return word_tree


x = [('S', 'strait'), ('P', 'left'), ('H', 'left'), ('H', 'strait'), ('P', 'left'), ('H', 'strait'), ('H', 'left'), ('H', 'right'), ('P', 'left'), ('P', 'left'), ('H', 'right'), ('P', 'left'), ('H', 'strait'), ('H', 'strait'), ('P', 'left'), ('H', 'strait'), ('H', 'right'), ('P', 'left'), ('P', 'left'), ('H', 'right'), ('P', 'left'), ('H', 'strait'), ('H', 'right'), ('P', 'left'), ('P', 'left'), ('H', 'right'), ('P', 'left'), ('H', 'right'), ('P', 'left'), ('H', 'strait'), ('H', 'strait'), ('P', 'left'), ('H', 'right'), ('P', 'left'), ('H', 'right'), ('H', 'strait'), ('H', 'left'), ('H', 'right'), ('H', 'strait'), ('H', 'right'), ('H', 'right'), ('H', 'left'), ('P', 'strait'), ('P', 'left'), ('P', 'left'), ('H', 'right'), ('H', 'strait'), ('H', 'right'), ('H', None)]

x = [('S', 'strait'), ('P', 'left'), ('H', 'left'), ('H', 'strait'), ('P', 'left'), ('H', 'strait'), ('H', 'left'), ('H', 'right'), ('P', 'left'), ('P', 'left'), ('H', 'right'), ('P', 'left'), ('H', 'strait'), ('H', 'strait'), ('P', 'left'), ('H', 'right'), ('H', 'left'), ('H', 'strait'), ('H', 'right'), ('H', 'right'), ('H', 'left'), ('H', 'strait'), ('H', 'strait'), ('H', 'right'), ('H', 'right'), ('H', 'left'), ('P', 'right'), ('P', 'right'), ('H', 'left'), ('H', 'left'), ('P', 'right'), ('H', 'left'), ('H', 'right'), ('H', 'left'), ('H', 'strait'), ('H', 'strait'), ('H', 'left'), ('H', 'strait'), ('H', 'left'), ('H', None)]

x = [('S', 'strait'), ('P', 'left'), ('H', 'left'), ('P', 'strait'), ('H', 'left'), ('P', 'strait'), ('H', 'left'), ('P', 'strait'), ('P', 'strait'), ('H', 'left'), ('P', 'strait'), ('P', 'strait'), ('H', 'left'), ('P', 'strait'), ('P', 'strait'), ('P', 'strait'), ('H', 'left'), ('P', 'strait'), ('P', 'strait'), ('P', 'strait'), ('H', 'left'), ('P', 'strait'), ('P', 'strait'), ('P', 'strait'), ('P', 'strait'), ('H', 'left'), ('H', 'strait'), ('P', 'strait'), ('P', 'strait'), ('P', 'strait'), ('H', 'left'), ('H', 'strait'), ('H', 'right'), ('H', 'right'), ('P', 'strait'), ('P', 'strait'), ('P', 'strait'), ('P', 'strait'), ('H', 'left'), ('H', 'strait'), ('H', 'strait'), ('H', 'right'), ('P', 'strait'), ('P', 'strait'), ('P', 'strait'), ('P', 'strait'), ('H', 'right'), ('P', 'strait'), ('P', 'strait'), ('P', 'strait'), ('P', 'strait'), ('H', 'right'), ('H', 'strait'), ('H', 'strait'), ('H', 'strait'), ('H', 'right'), ('H', 'left'), ('H', 'strait'), ('H', 'strait'), ('H', 'left'), ('H', 'strait'), ('H', 'left'), ('H', None)]

x = [('S', 'strait'), ('P', 'left'), ('H', 'left'), ('H', 'strait'), ('P', 'left'), ('H', 'right'), ('P', 'left'), ('H', 'left'), ('H', 'strait'), ('P', 'strait'), ('P', 'right'), ('P', 'left'), ('H', 'left'), ('H', 'strait'), ('P', 'strait'), ('P', 'right'), ('P', 'left'), ('H', 'left'), ('H', 'strait'), ('H', 'right'), ('P', 'left'), ('H', 'strait'), ('H', 'right'), ('H', 'right'), ('H', 'left'), ('H', 'strait'), ('H', 'strait'), ('H', 'right'), ('H', 'right'), ('H', 'left'), ('P', 'right'), ('H', 'strait'), ('H', 'strait'), ('H', 'left'), ('P', 'strait'), ('H', 'right'), ('H', 'left'), ('H', 'strait'), ('H', 'strait'), ('H', 'left'), ('H', 'strait'), ('H', 'left'), ('H', None)]

x = [('S', 'strait'), ('P', 'left'), ('H', 'left'), ('H', 'strait'), ('P', 'left'), ('H', 'strait'), ('H', 'strait'), ('H', 'right'), ('H', 'left'), ('H', 'strait'), ('H', 'strait'), ('H', 'strait'), ('H', 'left'), ('H', 'left'), ('H', 'strait'), ('P', 'right'), ('P', 'left'), ('H', 'right'), ('H', 'strait'), ('H', 'strait'), ('H', 'strait'), ('H', 'right'), ('H', 'left'), ('H', 'left'), ('H', 'strait'), ('P', 'left'), ('H', 'right'), ('H', 'left'), ('H', 'strait'), ('H', 'right'), ('H', 'strait'), ('H', 'right'), ('H', 'right'), ('H', 'left'), ('P', 'strait'), ('P', 'right'), ('H', 'left'), ('H', None)]

turns = ""
prev_dir = x[0][1]
for i in x:
    if i[0] == "H":
        if prev_dir == "left":
            turns += "L"
        elif prev_dir == "right":
            turns += "R"
        else:
            turns += "F"
    prev_dir = i[1]
    
l = len(turns) // 3
t1, t2, t3 = turns[:l], turns[l:l*2], turns[l*2:]
p1, p2, p3 = turns_to_path(t1), turns_to_path(t2), turns_to_path(t3)
word_tree = rec_tree(10)

print(path_to_turns(index_to_path(extract_words().index("aah"))))
print(t1)
print(traverse_word_tree(word_tree, p1))
print()
print(path_to_turns(index_to_path(extract_words().index("intrepidity"))))
print(t2)
print(traverse_word_tree(word_tree, p2))
print()
print(path_to_turns(index_to_path(extract_words().index("zygotene"))))
print(t3)
print(traverse_word_tree(word_tree, p3))

print(turns)