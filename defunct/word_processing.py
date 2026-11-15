words = []

with open("durhack_word_list.csv") as doc:
    for line in doc.readlines():
        words.append(line.strip())
        
longest = max(words, key = len)
shortest = min(words, key = len)

#print("Longest Word Length:", len(longest))
#print("Shortest Word Length:", len(shortest))

sorted_words = sorted(words, key=len)

words_to_keys = {}
keys_to_words = {}

for index, word in enumerate(words):
    words_to_keys[word] = index
    keys_to_words[index] = word

test = words_to_keys["yogurt"]
#print(test)
#print(keys_to_words[test])

alphabet = {}
for i in words:
    if i[0] in alphabet.keys():
        alphabet[i[0]] += 1
    else:
        alphabet[i[0]] = 1
        
alphabet = dict(sorted(alphabet.items(), key=lambda item: item[1], reverse=True))
#print(alphabet)



words_list = words.copy()
def rec_tree(depth):
    if depth == 0:
        global words
        if len(words) > 0:
            word = words[0]
            words = words[1:]
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

#path = words_path(words_list, "adversely", "aeriform", "alertly")
#print(path)