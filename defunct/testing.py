from word_tree import extract_words, rec_tree, word_to_turns, turns_to_path, traverse_word_tree
from main import fold
import random

words = extract_words()
word_tree = rec_tree(10)

def calculate_turns(coords):
    direction = (1, 0)
    prev = (1, 0)
    turns = ""
    for x, y, p in coords[2:]:
        current_direction = (x - prev[0], y - prev[1])
        if p == "H":
            if direction == current_direction:
                turns += "F"
            else:
                if direction == (1, 0):
                    if current_direction == (0, 1):
                        turns += "L"
                    elif current_direction == (0, -1):
                        turns += "R"
                elif direction == (0, 1):
                    if current_direction == (-1, 0):
                        turns += "L"
                    elif current_direction == (1, 0):
                        turns += "R"
                elif direction == (-1, 0):
                    if current_direction == (0, -1):
                        turns += "L"
                    elif current_direction == (0, 1):
                        turns += "R"
                if direction == (0, -1):
                    if current_direction == (1, 0):
                        turns += "L"
                    elif current_direction == (-1, 0):
                        turns += "R"
        direction = current_direction
        prev = (x, y)
    return turns
            
def test(w1="fabulously", w2="quiescence", w3="tenfold"):
    turns = word_to_turns(words, w1) + word_to_turns(words, w2) + word_to_turns(words, w3)
    coords = fold(turns)
    result_turns = calculate_turns(coords)
    turns_per_word = len(result_turns) // 3
    t1, t2, t3 = result_turns[:turns_per_word], result_turns[turns_per_word:2*turns_per_word], result_turns[turns_per_word*2:]
    p1, p2, p3 = turns_to_path(t1), turns_to_path(t2), turns_to_path(t3)
    result_w1, result_w2, result_w3 = traverse_word_tree(word_tree, p1), traverse_word_tree(word_tree, p2), traverse_word_tree(word_tree, p3)
    assert w1 == result_w1 and w2 == result_w2 and w3 == result_w3
    
    
def test_suite():
    for i in range(40):
        test(random.choice(words), random.choice(words), random.choice(words))
        print("Test", i + 1, "Passed!")
        
    
#test_suite()