import argparse
import subprocess
from graph_scanner import image
import cv2
from word_tree import turns_to_path, traverse_word_tree, rec_tree, extract_words

import random

if __name__ == "__main__":
    ##### Encoding #####

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-w1", "--word1", help="First word to encoding")
    ap.add_argument("-w2", "--word2", help="Second word to encoding")
    ap.add_argument("-w3", "--word3", help="Third word to encoding")
    ap.add_argument("-o", "--output", default="proteins", help="Output directory for visualisation")
    ap.add_argument("-s", "--seed", default=1, help="Seed for random number generator")
    ap.add_argument("-r", "--random", default=False, help="Generate random")
    args = ap.parse_args()



    #### Decoding ####

    words = extract_words()
    if args.random:
        args.word1 = random.choice(words)
        args.word2 = random.choice(words)
        args.word3 = random.choice(words)

    if args.word1 not in words or args.word2 not in words or args.word3 not in words:
        print("One or more words are not in the dictionary.")
        exit(1)

    subprocess.run([
        "python",
        "main.py",
        "-w1", args.word1,
        "-w2", args.word2,
        "-w3", args.word3,
        "-s", str(args.seed)
    ])

    file_path = "proteins/" + args.word1 + "_" + args.word2 + "_" + args.word3 + ".png"

    result = image(file_path)
    turns = ""
    for i in result:
        if i[1] == "H":
            if i[0] == "L":
                turns += "L"
            elif i[0] == "R":
                turns += "R"
            else:
                turns += "F"
        
    l = len(turns) // 3
    t1, t2, t3 = turns[:l], turns[l:l*2], turns[l*2:]
    p1, p2, p3 = turns_to_path(t1), turns_to_path(t2), turns_to_path(t3)
    word_tree = rec_tree(10)
    print(traverse_word_tree(word_tree, p1))
    print(traverse_word_tree(word_tree, p2))
    print(traverse_word_tree(word_tree, p3))