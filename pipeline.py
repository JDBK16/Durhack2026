import argparse
import subprocess
from graph_scanner import image
import cv2
from word_tree import turns_to_path, traverse_word_tree, rec_tree

##### Encoding #####

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument("-w1", "--word1", required=True, help="First word to encoding")
ap.add_argument("-w2", "--word2", required=True, help="Second word to encoding")
ap.add_argument("-w3", "--word3", required=True, help="Third word to encoding")
ap.add_argument("-o", "--output", default="proteins", help="Output directory for visualisation")
ap.add_argument("-s", "--seed", default=1, help="Seed for random number generator")
args = ap.parse_args()

subprocess.run([
    "python",
    "main.py",
    "-w1", args.word1,
    "-w2", args.word2,
    "-w3", args.word3,
    "-s", str(args.seed)
])


#### Decoding ####

file_path = "proteins/" + args.word1 + "_" + args.word2 + "_" + args.word3 + ".png"

result = image(file_path)
turns = ""
prev_dir = result[0][1]
for i in result:
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
print(traverse_word_tree(word_tree, p1))
print(traverse_word_tree(word_tree, p2))
print(traverse_word_tree(word_tree, p3))