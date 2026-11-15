from Visualisation2D import visualise_sequence
from word_tree import word_to_turns, extract_words
from word_processing import words

import random
import argparse

DIRECTIONS = [
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1)
]

def turn_direction(direction, turn):
    dx, dy = direction
    if turn == "F":
        return direction
    elif turn == "L":
        return (-dy, dx)
    elif turn == "R":
        return (dy, -dx)
    
    raise ValueError(f"Invalid turn: {turn}. Must be 'L', 'R', or 'F'.")
"""
Recursively called to generate HP path

direction - indicates the direction looking forward, and is a tuple of (x, y) where x and y are either -1, 0, or 1. The initial direction is (1, 0), which means looking to the right.
"""
def fold(turns, sequence=None, direction=(1, 0)):

    if sequence is None:
        sequence = [(0, 0, "S"), (1, 0, "P")]

    if not turns:
        return sequence

    occupied = {(x, y) for x, y, _ in sequence}

    # First try the requested move
    preferred_direction = turn_direction(direction, turns[0])

    directions = [preferred_direction]

    # Then alternative directions in random order
    alternatives = [d for d in DIRECTIONS if d != preferred_direction]
    random.shuffle(alternatives)

    directions.extend(alternatives)

    for d in directions:

        x = sequence[-1][0] + d[0]
        y = sequence[-1][1] + d[1]

        if (x, y) in occupied:
            continue

        new_sequence = sequence.copy()

        if d == preferred_direction:
            # Valid turn, consume instruction
            new_sequence.append((x, y, "H"))

            result = fold(turns[1:],new_sequence,d)

        else:
            # Alternative move, insert P and retry same turn
            new_sequence.append((x, y, "P"))

            result = fold(turns,new_sequence,d)

        if result is not None:
            return result

    # No possible moves
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-w", "--words", required=True, help="Words")
    ap.add_argument("-o", "--ouput", default="proteins", help="Output directory for visualisation")
    ap.add_argument("-s", "--seed", default=1, help="Seed for random number generator")

    set_seed = int(ap.parse_args().seed)
    random.seed(set_seed)
    args = ap.parse_args()

    characters = extract_words("alphabet.csv")
    words = args.words
    for character in args.words:
        if character not in characters:
            raise ValueError(f"Character '{character}' not found in the word list.")
 
        

    print("Combined word length:", words)
    w = ""
    for character in words:
        w += word_to_turns(characters, character)
    print("Moves required:", w)
    word_sequence = fold(w)
    if word_sequence is None:
        print("No valid folding sequence found for the given words.")
    else:
        visualise_sequence(sequence=word_sequence, filename=f"{words.replace(" ", "_")}.png", output_dir=args.ouput)