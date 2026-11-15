from Visualisation2D import visualise_sequence
from word_processing import words
from word_tree import extract_words, word_to_turns

import random
import argparse

DIRECTIONS = [
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1)
]

def compactness(sequence):
    xs = [x for x, _, _ in sequence]
    ys = [y for _, y, _ in sequence]

    width = max(xs) - min(xs) + 1
    height = max(ys) - min(ys) + 1

    return width * height


def fold_score(sequence):

    contacts = -hh_energy(sequence)

    area = compactness(sequence)

    h_count = sum(
        1
        for _, _, t in sequence
        if t == "H"
    )

    if h_count == 0:
        return -area

    return (
        contacts * 100
        - area
    )

def turn_direction(direction, turn):
    dx, dy = direction
    if turn == "F":
        return direction
    elif turn == "L":
        return (-dy, dx)
    elif turn == "R":
        return (dy, -dx)
    
    raise ValueError(f"Invalid turn: {turn}. Must be 'L', 'R', or 'F'.")

def p_extensions(sequence, direction, depth):
    if depth == 0:
        return [(sequence, direction)]
    
    occupied = {(x, y) for x, y, _ in sequence}

    reverse_direction = (-direction[0], -direction[1])

    results = []

    for d in DIRECTIONS:

        if d == reverse_direction:
            continue

        x = sequence[-1][0] + d[0]
        y = sequence[-1][1] + d[1]

        if (x, y) in occupied:
            continue

        new_sequence = sequence.copy()
        new_sequence.append((x, y, "P"))

        results.extend(p_extensions(new_sequence, d, depth - 1))
    return results

def hh_energy(sequence):
    """
    HP lattice energy.

    Counts non-consecutive H-H contacts.

    Lower (more negative) is better.
    """

    energy = 0

    h_nodes = [
        (i, x, y)
        for i, (x, y, t) in enumerate(sequence)
        if t == "H"
    ]

    for n1, (i, x1, y1) in enumerate(h_nodes):

        for j, x2, y2 in h_nodes[n1 + 1:]:

            # Ignore backbone neighbours
            if abs(i - j) == 1:
                continue

            # Lattice adjacent
            if abs(x1 - x2) + abs(y1 - y2) == 1:
                energy -= 1

    return energy

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

    hx = sequence[-1][0] + preferred_direction[0]
    hy = sequence[-1][1] + preferred_direction[1]

    if (hx, hy) not in occupied:
        new_sequence = sequence.copy()
        new_sequence.append((hx, hy, "H"))
        result = fold(turns[1:], new_sequence, preferred_direction)
        if result is not None:
            return result
        
    candidates = []

    for p_depth in range(1, 4):
        for p_sequence, final_direction in p_extensions(sequence, direction, p_depth):
            occupied = {(x, y) for x, y, _ in p_sequence}

            hx = p_sequence[-1][0] + preferred_direction[0]
            hy = p_sequence[-1][1] + preferred_direction[1]

            if (hx, hy) in occupied:
                continue

            trial = p_sequence
            trial.append((hx, hy, "H"))

            score = -100 * hh_energy(trial) - compactness(trial)
            candidates.append((score, trial, final_direction))

        candidates.sort(key=lambda x: x[0], reverse=True)

        for _, trial, new_direction in candidates:
            result = fold(turns[1:], trial, new_direction)
            if result is not None:
                return result

    # No possible moves
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-w1", "--word1", required=True, help="First word to encoding")
    ap.add_argument("-w2", "--word2", required=True, help="Second word to encoding")
    ap.add_argument("-w3", "--word3", required=True, help="Third word to encoding")
    ap.add_argument("-o", "--output", default="proteins", help="Output directory for visualisation")
    ap.add_argument("-s", "--seed", default=1, help="Seed for random number generator")

    set_seed = int(ap.parse_args().seed)
    random.seed(set_seed)
    args = ap.parse_args()

    words = extract_words()
    for word in [args.word1, args.word2, args.word3]:
        if word not in words:
            raise ValueError(f"Word '{word}' not found in the word list.")
        
    word_combined = f"{args.word1}_{args.word2}_{args.word3}"

    print("Combined word length:", word_combined)
    w1 = word_to_turns(words, args.word1)
    w2 = word_to_turns(words, args.word2)
    w3 = word_to_turns(words, args.word3)
    print("Moves required:", w1, w2, w3)
    word_sequence = fold(w1 + w2 + w3)
    if word_sequence is None:
        print("No valid folding sequence found for the given words.")
    else:
        print(len(word_sequence))
        visualise_sequence(sequence=word_sequence, filename=f"{word_combined}.png", output_dir=args.output)