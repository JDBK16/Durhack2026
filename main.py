from Visualisation2D import visualise_sequence
from word_tree import extract_words, word_to_turns

import random
import argparse
import time
from heapq import nlargest

DIRECTIONS = [
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1)
]

# Prevent beam search from wandering forever
MAX_P_RUN = 100


class State:
    def __init__(self, turns, sequence, direction, p_run=0):
        self.turns = turns
        self.sequence = sequence
        self.direction = direction
        self.p_run = p_run


def turn_direction(direction, turn):
    dx, dy = direction

    if turn == "F":
        return direction
    elif turn == "L":
        return (-dy, dx)
    elif turn == "R":
        return (dy, -dx)

    raise ValueError(
        f"Invalid turn: {turn}. Must be 'L', 'R', or 'F'."
    )


def energy(sequence):
    """
    Standard HP-like energy:
    -1 per non-consecutive H-H contact.
    """

    positions = {}

    for i, (x, y, residue) in enumerate(sequence):
        positions[(x, y)] = (i, residue)

    contacts = 0

    for i, (x, y, residue) in enumerate(sequence):

        if residue != "H":
            continue

        neighbours = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]

        for n in neighbours:

            if n not in positions:
                continue

            j, other = positions[n]

            if other != "H":
                continue

            # Ignore chain neighbours
            if abs(i - j) == 1:
                continue

            contacts += 1

    return -(contacts // 2)


def score_state(state, total_turns):
    """
    Strongly favour progress through the
    word encoding before energy.

    This prevents the beam spending excessive
    time exploring long P-detours.
    """

    progress = total_turns - len(state.turns)

    return (
        progress * 100000
        - energy(state.sequence)
    )


def expand_state(state):

    if not state.turns:
        return []

    occupied = {
        (x, y)
        for x, y, _ in state.sequence
    }

    preferred = turn_direction(
        state.direction,
        state.turns[0]
    )

    head_x, head_y, _ = state.sequence[-1]

    children = []

    # ----------------------------------
    # Preferred move -> H
    # ----------------------------------

    x = head_x + preferred[0]
    y = head_y + preferred[1]

    if (x, y) not in occupied:

        seq = state.sequence.copy()
        seq.append((x, y, "H"))

        children.append(
            State(
                state.turns[1:],
                seq,
                preferred,
                0
            )
        )

    # ----------------------------------
    # Alternative moves -> P
    # ----------------------------------

    if state.p_run < MAX_P_RUN:

        for d in DIRECTIONS:

            if d == preferred:
                continue

            x = head_x + d[0]
            y = head_y + d[1]

            if (x, y) in occupied:
                continue

            seq = state.sequence.copy()
            seq.append((x, y, "P"))

            children.append(
                State(
                    state.turns,
                    seq,
                    d,
                    state.p_run + 1
                )
            )

    return children


def fold_beam(turns, beam_width=100):

    total_turns = len(turns)

    beam = [
        State(
            turns,
            [
                (0, 0, "S"),
                (1, 0, "P")
            ],
            (1, 0),
            0
        )
    ]

    best_complete = None
    best_energy = float("inf")

    while beam:

        next_beam = []

        for state in beam:

            if not state.turns:

                e = energy(state.sequence)

                if e < best_energy:
                    best_energy = e
                    best_complete = state.sequence

                continue

            next_beam.extend(
                expand_state(state)
            )

        if not next_beam:
            break

        # ----------------------------------
        # Deduplicate geometrically similar
        # states before ranking.
        # ----------------------------------

        unique = {}

        for state in next_beam:

            x, y, _ = state.sequence[-1]

            sig = (
                x,
                y,
                state.direction,
                len(state.turns)
            )

            s = score_state(
                state,
                total_turns
            )

            if (
                sig not in unique
                or s > unique[sig][0]
            ):
                unique[sig] = (s, state)

        deduped = [
            item[1]
            for item in unique.values()
        ]

        beam = nlargest(
            beam_width,
            deduped,
            key=lambda s: score_state(
                s,
                total_turns
            )
        )

    return best_complete


"""
Original DFS implementation retained for testing
and comparison.
"""
def fold(turns, sequence=None, direction=(1, 0)):

    if sequence is None:
        sequence = [
            (0, 0, "S"),
            (1, 0, "P")
        ]

    if not turns:
        return sequence

    occupied = {
        (x, y)
        for x, y, _ in sequence
    }

    preferred_direction = turn_direction(
        direction,
        turns[0]
    )

    directions = [preferred_direction]

    alternatives = [
        d
        for d in DIRECTIONS
        if d != preferred_direction
    ]

    random.shuffle(alternatives)

    directions.extend(alternatives)

    for d in directions:

        x = sequence[-1][0] + d[0]
        y = sequence[-1][1] + d[1]

        if (x, y) in occupied:
            continue

        new_sequence = sequence.copy()

        if d == preferred_direction:

            new_sequence.append(
                (x, y, "H")
            )

            result = fold(
                turns[1:],
                new_sequence,
                d
            )

        else:

            new_sequence.append(
                (x, y, "P")
            )

            result = fold(
                turns,
                new_sequence,
                d
            )

        if result is not None:
            return result

    return None


if __name__ == "__main__":

    start_time = time.time()

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )

    ap.add_argument(
        "-w1",
        "--word1",
        required=True,
        help="First word to encode"
    )

    ap.add_argument(
        "-w2",
        "--word2",
        required=True,
        help="Second word to encode"
    )

    ap.add_argument(
        "-w3",
        "--word3",
        required=True,
        help="Third word to encode"
    )

    ap.add_argument(
        "-o",
        "--output",
        default="proteins",
        help="Output directory"
    )

    ap.add_argument(
        "-s",
        "--seed",
        default=1,
        help="Random seed"
    )

    args = ap.parse_args()

    random.seed(int(args.seed))

    words = extract_words()

    for word in [
        args.word1,
        args.word2,
        args.word3
    ]:
        if word not in words:
            raise ValueError(
                f"Word '{word}' not found."
            )

    word_combined = f"{args.word1}_{args.word2}_{args.word3}"

    print(
        "Combined word:",
        word_combined
    )

    w1 = word_to_turns(words, args.word1)

    w2 = word_to_turns(words, args.word2)

    w3 = word_to_turns(words, args.word3)

    all_turns = w1 + w2 + w3

    print("Moves required:", w1, w2, w3)

    word_sequence = fold_beam(all_turns, beam_width=100)

    if word_sequence is None:

        print(
            "No valid folding sequence found."
        )

    else:

        print("Energy:", energy(word_sequence))

        visualise_sequence(
            sequence=word_sequence,
            filename=f"{word_combined}.png",
            output_dir=args.output
        )

    print("Runtime:",round(time.time() - start_time, 3),"seconds")