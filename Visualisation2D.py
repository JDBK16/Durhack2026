import argparse

import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = "data"

H_COLOUR = "#ff0000"
P_COLOUR = "#000000"
EDGE = "#000000"
START = "#0066ff"
CHAIN = "#000000"
BEST = "#c8402f"
CURRENT = "#9aa0a6"
TEMP = "#2a6fb0"
NODE_SIZE = 150

"""
Taking an input of a sequence of co-ordinates [x,y,type], this function visualises the sequence in a 2D lattice.
The start node is indicated by a triangle
H nodes are indicated by red circles
P nodes are indicated by white squares
"""
def visualise_sequence(sequence, filename="sequence_visualisation.png", output_dir=OUTPUT_DIR):
    n = len(sequence)

    fig, ax = plt.subplots(figsize=(6, 6))

    xs = [x for x, _, _ in sequence]
    ys = [y for _, y, _ in sequence]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    margin = 1

    centre_x = (min_x + max_x) / 2
    centre_y = (min_y + max_y) / 2

    span = max(width, height) + 2 * margin

    ax.set_xlim(
        centre_x - span / 2,
        centre_x + span / 2
    )

    ax.set_ylim(
        centre_y - span / 2,
        centre_y + span / 2
    )

    prev_x = None
    prev_y = None
    # Draw the edges of the lattice
    for x, y, node_type in sequence:
        if prev_x is not None and prev_y is not None:
            ax.plot([prev_x, x], [prev_y, y], color=CHAIN, linewidth=2, zorder=1)

        prev_x, prev_y = x, y

        if node_type == "S":
            ax.scatter(x, y, NODE_SIZE, marker="^", color=START, edgecolors=EDGE, zorder=3)
        elif node_type == "H":
            ax.scatter(x, y, NODE_SIZE, marker="o", color=H_COLOUR, edgecolors=EDGE, zorder=3)
        elif node_type == "P":
            ax.scatter(x, y, NODE_SIZE, marker="s", color=P_COLOUR, edgecolors=EDGE, zorder=3)
        else:
            raise ValueError(f"Unknown node type: {node_type}")
    
    ax.set_aspect("equal")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.suptitle("2D Lattice Visualisation of Sequence", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / filename
    plt.savefig(filename, dpi=600)
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    args = ap.parse_args()

    matplotlib.use("Agg")
    import time
    t = time.time()

    testing_sequence = [(0, 0, "S"), (1, 0, "H"), (2, 0, "P"), (3, 0, "H"), (4, 0, "P"),
                    (5, 0, "H"), (5, 1, "P"), (6,1,"P"), (6,0,"P"), (7, 0, "H"), (8, 0, "P"), (9, 0, "H"),
                    (10, 0, "P"), (11, 0, "H"), (12, 0, "P"), (13, 0, "H"), (14, 0, "P"),
                    (15, 0, "H"), (16, 0, "P"), (17, 0, "H"), (18, 0, "P"), (19, 0, "H")]

    visualise_sequence(testing_sequence)
    print(time.time() - t, "seconds")
