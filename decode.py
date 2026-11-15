"""
Run with no arguments:  python decode_words.py

1. Grabs frames from the camera.
2. graph_scanner finds the shapes by their characteristics and walks the
   chain (unchanged detection).
3. The H turns are split into three parts, each turned into an index path
   through the word tree, which gives the word.
4. Prints the three words.
"""
import contextlib
import io
import sys
import time

import cv2

from graph_scanner import parse_frame, draw_overlay, fit_for_screen
from word_tree import turns_to_path, traverse_word_tree, rec_tree

N_WORDS = 3
TURNS_PER_WORD = 10                          # depth of the word tree
EXPECTED_TURNS = N_WORDS * TURNS_PER_WORD    # number of H nodes (30)
MAX_FRAMES = 40                              # attempts before giving up
WINDOW = "Camera (Q to quit)"
RESULT_MS = 3000                             # how long the result overlay stays up

try:                    # hide OpenCV's own warning spam
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


# ----------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------

def grab(cap, tries=30):
    for _ in range(tries):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            return frame
        time.sleep(0.05)
    return None


def open_camera():
    """First backend / index / resolution that actually delivers frames."""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        for index in (0, 1, 2):
            for size in ((1920, 1080), None):
                cap = cv2.VideoCapture(index, backend)

                if not cap.isOpened():
                    cap.release()
                    continue

                if size:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

                if grab(cap) is not None:
                    return cap

                cap.release()

    return None


# ----------------------------------------------------------------------
# Turns -> word tree -> words
# ----------------------------------------------------------------------

def steps_to_turns(steps):
    """Only H (circle) nodes carry information: their L / R / F move."""
    return "".join(
        move if move in ("L", "R") else "F"
        for move, node_type in steps
        if node_type == "H"
    )


def turns_to_words(turns, word_tree):
    n = len(turns) // N_WORDS
    parts = [turns[:n], turns[n:2 * n], turns[2 * n:]]

    # turns -> index path -> walk the tree -> word
    words = []
    for part in parts:
        leaf = traverse_word_tree(word_tree, turns_to_path(part))
        # a leaf is a one-item list [word] (empty if past the end of the list)
        words.append(leaf[0] if isinstance(leaf, list) and leaf else leaf)
    return words


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    cap = open_camera()

    if cap is None:
        print("Could not get frames from any camera (is another program "
              "using it?).", file=sys.stderr)
        sys.exit(1)

    word_tree = rec_tree(TURNS_PER_WORD)

    words = None
    overlay = None
    last_len = None

    try:
        for _ in range(30):             # let exposure / focus settle
            ok, f = cap.read()
            if ok and f is not None:
                cv2.imshow(WINDOW, fit_for_screen(f))
                cv2.waitKey(1)

        for _ in range(MAX_FRAMES):
            cap.read()                  # drop stale buffered frames
            frame = grab(cap, tries=5)

            if frame is None:
                continue

            # live preview (the window freezes briefly while a frame is analysed)
            cv2.imshow(WINDOW, fit_for_screen(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # scanner prints a lot per frame; keep the output to the words
            with contextlib.redirect_stdout(io.StringIO()):
                steps, nodes, path, board = parse_frame(frame, debug=False)

            if not steps:
                continue

            turns = steps_to_turns(steps)
            last_len = len(turns)
            overlay = draw_overlay(board, nodes, path)

            # only decode a frame that read exactly 30 H turns; any other
            # count means a circle was missed or mistyped, so try next frame
            if len(turns) != EXPECTED_TURNS:
                continue

            try:
                words = turns_to_words(turns, word_tree)
            except (IndexError, TypeError, ValueError):
                words = None
                continue

            break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if overlay is not None:
        cv2.imwrite("debug_overlay.png", overlay)

    if words is None:
        extra = f" (last frame had {last_len} H turns, need {EXPECTED_TURNS})" \
            if last_len is not None else ""
        print("Could not read a fold from the camera" + extra + ".",
              file=sys.stderr)
        sys.exit(1)

    print(" ".join(str(w) for w in words))

    # show what was detected for a few seconds, then close
    if overlay is not None:
        cv2.imshow("Detected Graph", fit_for_screen(overlay))
        cv2.waitKey(RESULT_MS)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()