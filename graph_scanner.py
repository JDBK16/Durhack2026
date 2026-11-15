import cv2
import math
import sys
import numpy as np

# Overlay colours (BGR), used for drawing only, never for detection
COLOURS = {
    "S": (255, 0, 0),
    "H": (0, 0, 255),
    "P": (0, 255, 0),
}

RAW_MODE = "--raw" in sys.argv     # skip board-outline detection
RAW_MODE = True

TARGET_SIZE = 1600      # board is rescaled so its longest side is this
COARSE_THRESH = 20      # pass-1 darkness threshold
CORE_MIN = 12           # minimum node contrast in refine_blob
LINE_THRESH = 8         # connector line darkness threshold


# ----------------------------------------------------------------------
# Board detection / perspective correction (optional)
# ----------------------------------------------------------------------

def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_square(image, pts, size=1000):
    pts = order_points(pts)

    dst = np.array(
        [
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, matrix, (size, size))


def find_board(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 10000:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        _, _, w, h = cv2.boundingRect(approx)
        ratio = w / float(h)

        if not (0.8 <= ratio <= 1.2):
            continue

        if area > best_area:
            best_area = area
            best = approx.reshape(4, 2)

    if best is None:
        return None, None

    board = warp_square(frame, best)

    margin = int(board.shape[0] * 0.04)
    board = board[margin:-margin, margin:-margin]

    return board, best


def normalise(board, target=TARGET_SIZE):
    """Rescale so nodes are big enough for reliable shape measurement."""
    h, w = board.shape[:2]
    s = target / float(max(h, w))

    if abs(s - 1.0) < 0.05:
        return board

    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    return cv2.resize(board, None, fx=s, fy=s, interpolation=interp)


# ----------------------------------------------------------------------
# Shape detection (no colour, only "darker than the local page")
# ----------------------------------------------------------------------

def _ellipse(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def darkness(board, blur=7):
    """
    How much darker each pixel is than the page around it (0 = paper).
    The page brightness is estimated locally with a large closing, so
    shading / vignetting is removed.
    """
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)

    if blur > 1:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    # kernel must be larger than a node, smaller than the shading gradient
    k = int(min(board.shape[:2]) * 0.08) | 1
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, _ellipse(k))

    return cv2.subtract(bg, gray)


def line_kernel(board):
    """Wider than a connector line, much smaller than a node."""
    return _ellipse(max(7, int(max(board.shape[:2]) * 0.010)) | 1)


def refine_blob(diff, cx, cy, r, kernel):
    """
    Re-threshold one node at half of its own contrast (half-max sits on
    the true edge of a blurred shape), remove connector stubs, and
    return the contour of the blob containing the node centre.
    """
    H, W = diff.shape

    half = int(r * 1.8) + 2
    x0, x1 = max(0, int(cx) - half), min(W, int(cx) + half + 1)
    y0, y1 = max(0, int(cy) - half), min(H, int(cy) + half + 1)

    roi = diff[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    yy, xx = np.ogrid[y0:y1, x0:x1]
    core_px = roi[(xx - cx) ** 2 + (yy - cy) ** 2 <= (0.5 * r) ** 2]

    if core_px.size == 0:
        return None

    core = float(np.median(core_px))
    if core < CORE_MIN:
        return None

    mask = (roi > core * 0.5).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n, labels = cv2.connectedComponents(mask)
    if n < 2:
        return None

    lx = min(max(int(round(cx)) - x0, 0), labels.shape[1] - 1)
    ly = min(max(int(round(cy)) - y0, 0), labels.shape[0] - 1)
    lab = labels[ly, lx]

    if lab == 0:
        lab = int(np.argmax(np.bincount(labels.ravel())[1:])) + 1

    comp = (labels == lab).astype(np.uint8)

    cnts, _ = cv2.findContours(
        comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    return c + np.array([x0, y0], dtype=c.dtype)


def shape_metrics(c):
    """
    Two rotation- and blur-tolerant shape measures:

      fill = area / area of minimum enclosing circle
             circle ~0.9-1.0, square ~0.64-0.75, triangle ~0.41-0.55

      tri  = area / area of minimum enclosing triangle
             triangle ~0.8-1.0, circle ~0.6, square ~0.5

    A triangle is the shape with high tri AND low fill.
    """
    area = cv2.contourArea(c)
    (_, _), rad = cv2.minEnclosingCircle(c)

    if rad < 1 or area < 1:
        return None

    fill = area / (math.pi * rad * rad)

    try:
        tri_area, _ = cv2.minEnclosingTriangle(c.astype(np.float32))
        tri = area / tri_area if tri_area > 0 else 0.0
    except cv2.error:
        tri = 0.0

    return fill, min(tri, 1.2)


def detect_shapes(board, debug=False):
    diff = darkness(board)
    h, w = diff.shape

    # must be wider than the connector lines so they are removed
    kernel = line_kernel(board)

    # ---- pass 1: coarse blobs --------------------------------------
    coarse = (diff > COARSE_THRESH).astype(np.uint8) * 255
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN, kernel)

    if debug:
        cv2.imwrite("debug_board.png", board)
        cv2.imwrite(
            "debug_darkness.png",
            cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX),
        )
        cv2.imwrite("debug_coarse.png", coarse)

    contours, _ = cv2.findContours(
        coarse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = 0.00010 * h * w

    blobs = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue                      # touches the border -> not a node

        blobs.append((a, c))

    if debug:
        print(
            f"board {w}x{h}, kernel {kernel.shape[0]}px, "
            f"coarse blobs: {len(blobs)} "
            f"(contours before filtering: {len(contours)}), "
            f"max darkness: {diff.max()}"
        )

    if not blobs:
        return []

    med = float(np.median([a for a, _ in blobs]))

    # ---- pass 2: refine each node and measure its shape ------------
    nodes = []

    for a, c in blobs:
        # triangle is smaller than the others, so keep the lower bound loose
        if a < 0.15 * med or a > 2.5 * med:
            if debug:
                print(f"skipped blob area={a:.0f} (median {med:.0f})")
            continue

        m = cv2.moments(c)
        if m["m00"] == 0:
            continue

        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        r = math.sqrt(a / math.pi)

        refined = refine_blob(diff, cx, cy, r, kernel)

        if refined is None:
            if debug:
                print(f"refine failed at ({cx:.0f}, {cy:.0f}) area={a:.0f}")
            continue

        metrics = shape_metrics(refined)
        if metrics is None:
            continue

        fill, tri = metrics

        rm = cv2.moments(refined)
        if rm["m00"] == 0:
            continue

        rarea = cv2.contourArea(refined)

        nodes.append(
            {
                "type": None,
                "fill": fill,
                "tri": tri,
                "score": tri - fill,          # high = triangle-like
                "x": rm["m10"] / rm["m00"],
                "y": rm["m01"] / rm["m00"],
                "r": math.sqrt(rarea / math.pi),
            }
        )

    return nodes


def estimate_spacing(nodes):
    """
    Grid spacing = the most common nearest-neighbour distance.
    Robust against stray fragments and duplicate detections.
    """
    if len(nodes) < 2:
        return None

    pts = np.array([[n["x"], n["y"]] for n in nodes])
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)

    best_members = None

    for c in nn:
        members = nn[(nn > 0.85 * c) & (nn < 1.15 * c)]
        if best_members is None or len(members) > len(best_members):
            best_members = members

    return float(np.median(best_members))


def dedupe(nodes, spacing):
    """Merge nodes closer together than half the grid spacing."""
    kept = []

    for n in sorted(nodes, key=lambda n: -n["r"]):
        if all(
            math.hypot(n["x"] - k["x"], n["y"] - k["y"]) > 0.5 * spacing
            for k in kept
        ):
            kept.append(n)

    return kept


def keep_main_cluster(nodes, spacing):
    """
    Keep only the largest group of nodes that are chained together
    by grid-sized gaps. Everything outside the graph is discarded.
    """
    n = len(nodes)
    if n == 0:
        return nodes

    link = 1.3 * spacing
    seen = [False] * n
    best = []

    for s in range(n):
        if seen[s]:
            continue

        comp = [s]
        seen[s] = True
        stack = [s]

        while stack:
            i = stack.pop()
            for j in range(n):
                if seen[j]:
                    continue
                d = math.hypot(
                    nodes[i]["x"] - nodes[j]["x"],
                    nodes[i]["y"] - nodes[j]["y"],
                )
                if d <= link:
                    seen[j] = True
                    comp.append(j)
                    stack.append(j)

        if len(comp) > len(best):
            best = comp

    return [nodes[i] for i in best]


# ----------------------------------------------------------------------
# Typing: start (triangle) first, then circle vs square
# ----------------------------------------------------------------------

def circle_square_threshold(fills):
    """
    Split the remaining nodes into two groups by fill (two-means).
    Circles have the high fill, squares the lower. Falls back to a
    fixed value if the two groups are not clearly separate.
    """
    if len(fills) < 4:
        return 0.80

    lo, hi = min(fills), max(fills)

    for _ in range(25):
        t = (lo + hi) / 2.0
        a = [f for f in fills if f < t]
        b = [f for f in fills if f >= t]
        if a:
            lo = float(np.mean(a))
        if b:
            hi = float(np.mean(b))

    if hi - lo < 0.08:
        return 0.80

    return (lo + hi) / 2.0


def assign_types(nodes, debug=False):
    """
    There is always exactly one triangle: the node with the highest
    triangle score (tri - fill). No fixed cut-off, so it is always found.
    Every other node is a circle (H) or square (P).
    """
    if not nodes:
        return 0.80

    start = max(range(len(nodes)), key=lambda i: nodes[i]["score"])

    others = [n["fill"] for i, n in enumerate(nodes) if i != start]
    thr = circle_square_threshold(others)

    for i, n in enumerate(nodes):
        if i == start:
            n["type"] = "S"
        else:
            n["type"] = "H" if n["fill"] >= thr else "P"

    if debug:
        ranked = sorted(nodes, key=lambda n: -n["score"])[:3]
        print("start candidates (score, tri, fill):")
        for n in ranked:
            print(f"  ({n['x']:.0f}, {n['y']:.0f}) "
                  f"score={n['score']:.2f} tri={n['tri']:.2f} "
                  f"fill={n['fill']:.2f}")
        print(f"circle/square fill threshold: {thr:.2f}")

    return thr


def set_start(nodes, new_idx, thr):
    """Move the S label to another node."""
    for n in nodes:
        if n["type"] == "S":
            n["type"] = "H" if n["fill"] >= thr else "P"

    nodes[new_idx]["type"] = "S"


# ----------------------------------------------------------------------
# Graph building
# ----------------------------------------------------------------------

def line_exists(line_map, a, b, samples=24):
    """
    line_map is the (dilated) darkness map. A connector exists if the
    middle part of the segment is consistently darker than the page.
    """
    hits = 0
    total = 0

    for t in np.linspace(0.3, 0.7, samples):
        x = int(a["x"] + t * (b["x"] - a["x"]))
        y = int(a["y"] + t * (b["y"] - a["y"]))

        if 0 <= x < line_map.shape[1] and 0 <= y < line_map.shape[0]:
            total += 1
            if line_map[y, x] > LINE_THRESH:
                hits += 1

    return total > 0 and hits >= 0.7 * total


def build_graph(nodes, board, spacing):
    if len(nodes) < 2:
        return {}

    # connector lines: faint and thin; tolerate small positional error
    line_map = darkness(board, blur=5)
    line_map = cv2.dilate(line_map, np.ones((5, 5), np.uint8))

    adj = {i: [] for i in range(len(nodes))}

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            d = math.hypot(
                nodes[i]["x"] - nodes[j]["x"],
                nodes[i]["y"] - nodes[j]["y"],
            )

            # direct grid neighbours only (rejects diagonals and far nodes)
            if not (0.7 * spacing <= d <= 1.3 * spacing):
                continue

            if line_exists(line_map, nodes[i], nodes[j]):
                adj[i].append(j)
                adj[j].append(i)

    edges = sum(len(v) for v in adj.values()) // 2
    print(f"grid spacing: {spacing:.0f}px, edges found: {edges}")

    return adj


# ----------------------------------------------------------------------
# Walking the chain and producing F / L / R + H / P
# ----------------------------------------------------------------------

def _turn(v_prev, v_next):
    """
    F = same direction as the previous step
    L / R = left / right turn relative to the previous step
    Image y points down, so a positive cross product is a right turn.
    """
    cross = v_prev[0] * v_next[1] - v_prev[1] * v_next[0]
    dot = v_prev[0] * v_next[0] + v_prev[1] * v_next[1]

    ang = math.degrees(math.atan2(cross, dot))

    if abs(ang) < 45:
        return "F"
    if abs(ang) > 135:
        return "B"      # should not happen on a simple chain

    return "R" if ang > 0 else "L"


def walk_chain(nodes, adj):
    """
    Walks from the start (triangle) node along the chain.

    Returns (steps, path) where steps is a list with one entry per node
    AFTER the start: (move, type)
        move = how we turned to arrive at that node
               (F = kept the previous direction, L, R)
        type = "H" (circle) or "P" (square)

    The first step has no previous direction, so it is always "F".
    """
    start = next(
        (i for i, n in enumerate(nodes) if n["type"] == "S"),
        None,
    )

    if start is None:
        return None, []

    path = [start]
    visited = {start}
    current = start

    while True:
        nxt = [j for j in adj[current] if j not in visited]

        if not nxt:
            break

        if len(nxt) > 1:
            nxt.sort(key=lambda j: len(adj[j]))

        current = nxt[0]
        path.append(current)
        visited.add(current)

    steps = []
    prev_vec = None

    for k in range(1, len(path)):
        a = nodes[path[k - 1]]
        b = nodes[path[k]]

        vec = (b["x"] - a["x"], b["y"] - a["y"])

        move = "F" if prev_vec is None else _turn(prev_vec, vec)

        steps.append((move, b["type"]))
        prev_vec = vec

    return steps, path


# ----------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------

def parse_frame(frame, debug=False):
    """
    camera/image -> (optional board outline) -> upscale -> detect shapes
    -> spacing -> keep main cluster -> pick start -> graph -> walk chain

    Returns (steps, nodes, path, board). board is never None.
    """
    if RAW_MODE:
        print("RAW mode: using the whole image as the board")
        board = frame.copy()
    else:
        board, _ = find_board(frame)
        if board is None:
            print("No board outline found, using the whole image as the board")
            board = frame.copy()

    board = normalise(board)

    nodes = detect_shapes(board, debug=debug)

    if not nodes:
        print("FAILED: no nodes detected (lower COARSE_THRESH / CORE_MIN)")
        return None, nodes, [], board

    spacing = estimate_spacing(nodes)

    if spacing is None:
        print("FAILED: not enough nodes to estimate spacing")
        return None, nodes, [], board

    before = len(nodes)
    nodes = dedupe(nodes, spacing)

    # discard everything outside the graph, then re-measure the spacing
    nodes = keep_main_cluster(nodes, spacing)
    spacing = estimate_spacing(nodes) or spacing

    thr = assign_types(nodes, debug=debug)

    counts = {t: sum(1 for n in nodes if n["type"] == t) for t in "HPS"}
    print(f"Nodes detected: {len(nodes)} (discarded {before - len(nodes)} "
          f"duplicates/outliers) types: {counts}")

    adj = build_graph(nodes, board, spacing)

    if not any(adj.values()):
        print("FAILED: no edges found (lower LINE_THRESH)")
        return None, nodes, [], board

    start_idx = next(i for i, n in enumerate(nodes) if n["type"] == "S")

    # The start must be the end of a chain. If the chosen node is not
    # connected to anything, use the most triangle-like chain end instead.
    if not adj[start_idx]:
        ends = [i for i in adj if len(adj[i]) == 1]

        print(f"Start candidate at ({nodes[start_idx]['x']:.0f}, "
              f"{nodes[start_idx]['y']:.0f}) has no connector; "
              f"trying {len(ends)} chain ends")

        if ends:
            start_idx = max(ends, key=lambda i: nodes[i]["score"])
            set_start(nodes, start_idx, thr)

    if debug:
        s = nodes[start_idx]
        print(f"start node at ({s['x']:.0f}, {s['y']:.0f}), "
              f"neighbours: {len(adj[start_idx])}")

        for j, n in enumerate(nodes):
            if j == start_idx:
                continue
            d = math.hypot(n["x"] - s["x"], n["y"] - s["y"])
            if d < 1.6 * spacing:
                print(f"  near start: {n['type']} at "
                      f"({n['x']:.0f}, {n['y']:.0f}) dist={d:.0f} "
                      f"(connected={j in adj[start_idx]})")

    steps, path = walk_chain(nodes, adj)

    if not steps:
        print("FAILED: start node has no connected neighbour "
              "(see 'near start' lines above)")
    else:
        print(f"Walked {len(path)} nodes out of {len(nodes)} detected")

    return steps, nodes, path, board


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------

def draw_overlay(image, nodes, path):
    out = image.copy()

    for i in range(len(path) - 1):
        a = nodes[path[i]]
        b = nodes[path[i + 1]]

        cv2.line(
            out,
            (int(a["x"]), int(a["y"])),
            (int(b["x"]), int(b["y"])),
            (0, 255, 255),
            3,
        )

    for node in nodes:
        colour = COLOURS.get(node["type"], (255, 255, 255))

        cv2.circle(
            out,
            (int(node["x"]), int(node["y"])),
            int(max(5, node["r"])),
            colour,
            2,
        )

        cv2.putText(
            out,
            str(node["type"]),
            (int(node["x"]) - 8, int(node["y"]) - int(node["r"]) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
        )

    return out


def fit_for_screen(img, max_side=1000):
    h, w = img.shape[:2]
    s = max_side / float(max(h, w))
    if s >= 1:
        return img
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def report(steps):
    print()

    if not steps:
        print("No path found (see the FAILED line above for the reason)")
        print()
        return

    """
    print("Path (move, type):")
    for i, (move, node_type) in enumerate(steps, 1):
        print(f"  {i:2d}: {move} {node_type}")
    """

    print()
    print("Compact:", " ".join(f"{m}{t}" for m, t in steps))
    print("Moves:  ", "".join(m for m, _ in steps))
    print("Types:  ", "".join(t for _, t in steps))
    print()


# ----------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------

def image(source, show=False):
    frame = cv2.imread(source)

    if frame is None:
        raise FileNotFoundError(source)

    steps, nodes, path, board = parse_frame(frame, debug=False)

    report(steps)

    overlay = draw_overlay(board, nodes, path)
    cv2.imwrite("debug_overlay.png", overlay)

    if show:
        cv2.imshow("Parser Output", fit_for_screen(overlay))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return steps


def snapshot(camera=0):
    cap = cv2.VideoCapture(camera)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    try:
        for _ in range(30):
            cap.read()

        ret, frame = cap.read()

        if not ret:
            raise RuntimeError("Could not capture image")

        steps, nodes, path, board = parse_frame(frame, debug=False)

        report(steps)

        overlay = draw_overlay(board, nodes, path)
        cv2.imshow("Captured Image", fit_for_screen(overlay))
        cv2.waitKey(0)

        return steps

    finally:
        cap.release()
        cv2.destroyAllWindows()


def video(camera=0):
    cap = cv2.VideoCapture(camera)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print()
    print("SPACE = capture")
    print("Q     = quit")
    print()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            display = frame.copy()

            _, corners = find_board(frame)

            if corners is not None:
                cv2.polylines(
                    display,
                    [corners.astype(np.int32)],
                    True,
                    (0, 255, 255),
                    3,
                )

            cv2.imshow("Graph Scanner", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                steps, nodes, path, board = parse_frame(frame, debug=False)

                report(steps)

                overlay = draw_overlay(board, nodes, path)
                cv2.imshow("Detected Graph", fit_for_screen(overlay))
                cv2.waitKey(0)

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = sys.argv[1:]

    if "--video" in args:
        video()
        return

    if "--snapshot" in args:
        snapshot()
        return

    files = [a for a in args if not a.startswith("--")]

    if files:
        image(files[0], show=True)
        return

    print(
        "Usage:\n"
        "  python graph_scanner.py image.png [--raw]\n"
        "  python graph_scanner.py --video\n"
        "  python graph_scanner.py --snapshot\n"
        "\n"
        "  --raw  skip board-outline detection (use for already-cropped images)"
    )


if __name__ == "__main__":
    main()