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

RAW_MODE = "--raw" in sys.argv     # skip page isolation (already-cropped input)

TARGET_SIZE = 1600      # board is rescaled so its longest side is this
COARSE_THRESH = 20      # pass-1 darkness threshold
CORE_MIN = 12           # minimum node contrast in refine_blob
LINE_THRESH = 8         # connector line darkness threshold
N_CIRCLES = 30          # exactly this many H (circle) nodes; the rest are P

# Weights for the circle-likeness ranking. Colour and ink are only used
# when the image actually contains that signal (checked automatically).
W_SHAPE = 1.0           # geometry (always used)
W_COLOUR = 3.0          # saturated (red) vs black
W_INK = 1.0             # circles are lighter than black squares in grey


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


def isolate_page(frame):
    """
    Find the bright page (paper, or the white picture on a monitor) and
    blank everything outside it. Without this, a dark screen bezel forms
    one big ring-shaped blob that hides every node inside it.

    The page is the largest bright region (Otsu); its convex hull fills
    in the nodes, which are dark holes. Geometry is NOT changed, so turn
    directions are unaffected. Returns the cropped, blanked image.
    """
    h, w = frame.shape[:2]
    s = min(1.0, 640.0 / max(h, w))

    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    _, bright = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_CLOSE, _ellipse(max(3, int(max(g.shape) * 0.02)) | 1)
    )

    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return frame

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.02 * g.shape[0] * g.shape[1]:
        return frame                      # no obvious page, use everything

    small_mask = np.zeros(g.shape, np.uint8)
    cv2.fillConvexPoly(small_mask, cv2.convexHull(c), 255)

    mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # drop the blurry page edge / bezel rim
    bx, by, bw, bh = cv2.boundingRect(mask)
    m = max(3, int(0.015 * min(bw, bh))) | 1
    mask = cv2.erode(mask, _ellipse(m))

    inside = mask > 0
    if not inside.any():
        return frame

    paper = np.percentile(frame[inside][::37], 90, axis=0).astype(np.uint8)

    out = frame.copy()
    out[~inside] = paper

    x, y, bw, bh = cv2.boundingRect(mask)
    return out[y:y + bh, x:x + bw]



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


def coarse_threshold(diff):
    """
    Pass-1 threshold, relative to how dark the nodes actually are.
    A fixed low value keeps the faint blurred halo around thin connector
    lines, which then fuses neighbouring nodes into one blob.
    """
    return max(float(COARSE_THRESH), 0.4 * float(np.percentile(diff, 99)))


def estimate_node_radius(diff, thr):
    """
    Node radius from the picture itself (distance transform): nodes are
    the fat parts of the dark mask, connectors are thin ridges. This is
    independent of how much of the camera frame the board fills.
    """
    mask = (diff > thr).astype(np.uint8)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    dmax = float(dt.max())
    if dmax < 2.0:
        return None

    peaks = (dt == cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt >= 0.5 * dmax)
    return float(np.median(dt[peaks]))


def node_kernel(r):
    """Wider than a (blurred) connector, clearly smaller than a node."""
    return _ellipse(max(3, int(round(0.75 * r))) | 1)


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


def node_colour(board, diff, contour):
    """
    Median colourfulness and darkness of the INSIDE of a node (the blob
    is eroded first so edge pixels and connector stubs are ignored).

      chroma = max(B,G,R) - min(B,G,R), scaled 0..1
               black ~0, red / blue ~1, unaffected by perspective
      ink    = darkness vs the page
               black ~255, red ~180 (red is lighter than black in grey)
    """
    x, y, w, h = cv2.boundingRect(contour)

    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(
        mask, [contour - np.array([x, y], dtype=contour.dtype)], -1, 255, -1
    )

    k = max(3, int(min(w, h) * 0.25)) | 1
    inner = cv2.erode(mask, _ellipse(k))
    if cv2.countNonZero(inner) < 10:
        inner = mask

    sel = inner > 0

    patch = board[y:y + h, x:x + w].astype(np.int16)
    chroma = (patch.max(axis=2) - patch.min(axis=2))[sel]
    ink = diff[y:y + h, x:x + w][sel]

    return float(np.median(chroma)) / 255.0, float(np.median(ink))


def shape_metrics(c):
    """
    Three rotation- and blur-tolerant shape measures:

      fill = area / area of minimum enclosing circle
             circle ~0.9-1.0, square ~0.64-0.75, triangle ~0.41-0.55

      tri  = area / area of minimum enclosing triangle
             triangle ~0.8-1.0, circle ~0.6, square ~0.5

      rect = area / area of minimum-area bounding rectangle
             square ~1.0, circle ~0.785, triangle ~0.5

    A triangle is the shape with high tri AND low fill.
    Circle-vs-square is best told apart by fill - rect (see "round").
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

    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    rect = area / (rw * rh) if rw * rh > 0 else 0.0

    return fill, min(tri, 1.2), min(rect, 1.0)


def split_peaks(contour, r_node):
    """
    Centres of the separate nodes inside a blob that is too big to be one
    node (blur fused a run of connected nodes). Nodes are the fat parts of
    the blob: peaks of its distance transform, kept at least ~1.3 radii
    apart.
    """
    x, y, w, h = cv2.boundingRect(contour)

    m = np.zeros((h, w), np.uint8)
    cv2.drawContours(
        m, [contour - np.array([x, y], dtype=contour.dtype)], -1, 255, -1
    )

    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    is_peak = (dt == cv2.dilate(dt, np.ones((5, 5), np.uint8))) & (dt >= 0.6 * r_node)

    cand = sorted(
        (tuple(p) for p in np.argwhere(is_peak)),
        key=lambda p: -dt[p[0], p[1]],
    )

    kept = []
    for py, px in cand:
        if all(math.hypot(px - qx, py - qy) > 1.3 * r_node for qx, qy in kept):
            kept.append((px, py))

    return [(x + px, y + py) for px, py in kept]


def detect_shapes(board, debug=False):
    diff = darkness(board)
    h, w = diff.shape

    thr = coarse_threshold(diff)
    r_node = estimate_node_radius(diff, thr)

    if r_node is None:
        return []

    # wider than the connector lines (so they are removed), smaller than a node
    kernel = node_kernel(r_node)

    # ---- pass 1: coarse blobs --------------------------------------
    coarse = (diff > thr).astype(np.uint8) * 255
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

    min_area = 0.2 * math.pi * r_node * r_node

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
            f"board {w}x{h}, node radius {r_node:.0f}px, "
            f"threshold {thr:.0f}, kernel {kernel.shape[0]}px, "
            f"coarse blobs: {len(blobs)} "
            f"(contours before filtering: {len(contours)}), "
            f"max darkness: {diff.max()}"
        )

    if not blobs:
        return []

    # ---- pass 2: refine each node and measure its shape ------------
    nodes = []

    node_area = math.pi * r_node * r_node

    # one (cx, cy, r) seed per node; fused blobs are split into several
    seeds = []

    for a, c in blobs:
        if a < 0.25 * node_area:
            if debug:
                print(f"skipped small blob area={a:.0f} (node ~{node_area:.0f})")
            continue

        if a > 1.8 * node_area:
            peaks = split_peaks(c, r_node)
            if debug:
                print(f"split blob area={a:.0f} into {len(peaks)} nodes")
            seeds.extend((px, py, r_node) for px, py in peaks)
            continue

        m = cv2.moments(c)
        if m["m00"] == 0:
            continue

        seeds.append((m["m10"] / m["m00"], m["m01"] / m["m00"],
                      math.sqrt(a / math.pi)))

    for cx, cy, r in seeds:
        refined = refine_blob(diff, cx, cy, r, kernel)

        if refined is None:
            if debug:
                print(f"refine failed at ({cx:.0f}, {cy:.0f}) r={r:.0f}")
            continue

        metrics = shape_metrics(refined)
        if metrics is None:
            continue

        fill, tri, rect = metrics

        rm = cv2.moments(refined)
        if rm["m00"] == 0:
            continue

        rarea = cv2.contourArea(refined)
        chroma, ink = node_colour(board, diff, refined)

        nodes.append(
            {
                "type": None,
                "fill": fill,
                "tri": tri,
                "rect": rect,
                "chroma": chroma,
                "ink": ink,
                "round": fill - rect,         # high = circle-like
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
# Typing: start (triangle) first, then exactly N_CIRCLES circles (H)
# ----------------------------------------------------------------------

def _norm01(vals):
    """Scale to 0..1 between the 5th and 95th percentile; also return the spread."""
    vals = np.asarray(vals, dtype=np.float64)
    lo, hi = np.percentile(vals, 5), np.percentile(vals, 95)

    if hi - lo < 1e-9:
        return np.zeros_like(vals), 0.0

    return np.clip((vals - lo) / (hi - lo), 0.0, 1.0), float(hi - lo)


def circle_scores(nodes, idx):
    """
    Combined circle-likeness (0..1, higher = more circle-like) for the
    nodes in idx. Each cue is only used if the image really contains it:

      shape  : fill - rect          (always)
      colour : saturated vs black   (only if the nodes differ in chroma)
      ink    : lighter than black   (only if the nodes differ in darkness)
    """
    shape, _ = _norm01([nodes[i]["round"] for i in idx])
    score = W_SHAPE * shape
    total = W_SHAPE
    used = ["shape"]

    chroma, c_spread = _norm01([nodes[i]["chroma"] for i in idx])
    if c_spread > 0.25:
        score = score + W_COLOUR * chroma
        total += W_COLOUR
        used.append("colour")

    ink_raw = np.array([nodes[i]["ink"] for i in idx])
    ink, i_spread = _norm01(ink_raw)
    if i_spread > 0.20 * max(float(np.percentile(ink_raw, 95)), 1e-9):
        score = score + W_INK * (1.0 - ink)
        total += W_INK
        used.append("ink")

    return score / total, used


def assign_types_on_path(nodes, path, debug=False):
    """
    Typing is done AFTER the chain has been walked, and only on the
    nodes that are really on it. Known facts about the encoder are used:

      * path[0] is the triangle (S)
      * the node right after S is ALWAYS a P (main.py hard-codes it)
      * the chain ALWAYS ends on an H (the fold only finishes when the
        last turn has been used, and a turn is consumed by an H)
      * exactly N_CIRCLES of the remaining nodes are H

    The rest are ranked by circle-likeness (shape, colour, ink).
    """
    for n in nodes:
        n["type"] = "?"                     # off the chain

    start = path[0]
    nodes[start]["type"] = "S"

    idx = path[1:]
    if not idx:
        return

    score, used = circle_scores(nodes, idx)
    score = np.array(score, dtype=np.float64)

    score[0] = -1.0         # first node after S is a P
    score[-1] = 2.0         # last node of the chain is an H

    order = np.argsort(-score, kind="stable")

    for rank, k in enumerate(order):
        node = nodes[idx[int(k)]]
        node["circ"] = float(score[int(k)])
        node["type"] = "H" if rank < N_CIRCLES else "P"

    print("circle/square cues used: " + ", ".join(used))

    if len(idx) < N_CIRCLES:
        print(f"WARNING: only {len(idx)} nodes on the chain after the start, "
              f"expected at least {N_CIRCLES} circles")

    if len(idx) > N_CIRCLES:
        hi = nodes[idx[int(order[N_CIRCLES - 1])]]["circ"]
        lo = nodes[idx[int(order[N_CIRCLES])]]["circ"]

        if debug or hi - lo < 0.15:
            print(f"circle/square boundary: {N_CIRCLES}th={hi:.2f}, "
                  f"{N_CIRCLES + 1}th={lo:.2f}, gap={hi - lo:.2f}  "
                  f"(small gap = unreliable)")


# ----------------------------------------------------------------------
# Graph building
# ----------------------------------------------------------------------
#
# The picture is ONE self-avoiding chain: every node has at most two
# connectors, there are no cycles, and the chain has exactly two ends
# (the triangle start and the final H). Edges are therefore chosen
# greedily by line strength, refusing any edge that would give a node a
# third connector or close a loop. That rejects false edges between
# lattice-neighbours that are not connected, and lets a faint real
# connector be recovered when it is needed to join two pieces of chain.

def edge_strength(line_map, a, b, samples=24):
    """
    How strongly a connector line is present between two nodes: a low
    percentile of the darkness along the middle of the segment, so the
    line has to be there along nearly the whole length.
    """
    vals = []

    for t in np.linspace(0.3, 0.7, samples):
        x = int(a["x"] + t * (b["x"] - a["x"]))
        y = int(a["y"] + t * (b["y"] - a["y"]))

        if 0 <= x < line_map.shape[1] and 0 <= y < line_map.shape[0]:
            vals.append(float(line_map[y, x]))

    return float(np.percentile(vals, 20)) if vals else 0.0


def build_graph(nodes, board, spacing):
    n = len(nodes)
    adj = {i: [] for i in range(n)}

    if n < 2:
        return adj

    # connector lines: faint and thin; tolerate small positional error
    line_map = darkness(board, blur=5)
    line_map = cv2.dilate(line_map, np.ones((5, 5), np.uint8))

    cands = []
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(
                nodes[i]["x"] - nodes[j]["x"],
                nodes[i]["y"] - nodes[j]["y"],
            )

            # direct grid neighbours only (rejects diagonals and far nodes)
            if 0.7 * spacing <= d <= 1.3 * spacing:
                cands.append((edge_strength(line_map, nodes[i], nodes[j]), i, j))

    cands.sort(reverse=True)

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def try_add(i, j):
        if len(adj[i]) >= 2 or len(adj[j]) >= 2:
            return False
        ri, rj = find(i), find(j)
        if ri == rj:
            return False                      # would close a loop
        parent[ri] = rj
        adj[i].append(j)
        adj[j].append(i)
        return True

    for s, i, j in cands:                     # clear connectors
        if s >= LINE_THRESH:
            try_add(i, j)

    for s, i, j in cands:                     # faint ones, only to join pieces
        if s >= 0.5 * LINE_THRESH:
            try_add(i, j)

    edges = sum(len(v) for v in adj.values()) // 2
    pieces = len({find(i) for i in range(n)})
    print(f"grid spacing: {spacing:.0f}px, edges found: {edges}, "
          f"separate pieces: {pieces}")

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


def walk_chain(adj, start):
    """Indices of the nodes in chain order, beginning at start."""
    path = [start]
    visited = {start}
    current = start

    while True:
        nxt = [j for j in adj[current] if j not in visited]
        if not nxt:
            break
        current = nxt[0]
        path.append(current)
        visited.add(current)

    return path


def path_to_steps(nodes, path):
    """
    One entry per node AFTER the start: (move, type)
        move = how the chain turned to arrive at that node
               (F = kept the previous direction, L, R)
        type = "H" (circle) or "P" (square)

    This matches the encoder: an H is placed by a turn relative to the
    direction of the previous step, whether that step ended on a P or H.
    The very first step has no previous direction (always "F"); it goes
    to the forced P, so it never carries information.
    """
    steps = []
    prev_vec = None

    for k in range(1, len(path)):
        a = nodes[path[k - 1]]
        b = nodes[path[k]]

        vec = (b["x"] - a["x"], b["y"] - a["y"])
        move = "F" if prev_vec is None else _turn(prev_vec, vec)

        steps.append((move, b["type"]))
        prev_vec = vec

    return steps


# ----------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------

def parse_frame(frame, debug=False):
    """
    camera/image -> isolate the page -> upscale -> detect shapes
    -> spacing -> main cluster -> connector graph (one chain)
    -> start = triangle-like chain end -> walk -> type the chain nodes

    Returns (steps, nodes, path, board). board is never None.
    """
    board = frame.copy() if RAW_MODE else isolate_page(frame)
    board = normalise(board)

    nodes = detect_shapes(board, debug=debug)

    if not nodes:
        print("FAILED: no nodes detected")
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

    print(f"Nodes detected: {len(nodes)} (discarded {before - len(nodes)} "
          f"duplicates/outliers)")

    adj = build_graph(nodes, board, spacing)

    if not any(adj.values()):
        for n in nodes:
            n["type"] = "?"
        print("FAILED: no edges found (lower LINE_THRESH)")
        return None, nodes, [], board

    # The chain has two ends: the triangle start and the final H.
    # Of the ends, the more triangle-like one is the start.
    ends = [i for i in adj if len(adj[i]) == 1]

    if len(ends) != 2:
        print(f"WARNING: expected 2 chain ends, found {len(ends)} "
              "(a connector is missing or extra)")

    pool = ends if ends else list(adj)
    start = max(pool, key=lambda i: nodes[i]["score"])

    path = walk_chain(adj, start)

    if len(path) < len(nodes):
        print(f"WARNING: chain covers {len(path)} of {len(nodes)} nodes")

    assign_types_on_path(nodes, path, debug=debug)

    counts = {t: sum(1 for n in nodes if n["type"] == t) for t in "HPS?"}
    print(f"types: {counts}")

    steps = path_to_steps(nodes, path)

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

                if steps:
                    return steps      # success: hand the result to the caller

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
        "  --raw  skip page isolation (use for already-cropped images)"
    )


if __name__ == "__main__":
    main()