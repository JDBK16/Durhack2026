"""
Parse 2D lattice protein diagrams (HP model) from images or a live camera.

Conventions:
    grey triangle -> 'S' (start)
    red circle    -> 'H'
    white square  -> 'P'
Shapes are joined by grey lines. Starting at the triangle, the chain is walked
shape by shape. Each entry in the output is (state, direction of the move OUT
of that shape): 'strait', 'left' or 'right'. The first move counts as 'strait'.
The last shape has no outgoing move, so its direction is None.

Usage:
    python lattice_parser.py image.png        # parse a still image file
    python lattice_parser.py --snapshot       # grab one camera frame and parse it
    python lattice_parser.py --video          # live camera; SPACE = accept, q = quit
"""
import sys
import math
import cv2
import numpy as np


# --------------------------------------------------------------------------
# 1. Shape detection
# --------------------------------------------------------------------------
def _ink_mask(gray):
    """Binary mask of 'dark' pixels (outlines, connecting lines, triangle).
    The local paper brightness is estimated with a large max-filter and each
    pixel is compared to it, so shadows / uneven lighting in photos don't
    matter and filled shapes (the triangle) stay solid."""
    k = max(15, min(gray.shape) // 12) | 1
    bg = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.GaussianBlur(bg, (k, k), 0).astype(np.float32) + 1
    norm = gray.astype(np.float32) / bg
    return ((norm < 0.78) * 255).astype(np.uint8)


def _find_circles(hsv):
    """Red filled circles -> list of (cx, cy, radius)."""
    low = cv2.inRange(hsv, (0, 70, 50), (12, 255, 255))
    high = cv2.inRange(hsv, (165, 70, 50), (180, 255, 255))
    red = cv2.morphologyEx(low | high, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if area / (math.pi * r * r) > 0.6:                 # reasonably round
            m = cv2.moments(c)
            out.append((m["m10"] / m["m00"], m["m01"] / m["m00"], r))
    # discard outliers in size (stray red objects in a photo)
    if out:
        med = np.median([o[2] for o in out])
        out = [o for o in out if 0.5 * med < o[2] < 2.0 * med]
    return out


def _find_squares(ink, ref_area):
    """White squares = enclosed light regions (holes in the ink mask) that are
    roughly square and roughly the same size as the circles."""
    contours, hier = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hier is None:
        return out
    for c, h in zip(contours, hier[0]):
        if h[3] < 0:                                       # want holes only
            continue
        area = cv2.contourArea(c)
        if ref_area and not (0.3 * ref_area < area < 3.0 * ref_area):
            continue
        if not ref_area and area < 50:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        rect = cv2.minAreaRect(c)
        w, hgt = rect[1]
        if len(approx) == 4 and w > 0 and hgt > 0 and 0.6 < w / hgt < 1.67 \
                and area / (w * hgt) > 0.75:
            m = cv2.moments(c)
            out.append((m["m10"] / m["m00"], m["m01"] / m["m00"], math.sqrt(area) / 2))
    if out and not ref_area:                               # no circles to compare to
        med = np.median([o[2] for o in out])
        out = [o for o in out if 0.6 * med < o[2] < 1.6 * med]
    return out


def _triangle_candidates(ink, sat, ref_r):
    """Solid dark-grey blobs that could be the start triangle.
    Red pixels are removed, then an opening sized to the shapes strips away
    lines, outlines and text strokes; only solid filled areas survive. The
    final choice (in build_graph) is the candidate bonded to a chain end, so
    stray dark blobs elsewhere in the photo are ignored."""
    grey = cv2.bitwise_and(ink, (sat < 90).astype(np.uint8) * 255)
    k = max(3, int(ref_r * 0.8)) if ref_r else 7
    solid = cv2.morphologyEx(grey, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 15 or (ref_r and not (0.4 * ref_r ** 2 < area < 10 * ref_r ** 2)):
            continue
        tri_area, _ = cv2.minEnclosingTriangle(c)
        _, (w, h), _ = cv2.minAreaRect(c)
        # triangle: fill_tri high, fill_rect ~0.5-0.7; circle 0.6/0.79; square 0.5/1
        score = area / max(tri_area, 1) - area / max(w * h, 1)
        m = cv2.moments(c)
        out.append(dict(type="S", x=m["m10"] / m["m00"], y=m["m01"] / m["m00"],
                        r=math.sqrt(area) * 0.75, score=score))
    return out


def detect_shapes(bgr):
    """Return (nodes, triangle_candidates, ink_mask). Nodes are dicts with
    type 'H' or 'P', centre x, y and approximate radius r."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ink = _ink_mask(gray)

    circles = _find_circles(hsv)
    ref_r = float(np.median([c[2] for c in circles])) if circles else None
    ref_area = (2 * ref_r) ** 2 if ref_r else None         # square ~ same width

    squares = _find_squares(ink, ref_area)
    if ref_r is None and squares:
        ref_r = float(np.median([s[2] for s in squares]))

    nodes = [dict(type="H", x=x, y=y, r=r) for x, y, r in circles]
    nodes += [dict(type="P", x=x, y=y, r=r) for x, y, r in squares]
    tris = _triangle_candidates(ink, hsv[..., 1], ref_r)
    return nodes, tris, ink


# --------------------------------------------------------------------------
# 2. Bond detection (is there a line between two neighbouring shapes?)
# --------------------------------------------------------------------------
def _bonded(ink, a, b, min_fill=0.7):
    """Sample the ink mask along the gap between two shapes' edges."""
    ax, ay, bx, by = a["x"], a["y"], b["x"], b["y"]
    d = math.hypot(bx - ax, by - ay)
    t0 = (a["r"] * 1.15) / d                                # start just past shape a
    t1 = 1 - (b["r"] * 1.15) / d                            # stop just before shape b
    if t1 - t0 < 0.05:                                      # shapes nearly touching
        t0, t1 = 0.45, 0.55
    n = 15
    hits = 0
    tol = max(2, int(0.3 * min(a["r"], b["r"])))
    h, w = ink.shape
    for t in np.linspace(t0, t1, n):
        x, y = ax + t * (bx - ax), ay + t * (by - ay)
        xi, yi = int(round(x)), int(round(y))
        # look in a tiny perpendicular window to tolerate slight misalignment
        px, py = -(by - ay) / d, (bx - ax) / d
        found = False
        for s in range(-tol, tol + 1):
            xx, yy = int(round(x + s * px)), int(round(y + s * py))
            if 0 <= xx < w and 0 <= yy < h and ink[yy, xx]:
                found = True
                break
        hits += found
    return hits / n >= min_fill


def build_graph(nodes, tris, ink):
    """Bond neighbouring H/P nodes, then attach the start triangle.
    Returns (nodes_with_S, adjacency, lattice_spacing)."""
    if len(nodes) < 1:
        return nodes, {}, None
    pts = np.array([[n["x"], n["y"]] for n in nodes])
    if len(nodes) > 1:
        dists = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        np.fill_diagonal(dists, np.inf)
        nn = dists.min(axis=1)                              # local step size per node
        spacing = float(np.median(nn))                      # (varies under perspective)
    else:
        dists, spacing = np.zeros((1, 1)), 3 * nodes[0]["r"]
        nn = np.array([spacing])
    adj = {i: [] for i in range(len(nodes))}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            # direct lattice neighbours only; diagonals (~1.41x) are also
            # rejected by the bond check because their midpoint is blank paper
            if dists[i, j] < 1.3 * max(nn[i], nn[j]) and _bonded(ink, nodes[i], nodes[j]):
                adj[i].append(j)
                adj[j].append(i)

    # The start triangle must be bonded to a chain end (a node with <= 1 bond).
    best = None
    for t in tris:
        for i, n in enumerate(nodes):
            if len(adj[i]) > 1:
                continue
            d = math.hypot(t["x"] - n["x"], t["y"] - n["y"])
            if 0.6 * nn[i] < d < 1.5 * nn[i] and _bonded(ink, t, n):
                if best is None or t["score"] > best[0]["score"]:
                    best = (t, i)
    if best is None:
        return nodes, adj, spacing
    s_idx = len(nodes)
    nodes = nodes + [best[0]]
    adj[s_idx] = [best[1]]
    adj[best[1]].append(s_idx)
    return nodes, adj, spacing


# --------------------------------------------------------------------------
# 3. Walk the chain and classify turns
# --------------------------------------------------------------------------
def _turn(v_prev, v_next):
    """Image coords (y down): negative cross product = left turn."""
    cross = v_prev[0] * v_next[1] - v_prev[1] * v_next[0]
    dot = v_prev[0] * v_next[0] + v_prev[1] * v_next[1]
    ang = math.degrees(math.atan2(cross, dot))
    if abs(ang) < 45:
        return "strait"
    if abs(ang) > 135:
        return "back"                                       # shouldn't happen on a valid chain
    return "right" if ang > 0 else "left"


def walk_chain(nodes, adj):
    start = next((i for i, n in enumerate(nodes) if n["type"] == "S"), None)
    if start is None:
        return None, []
    path, visited = [start], {start}
    cur = start
    while True:
        nxt = [j for j in adj[cur] if j not in visited]
        if not nxt:
            break
        if len(nxt) > 1 and len(path) >= 2:
            # ambiguous branch: prefer the neighbour not adjacent to earlier path
            nxt.sort(key=lambda j: len(adj[j]))
        cur = nxt[0]
        path.append(cur)
        visited.add(cur)

    result = []
    prev_vec = None
    for k, idx in enumerate(path):
        if k == len(path) - 1:
            result.append((nodes[idx]["type"], None))       # end of chain
            break
        nxt = nodes[path[k + 1]]
        vec = (nxt["x"] - nodes[idx]["x"], nxt["y"] - nodes[idx]["y"])
        direction = "strait" if prev_vec is None else _turn(prev_vec, vec)
        result.append((nodes[idx]["type"], direction))
        prev_vec = vec
    return result, path


def parse_frame(bgr):
    """Full pipeline on one BGR image. Returns (result, nodes, path)."""
    nodes, tris, ink = detect_shapes(bgr)
    nodes, adj, _ = build_graph(nodes, tris, ink)
    if not adj:
        return None, nodes, []
    result, path = walk_chain(nodes, adj)
    return result, nodes, path


# --------------------------------------------------------------------------
# 4. Visualisation + entry points
# --------------------------------------------------------------------------
COLOURS = {"S": (0, 200, 0), "H": (255, 0, 0), "P": (0, 165, 255)}


def draw_overlay(bgr, nodes, path, result):
    out = bgr.copy()
    for k in range(len(path) - 1):
        a, b = nodes[path[k]], nodes[path[k + 1]]
        cv2.line(out, (int(a["x"]), int(a["y"])), (int(b["x"]), int(b["y"])), (0, 255, 0), 2)
    for n in nodes:
        cv2.circle(out, (int(n["x"]), int(n["y"])), int(n["r"]) + 3, COLOURS[n["type"]], 2)
    if result:
        for k, idx in enumerate(path):
            n = nodes[idx]
            cv2.putText(out, str(k), (int(n["x"]) - 6, int(n["y"]) - int(n["r"]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        txt = "".join(s for s, _ in result)
        cv2.putText(out, f"{len(result)} shapes: {txt}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


def image(source=None, show=False):
    """Parse a still image. `source` may be a file path, a BGR array, or None
    to grab a single frame from the default camera."""
    if source is None:
        cap = cv2.VideoCapture(0)
        for _ in range(5):                                  # let auto-exposure settle
            ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Could not read from camera")
    elif isinstance(source, str):
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(source)
    else:
        frame = source
    result, nodes, path = parse_frame(frame)
    if show:
        cv2.imshow("parsed", draw_overlay(frame, nodes, path, result))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return result

def centre_square(frame, size_ratio=0.6):
    """
    Return a centred square ROI and its coordinates.
    size_ratio=0.6 means the square occupies 60% of the smaller frame dimension.
    """
    h, w = frame.shape[:2]

    side = int(min(h, w) * size_ratio)

    x0 = (w - side) // 2
    y0 = (h - side) // 2

    roi = frame[y0:y0 + side, x0:x0 + side]

    return roi, (x0, y0, side)

def video(camera=0, stable_frames=8):
    """
    Live camera. Only the central square region is analysed.
    SPACE accepts a stable reading.
    q quits.
    """

    vid = cv2.VideoCapture(camera)

    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not vid.isOpened():
        raise RuntimeError("Could not open camera")

    # Let exposure/focus settle
    for _ in range(30):
        vid.read()

    last = None
    count = 0
    accepted = None

    try:
        while True:

            ret, frame = vid.read()
            if not ret:
                break

            roi, (x0, y0, side) = centre_square(frame, 0.6)

            try:
                result, nodes, path = parse_frame(roi)
            except Exception:
                result, nodes, path = None, [], []

            if result and result == last:
                count += 1
            else:
                last = result
                count = 0

            stable = count >= stable_frames

            # Convert ROI coordinates back into full-frame coordinates
            display_nodes = []

            for n in nodes:
                nn = n.copy()
                nn["x"] += x0
                nn["y"] += y0
                display_nodes.append(nn)

            disp = draw_overlay(frame, display_nodes, path, result)

            # Dim everything outside the scan region
            overlay = disp.copy()

            overlay[:y0, :] = (40, 40, 40)
            overlay[y0 + side:, :] = (40, 40, 40)
            overlay[y0:y0 + side, :x0] = (40, 40, 40)
            overlay[y0:y0 + side, x0 + side:] = (40, 40, 40)

            disp = cv2.addWeighted(overlay, 0.5, disp, 0.5, 0)

            # Draw scan box
            cv2.rectangle(
                disp,
                (x0, y0),
                (x0 + side, y0 + side),
                (255, 255, 0),
                2,
            )

            cv2.putText(
                disp,
                "STABLE - SPACE to accept" if stable else "Place diagram inside square",
                (10, disp.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0) if stable else (0, 0, 255),
                2,
            )

            cv2.imshow("lattice parser", disp)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" ") and stable:
                accepted = last
                break

    finally:
        vid.release()
        cv2.destroyAllWindows()

    return accepted

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--video":
        # 0 = default
        # 2 = Virtual Camera (OBS)
        print(video())
    elif args[0] == "--snapshot":
        print(image(None, show=True))
    else:
        print(image(args[0], show="--show" in args))