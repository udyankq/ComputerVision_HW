import cv2
import numpy as np
import mediapipe as mp
from collections import deque


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def get_points(hand_lm, w, h):
    return [(int(p.x * w), int(p.y * h)) for p in hand_lm.landmark]


def fingers_no_thumb(pts):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return [1 if pts[t][1] < pts[p][1] else 0 for t, p in zip(tips, pips)]


def thumb_open(pts):
    wrist = pts[0]
    middle_mcp = pts[9]
    hand_size = max(1.0, dist(wrist, middle_mcp))
    d = dist(pts[4], pts[5])
    return 1 if d > 0.45 * hand_size else 0


class AdaptiveEMA:
    """
    Less lag on fast motion, smoother on slow motion.
    alpha_low -> smooth, alpha_high -> responsive.
    """
    def __init__(self, alpha_low=0.18, alpha_high=0.55):
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.state = None
        self.prev = None

    def reset(self):
        self.state = None
        self.prev = None

    def update(self, pt):
        p = np.array(pt, dtype=np.float32)
        if self.state is None:
            self.state = p.copy()
            self.prev = p.copy()
            return (int(self.state[0]), int(self.state[1]))

        speed = float(np.hypot(p[0] - self.prev[0], p[1] - self.prev[1]))
        t = np.clip((speed - 10.0) / 30.0, 0.0, 1.0)
        alpha = (1 - t) * self.alpha_low + t * self.alpha_high

        self.state = (1 - alpha) * self.state + alpha * p
        self.prev = p.copy()
        return (int(self.state[0]), int(self.state[1]))


def catmull_rom(p0, p1, p2, p3, n_points=18):
    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    p3 = np.array(p3, dtype=np.float32)

    pts = []
    for i in range(n_points + 1):
        t = i / n_points
        t2 = t * t
        t3 = t2 * t
        point = 0.5 * (
            (2 * p1) +
            (-p0 + p2) * t +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
            (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )
        pts.append((int(point[0]), int(point[1])))
    return pts


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.7,
    )
    draw_utils = mp.solutions.drawing_utils

    canvas = None

    brush = 9
    COLORS = {"BLUE": (255, 0, 0), "RED": (0, 0, 255), "GREEN": (0, 255, 0)}
    color_name = "GREEN"
    color = COLORS[color_name]

    ema = AdaptiveEMA(alpha_low=0.18, alpha_high=0.55)
    stroke = deque(maxlen=10)

    stable_gesture = None
    stable_count = 0
    COLOR_STABLE_FRAMES = 6

    drawing = False
    draw_on_count = 0
    draw_off_count = 0
    DRAW_ON_FRAMES = 2
    DRAW_OFF_FRAMES = 10

    erase_count = 0
    ERASE_STABLE_FRAMES = 3

    HAND_LOST_TOL = 6
    lost_hand = 0

    MAX_JUMP = 180

    mode = "IDLE"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        detected = res.multi_hand_landmarks is not None

        if detected:
            lost_hand = 0
            hand_lm = res.multi_hand_landmarks[0]
            pts = get_points(hand_lm, w, h)

            up4 = fingers_no_thumb(pts)
            count4 = sum(up4)
            th = thumb_open(pts)

            open_palm_5 = (count4 == 4 and th == 1)
            index_only = (up4 == [1, 0, 0, 0])
            fist_like = (count4 == 0 and th == 0)

            if open_palm_5:
                erase_count += 1
            else:
                erase_count = 0

            if erase_count >= ERASE_STABLE_FRAMES:
                mode = "ERASE"
                drawing = False
                draw_on_count = 0
                draw_off_count = 0
                stroke.clear()
                ema.reset()

                hull = cv2.convexHull(np.array(pts, dtype=np.int32))
                cv2.fillConvexPoly(canvas, hull, (0, 0, 0))
                cv2.polylines(frame, [hull], True, (200, 200, 200), 2)

            else:
                wanted = None
                if count4 == 2:
                    wanted = "BLUE"
                elif count4 == 3:
                    wanted = "RED"
                elif count4 == 4 and th == 0:
                    wanted = "GREEN"

                if (not drawing) and wanted is not None:
                    if wanted == stable_gesture:
                        stable_count += 1
                    else:
                        stable_gesture = wanted
                        stable_count = 1

                    if stable_count >= COLOR_STABLE_FRAMES:
                        color_name = wanted
                        color = COLORS[color_name]
                else:
                    stable_gesture = None
                    stable_count = 0

                if index_only:
                    draw_on_count += 1
                    draw_off_count = 0
                else:
                    draw_on_count = 0
                    draw_off_count += 1

                if (not drawing) and draw_on_count >= DRAW_ON_FRAMES:
                    drawing = True
                    mode = "DRAW"

                if drawing:
                    mode = "DRAW"
                    if fist_like:
                        if draw_off_count >= 3:
                            drawing = False
                            mode = "IDLE"
                            stroke.clear()
                            ema.reset()
                    elif draw_off_count >= DRAW_OFF_FRAMES:
                        drawing = False
                        mode = "IDLE"
                        stroke.clear()
                        ema.reset()

                if drawing:
                    raw = pts[8]
                    p = ema.update(raw)

                    if len(stroke) > 0 and dist(stroke[-1], p) > MAX_JUMP:
                        stroke.clear()
                        stroke.append(p)
                    else:
                        stroke.append(p)

                    if len(stroke) >= 4:
                        p0, p1, p2, p3 = stroke[-4], stroke[-3], stroke[-2], stroke[-1]
                        curve = catmull_rom(p0, p1, p2, p3, n_points=18)
                        for i in range(1, len(curve)):
                            cv2.line(canvas, curve[i - 1], curve[i], color, brush, cv2.LINE_AA)
                    elif len(stroke) >= 2:
                        cv2.line(canvas, stroke[-2], stroke[-1], color, brush, cv2.LINE_AA)

                    cv2.circle(frame, p, brush + 4, color, 2)

            draw_utils.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        else:
            lost_hand += 1
            if lost_hand > HAND_LOST_TOL:
                drawing = False
                mode = "IDLE"
                stroke.clear()
                ema.reset()

        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        base = cv2.bitwise_and(frame, inv)
        out = cv2.bitwise_or(base, canvas)

        cv2.rectangle(out, (10, 10), (760, 70), (25, 25, 25), -1)
        cv2.putText(out, f"MODE: {mode}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        cv2.putText(out, f"COLOR: {color_name}", (220, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(out, "2=BLUE  3=RED  4=GREEN  |  Open palm(5)=ERASE  |  Fist=STOP  |  C=clear",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)

        cv2.imshow("AIR - WRITING", out)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in (ord('c'), ord('C')):
            canvas[:] = 0
        if key in (ord('+'), ord('=')):
            brush = min(40, brush + 1)
        if key in (ord('-'), ord('_')):
            brush = max(1, brush - 1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()