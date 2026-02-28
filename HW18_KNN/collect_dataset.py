import os
import cv2
from datetime import datetime

# ====== SETTINGS ======
DATASET_DIR = "dataset"          # inside HW18_KNN/
CLASSES = ["palm", "fist"]       # folder names
SAVE_SIZE = (128, 128)           # saved image size (better than 64x64)
BOX_SIZE = 380                   # bigger green square (try 350-450)
CAM_INDEX = 0                    # 0 usually works; if not, try 1
MIRROR = True                    # mirror camera like selfie
JPEG_QUALITY = 95                # 0-100

# Optional: slightly improve contrast (helps bad lighting)
USE_CLAHE = True


def ensure_folders():
    os.makedirs(DATASET_DIR, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, c), exist_ok=True)


def apply_clahe_bgr(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def main():
    ensure_folders()

    print(f"Classes: {', '.join(CLASSES)}")
    label = input("Enter label (palm or fist): ").strip().lower()
    while label not in CLASSES:
        print("Wrong label. Type exactly: palm or fist")
        label = input("Enter label (palm or fist): ").strip().lower()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Try CAM_INDEX=1 or check permissions.")

    # try to improve quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    saved = 0
    out_dir = os.path.join(DATASET_DIR, label)

    print("\nControls:\n  s -> save image\n  q -> quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        if MIRROR:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        # center ROI
        box = min(BOX_SIZE, w - 20, h - 20)  # keep inside frame
        x1 = w // 2 - box // 2
        y1 = h // 2 - box // 2
        x2 = x1 + box
        y2 = y1 + box

        # draw ROI rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # info
        cv2.putText(frame, f"Label: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Saved: {saved}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "Press 's' to save ROI, 'q' to quit", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Collector (ROI in box)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord("s"):
            roi = frame[y1:y2, x1:x2].copy()

            if USE_CLAHE:
                roi = apply_clahe_bgr(roi)

            roi_resized = cv2.resize(roi, SAVE_SIZE, interpolation=cv2.INTER_AREA)

            # filename
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = f"{label}_{ts}.jpg"
            fpath = os.path.join(out_dir, fname)

            cv2.imwrite(fpath, roi_resized, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            saved += 1

            # preview small saved image
            cv2.imshow("ROI (saved image preview)", roi_resized)
            print(f"Saved: {fpath}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()