import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from skimage.feature import hog

DATASET_DIR = "dataset"
IMG_SIZE = (96, 96)
CLASS_NAMES = ["palm", "fist"]

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys"
)

def extract_hog(gray):
    feat = hog(gray, **HOG_PARAMS)
    return feat.astype(np.float32)

def load_dataset():
    X, y = [], []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Missing folder: {class_dir}")

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            path = os.path.join(class_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue

            img = cv2.resize(img, IMG_SIZE)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            feat = extract_hog(gray)
            X.append(feat)
            y.append(label_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y

def main():
    X, y = load_dataset()
    print(f"Loaded samples: {len(X)}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"Class '{name}': {(y == i).sum()} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    best_k, best_acc, best_model = None, -1, None
    for k in [1, 3, 5, 7, 9, 11]:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        acc = accuracy_score(y_test, pred)
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_model = knn

    y_pred = best_model.predict(X_test)

    print("\n=== Results ===")
    print(f"Best KNN k={best_k} accuracy: {best_acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReport:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

    np.savez(
        "knn_model.npz",
        X_train=X_train,
        y_train=y_train,
        k=best_k,
        classes=np.array(CLASS_NAMES),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        hog_params=np.array(list(HOG_PARAMS.items()), dtype=object),
        img_size=np.array(IMG_SIZE)
    )
    print("\nSaved model data to knn_model.npz")

if __name__ == "__main__":
    main()
