import os
import zipfile
import shutil
import datetime
import warnings
import numpy as np
import pandas as pd
import joblib
import traceback
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
import tensorflow as tf 

warnings.filterwarnings("ignore", category=UserWarning)

TRAIN_DIR  = "preprocessed_dataset/train"
TEST_DIR   = "preprocessed_dataset/test"
MODELS_DIR = "models"
OUTPUT_XLSX = "results.xlsx"

IMG_SIZE   = (300, 300)
GRAY_MODE = "grayscale"
RGB_MODE = "rgb"
RANDOM_STATE = 42

CLIP_TRAIN_CACHE = "clip_train.npz"
CLIP_TEST_CACHE  = "clip_test.npz"

MODELS = [
    {
        "name": "RandomForest_Pixels",
        "path": "models/random_forest_model.pkl",
        "features": "pixels",
        "loader": "joblib",
    },
    {
        "name": "RandomForest_FFT",
        "path": "models/random_forest_model_fft.pkl",
        "features": "fft_pixels",
        "loader": "joblib",
    },
    {
        "name": "LightGBM",
        "path": "models/lgb_model.pkl",
        "features": "pixels",
        "loader": "joblib",
    },
    {
        "name": "LightGBM_FFT",
        "path": "models/lgb_fft_model.pkl",
        "features": "fft_pixels",
        "loader": "joblib",
    },
    {
        "name": "LightGBM_FFT_CLIP",
        "path": "models/lgb_fft_clip_model.pkl",
        "features": "fft_clip",
        "loader": "joblib",
    },
    {
        "name": "NeuralNetwork",
        "path": "models/nn_model.h5",
        "features": "pixels_rgb",
        "loader": "keras",
        "input_shape": (300, 300, 3),
    },
]
#Unzip all contents
def ensure_models_unzipped(models_dir=MODELS_DIR):
    """
    For every .zip in models_dir, extract its contents into models_dir
    (flattened). Skip files that already exist.
    """
    
    if not os.path.isdir(models_dir):
        print(f"[UNZIP] {models_dir}/ not found, skipping.")
        return []

    report = []
    zips = sorted(f for f in os.listdir(models_dir) if f.lower().endswith(".zip"))

    if not zips:
        return []


    for zname in zips:
        zpath = os.path.join(models_dir, zname)
        extracted, skipped = [], []

        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                for member in zf.namelist():
                    if member.endswith("/"):
                        continue
                    fname = os.path.basename(member)
                    if not fname:
                        continue

                    target = os.path.join(models_dir, fname)
                    if os.path.exists(target):
                        skipped.append(fname)
                        continue

                    with zf.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                    try:
                        info = zf.getinfo(member)
                        mtime = datetime.datetime(*info.date_time).timestamp()
                        os.utime(target, (mtime, mtime))
                    except Exception:
                        pass

                    extracted.append(fname)

        except zipfile.BadZipFile:
            print(f"[UNZIP]   ! {zname} is not a valid zip — skipped")
            continue

        report.append((zname, extracted, skipped))

    return report


def load_test_pixels(color_mode):
    """Load the test set with a given color mode. Returns (X, y, class_names)."""
    import tensorflow as tf
    data = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        batch_size=64,
        image_size=IMG_SIZE,
        color_mode=color_mode,
        shuffle=False,
    )
    class_names = data.class_names
    Xs, ys = [], []
    for x, y in data:
        x = x.numpy().astype(np.float32) / 255.0
        Xs.append(x.reshape(x.shape[0], -1))
        ys.append(y.numpy())
    return np.concatenate(Xs), np.concatenate(ys), class_names

def fft_only(flat_X, image_size=IMG_SIZE, color_mode=GRAY_MODE):
    """23-dim hand-crafted frequency features (grayscale)."""
    H, W = image_size
    C = 1 if color_mode == "grayscale" else 3
    N = flat_X.shape[0]
    X_img = flat_X.reshape(N, H, W, C)
    
    feats = []
    for i in range(N):
        img = X_img[i]
        gray = img.mean(axis=-1) if C > 1 else img[..., 0]
        F = np.fft.fftshift(np.fft.fft2(gray))
        mag = np.log1p(np.abs(F))
        f_mean, f_std, f_max = mag.mean(), mag.std(), mag.max()

        cy = H // 2
        cx = W // 2
        
        yy, xx = np.ogrid[:H, :W]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_max = r.max()
        edges = np.linspace(0, r_max, 5)
        band_means = []
        for b in range(4):
            mask = (r >= edges[b]) & (r < edges[b + 1])
            band_means.append(mag[mask].mean() if mask.any() else 0.0)

        row_bins = np.array_split(mag.mean(axis=1), 8)
        col_bins = np.array_split(mag.mean(axis=0), 8)
        row_feats = [rb.mean() for rb in row_bins]
        col_feats = [cb.mean() for cb in col_bins]

        feats.append([f_mean, f_std, f_max, *band_means, *row_feats, *col_feats])
    return np.array(feats, dtype=np.float32)


def load_clip_cache(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CLIP cache {path} not found.")
    return np.load(path)["emb"]


def build_features(X_gray, X_rgb):
    """Return a dict of feature sets keyed by registry string."""
    out = {}
    out["pixels"]     = X_gray
    out["fft"]        = fft_only(X_gray)
    out["fft_pixels"] = np.concatenate([X_gray, out["fft"]], axis=1)

    # RGB (for NN only)
    out["pixels_rgb"] = X_rgb

    # CLIP (optional)
    try:
        clip = load_clip_cache(CLIP_TEST_CACHE)
        out["clip"]     = clip
        out["fft_clip"] = np.concatenate([out["fft"], clip], axis=1)
    except FileNotFoundError as e:
        print(f"[WARN] {e}")
        print("[WARN] Models using CLIP features will be skipped.")

    return out

class KerasClassifierWrapper:
    """
    Wraps a Keras binary classifier so it exposes the sklearn interface.
    Optionally reshapes flat (N, 270000) input to (N, H, W, C).
    """
    def __init__(self, keras_model, input_shape=None):
        self.model = keras_model
        self.input_shape = input_shape

    def _prep(self, X):
        if self.input_shape is not None:
            return X.reshape(-1, *self.input_shape)
        return X

    def predict_proba(self, X):
        p1 = self.model.predict(self._prep(X), verbose=0).ravel()
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

def load_model(entry):
    """Return a model object with .predict() and .predict_proba()."""
    loader = entry.get("loader", "joblib")
    path = entry["path"]

    if loader == "joblib":
        return joblib.load(path)

    if loader == "keras":
        import tensorflow as tf
        try:
            keras_model = tf.keras.models.load_model(path)
        except Exception:
            keras_model = tf.keras.models.load_model(path, compile=False)
        return KerasClassifierWrapper(keras_model, entry.get("input_shape"))

    raise ValueError(f"Unknown loader: {loader}")

def evaluate_one(name, model, X, y_true, class_names):
    y_pred = model.predict(X)
    y_pred = np.asarray(y_pred).astype(int).ravel()

    has_proba = hasattr(model, "predict_proba")
    y_prob = None
    if has_proba:
        try:
            y_prob = model.predict_proba(X)[:, 1]
        except Exception:
            y_prob = None

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan

    metrics = {
        "model":       name,
        "accuracy":    round(acc, 4),
        "precision":   round(prec, 4),
        "recall":      round(rec, 4),
        "f1":          round(f1, 4),
        "auc_roc":     round(auc, 4) if not np.isnan(auc) else None,
        "specificity": round(specificity, 4),
        "balanced_acc": round(balanced, 4),
        "npv":         round(npv, 4),
        "fpr":         round(fpr, 4),
        "fnr":         round(fnr, 4),
        "tp":          int(tp),
        "fp":          int(fp),
        "tn":          int(tn),
        "fn":          int(fn),
        "n_test":      int(len(y_true)),
        "has_proba":   has_proba,
    }

    per_class_rows = []
    for i, cname in enumerate(class_names):
        y_bin_true = (y_true == i).astype(int)
        y_bin_pred = (y_pred == i).astype(int)
        per_class_rows.append({
            "model":     name,
            "class":     cname,
            "precision": round(precision_score(y_bin_true, y_bin_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_bin_true, y_bin_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_bin_true, y_bin_pred, zero_division=0), 4),
            "support":   int(y_bin_true.sum()),
        })

    cm = confusion_matrix(y_true, y_pred)
    conf_rows = []
    for i, tname in enumerate(class_names):
        for j, pname in enumerate(class_names):
            conf_rows.append({
                "model":      name,
                "true_label": tname,
                "pred_label": pname,
                "count":      int(cm[i, j]),
            })

    roc_rows = []
    if y_prob is not None:
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        step = max(1, len(fpr) // 100)
        for k in range(0, len(fpr), step):
            roc_rows.append({
                "model":     name,
                "fpr":       round(float(fpr[k]), 5),
                "tpr":       round(float(tpr[k]), 5),
                "threshold": round(float(thr[k]), 5),
            })

    return metrics, per_class_rows, conf_rows, roc_rows




def main():
    print("=" * 60)
    print("EVALUATING MODELS")
    print("=" * 60)

    print("\n[0/4] Ensuring model files are unzipped...")
    ensure_models_unzipped(MODELS_DIR)

    print("\n[1/4] Loading test pixels (grayscale + RGB)...")
    X_gray, y_test, class_names = load_test_pixels(GRAY_MODE)
    X_rgb, y_test_rgb, _        = load_test_pixels(RGB_MODE)

    # Sanity: same labels, same order
    assert np.array_equal(y_test, y_test_rgb), "Label mismatch between loads!"
    print(f"      Grayscale: {X_gray.shape}")
    print(f"      RGB      : {X_rgb.shape}")
    print(f"      Classes  : {class_names}")
    print(f"      Test size: {len(y_test)}")

    print("\n[2/4] Building feature sets...")
    features = build_features(X_gray, X_rgb)
    for k, v in features.items():
        print(f"      {k:12s} shape={v.shape}")

    print("\n[3/4] Evaluating models...")
    all_metrics, all_per_class, all_conf, all_roc = [], [], [], []

    for entry in MODELS:
        name = entry["name"]
        path = entry["path"]
        feat_key = entry["features"]

        if not os.path.exists(path):
            print(f"  [SKIP] {name}: {path} not found")
            continue
        if feat_key not in features:
            print(f"  [SKIP] {name}: features '{feat_key}' unavailable")
            continue

        print(f"  [RUN ] {name}  (features={feat_key})")
        try:
            model = load_model(entry)
            X = features[feat_key]
            m, pc, cf, rc = evaluate_one(name, model, X, y_test, class_names)
            all_metrics.append(m)
            all_per_class.extend(pc)
            all_conf.extend(cf)
            all_roc.extend(rc)
            print(f"         acc={m['accuracy']}  f1={m['f1']}  auc={m['auc_roc']}")
        except Exception as e:
            
            print(f"         [ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()

    if not all_metrics:
        print("\n[ERROR] No models evaluated successfully. Exiting.")
        return

    print("\n[4/4] Writing Excel...")
    metrics_long = []
    for m in all_metrics:
        for key, val in m.items():
            if key == "model":
                continue
            metrics_long.append({"model": m["model"], "metric": key, "value": val})

    summary_df   = pd.DataFrame(all_metrics)
    metrics_df   = pd.DataFrame(metrics_long)
    per_class_df = pd.DataFrame(all_per_class)
    conf_df      = pd.DataFrame(all_conf)
    roc_df       = pd.DataFrame(all_roc)

    meta_df = pd.DataFrame([
        {"key": "evaluated_at",  "value": datetime.datetime.now().isoformat(timespec="seconds")},
        {"key": "test_dir",      "value": TEST_DIR},
        {"key": "n_test_images", "value": len(y_test)},
        {"key": "n_classes",     "value": len(class_names)},
        {"key": "class_names",   "value": ", ".join(class_names)},
        {"key": "models_tested", "value": len(summary_df)},
        {"key": "img_size",      "value": str(IMG_SIZE)},
        {"key": "gray_models",   "value": "all tabular models"},
        {"key": "rgb_models",    "value": "NeuralNetwork_RGB"},
    ])

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as xw:
        summary_df.to_excel(xw,   sheet_name="summary",      index=False)
        metrics_df.to_excel(xw,   sheet_name="metrics_long", index=False)
        per_class_df.to_excel(xw, sheet_name="per_class",    index=False)
        conf_df.to_excel(xw,      sheet_name="confusion",    index=False)
        roc_df.to_excel(xw,       sheet_name="roc_curves",   index=False)
        meta_df.to_excel(xw,      sheet_name="metadata",     index=False)

    print(f"\nWrote {OUTPUT_XLSX}")
    print(f"\n{summary_df.to_string(index=False)}")


if __name__ == "__main__":

    main()