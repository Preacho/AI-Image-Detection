import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import load_data
# ---------- CONFIGURATION ----------
PREPROCESSED_DIR = "processed_dataset"
IMG_WIDTH = 300
IMG_HEIGHT = 300
RANDOM_STATE = 42
NPZ_PATH = "training_data.npz"


def train_random_forest():
    """Main training pipeline for Random Forest"""
    
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    X_train, X_test, y_train, y_test, class_names = load_data.load_data()
    
    
    print("\n" + "="*60)
    print("FEATURE SCALING")
    print("="*60)

    X_train = np.concatenate([X_train, (load_data.fft_features(X_train))], axis = 1)
    X_test = np.concatenate([X_test, (load_data.fft_features(X_test))], axis = 1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=RANDOM_STATE)

    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    
    rf_model = RandomForestClassifier(
        n_estimators=300,        # Number of trees
        max_depth=25,            # Maximum depth of trees
        min_samples_split=10,     # Minimum samples to split a node
        min_samples_leaf=4,      # Minimum samples in leaf
        max_features='sqrt',     # Features to consider for best split
        random_state=RANDOM_STATE,
        n_jobs=-1,              # Use all CPU cores
        class_weight='balanced'  # Handle any class imbalance
    )
    
    print("Training Random Forest (this may take a while)...")
    rf_model.fit(X_train, y_train)
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, "models/random_forest_model_fft.pkl")

    y_pred = rf_model.predict(X_test) 
    y_prob = rf_model.predict_proba(X_test)[:,1]

    model_acc = accuracy_score(y_test, y_pred)
    model_prec = precision_score(y_test,y_pred)
    model_f1 = f1_score(y_test, y_pred)
    model_recall = recall_score(y_test, y_pred)
    model_aucroc = roc_auc_score(y_test,y_prob)

    print(f"Accuracy    : {model_acc:.4f}")
    print(f"Precision   : {model_prec:.4f}")
    print(f"f1   : {model_f1:.4f}")
    print(f"Recall    : {model_recall:.4f}")
    print(f"Auc Roc    : {model_aucroc:.4f}")

if __name__ == "__main__":
    train_random_forest()