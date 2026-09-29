import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import torch 
import lightgbm as lgb
import load_data
import clip_utils

img_width = 300
img_height = 300
IMG_SIZE = (300,300)
RANDOM_STATE= 42


def train_lightgbm():
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    X_train, X_test, y_train, y_test, class_names = load_data.load_data()
    
    '''
    X_train = np.concatenate([X_train,load_data.fft_features(X_train)], axis = 1) 
    X_test = np.concatenate([X_test,load_data.fft_features(X_test)], axis = 1 )
    '''
    
    '''
    #Loading Clip features
    X_train_clip = clip_utils.clip_features(
        "processed_dataset/train",
        cache_path="clip_train.npz",
        image_size= IMG_SIZE,
        color_mode="grayscale")
    
    X_test_clip = clip_utils.clip_features(
        "processed_dataset/test",
        cache_path="clip_test.npz",
        image_size= IMG_SIZE,
        color_mode="grayscale")

    
    #Combine fft + clip
    X_train = np.concatenate([X_train, X_train_clip], axis=1)
    X_test = np.concatenate([X_test, X_test_clip], axis=1)
    '''
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=RANDOM_STATE)
    

    print("\n" + "="*60)
    print("TRAINING lgbClass")
    print("="*60)
    
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,              
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,       
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    
    print("training ... this may take a while ... ")
    model.fit(X_train, y_train, eval_set = [(X_val, y_val)], 
              eval_metric = 'auc', 
              callbacks = [
                  lgb.early_stopping(stopping_rounds=50, verbose=False),
                  lgb.log_evaluation(period=100)
              ])
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/lgb_model.pkl")
    print("Model saved to: models/lgb_model.pkl")
    
    y_pred = model.predict(X_test) 
    y_prob = model.predict_proba(X_test)[:,1]

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
    train_lightgbm()
