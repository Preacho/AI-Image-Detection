
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

IMG_SIZE = (300,300)
COLOR_MODE = "grayscale"

def load_dataset_from_imageset(directory, batch_size = 64, image_size = (96,96), color_mode=COLOR_MODE):
    data = tf.keras.utils.image_dataset_from_directory(
        directory,
        batch_size = batch_size,
        image_size = image_size, 
        color_mode = color_mode, 
        shuffle = False, 
    )
    class_names = data.class_names
    Xs, ys = [], []

    for x, y in data:
        x = x.numpy().astype(np.float32)/ 255.0
        Xs.append(x.reshape(x.shape[0], -1))
        ys.append(y.numpy())

    return np.concatenate(Xs), np.concatenate(ys), class_names 


def load_data():
    """Load train and test data from preprocessed directories"""
    
    X_train, y_train, class_names = load_dataset_from_imageset("processed_dataset/train", 64, IMG_SIZE)
    X_test, y_test, _ = load_dataset_from_imageset("processed_dataset/test", 64, IMG_SIZE)
    
    return X_train, X_test, y_train, y_test, class_names

def fft_features(flat_X, image_size = IMG_SIZE, color_mode = COLOR_MODE):
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

        f_mean = mag.mean()
        f_std  = mag.std()
        f_max  = mag.max()

        cy, cx = H // 2, W // 2
        yy, xx = np.ogrid[:H, :W]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_max = r.max()
        band_edges = np.linspace(0, r_max, 5)  
        band_means = []
        for b in range(4):
            mask = (r >= band_edges[b]) & (r < band_edges[b + 1])
            band_means.append(mag[mask].mean() if mask.any() else 0.0)

        row_mean = mag.mean(axis=1)   
        col_mean = mag.mean(axis=0)   

        row_bins = np.array_split(row_mean, 8)
        col_bins = np.array_split(col_mean, 8)
        row_feats = [rb.mean() for rb in row_bins]
        col_feats = [cb.mean() for cb in col_bins]

        feats.append([f_mean, f_std, f_max, *band_means,
                      *row_feats, *col_feats])

    fft = np.array(feats,dtype=np.float32)
    return fft