import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class UNSWDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sliding_windows(data_x, data_y, window_size):
    windows_x, windows_y = [], []
    for i in range(len(data_x) - window_size + 1):
        windows_x.append(data_x[i:(i + window_size)])
        windows_y.append(data_y[i + window_size - 1])
    return np.array(windows_x, dtype=np.float32), np.array(windows_y, dtype=np.int64)

def balance_windows(X_windows, y_windows, random_state=42):
    rng = np.random.RandomState(random_state)
    idx_pos = np.where(y_windows == 1)[0]
    idx_neg = np.where(y_windows == 0)[0]
    n_minority = min(len(idx_pos), len(idx_neg))

    if len(idx_pos) > len(idx_neg):
        idx_pos = rng.choice(idx_pos, size=n_minority, replace=False)
    else:
        idx_neg = rng.choice(idx_neg, size=n_minority, replace=False)

    idx = np.concatenate([idx_pos, idx_neg])
    rng.shuffle(idx)
    return X_windows[idx], y_windows[idx]

def load_and_preprocess_data(csv_path, window_size=10, val_size=0.15,
                             test_size=0.15, random_state=42):
    df = pd.read_csv(csv_path)

    # Drop de columnas de identificación y de leak
    df = df.drop(columns=[c for c in ['id', 'attack_cat', 'Label'] if c in df.columns],
                 errors='ignore')

    # Separar target ANTES de codificar features
    y = df['label'].values.astype(np.int64)
    X_df = df.drop(columns=['label'])

    # One-hot de categóricas (forzamos float para el scaler)
    cat_cols = [c for c in ['proto', 'service', 'state'] if c in X_df.columns]
    X_df = pd.get_dummies(X_df, columns=cat_cols, dtype=float)
    X = X_df.values.astype(np.float32)

    rng = np.random.RandomState(random_state)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    n = len(X)
    n_train = int(n * (1 - val_size - test_size))
    n_val = int(n * val_size)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test,  y_test  = X[n_train + n_val:], y[n_train + n_val:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    Xw_train, yw_train = create_sliding_windows(X_train, y_train, window_size)
    Xw_val,   yw_val   = create_sliding_windows(X_val,   y_val,   window_size)
    Xw_test,  yw_test  = create_sliding_windows(X_test,  y_test,  window_size)

    Xw_train, yw_train = balance_windows(Xw_train, yw_train, random_state)

    print(f"Ventanas train: {Xw_train.shape}, distribución: {np.bincount(yw_train)}")
    print(f"Ventanas val:   {Xw_val.shape}, distribución: {np.bincount(yw_val)}")
    print(f"Ventanas test:  {Xw_test.shape}, distribución: {np.bincount(yw_test)}")

    train_dataset = UNSWDataset(Xw_train, yw_train)
    val_dataset   = UNSWDataset(Xw_val,   yw_val)
    test_dataset  = UNSWDataset(Xw_test,  yw_test)
    input_dim = Xw_train.shape[2]

    return train_dataset, val_dataset, test_dataset, input_dim

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader