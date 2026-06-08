import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.features import FEATURE_COLS

WINDOW_SIZE = 30

def prepare_data(df: pd.DataFrame):
    features = df[FEATURE_COLS].values
    target = df["target"].values

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    target_scaler = MinMaxScaler()
    target_scaled = target_scaler.fit_transform(target.reshape(-1, 1))

    X, y = [], []
    for i in range(WINDOW_SIZE, len(features_scaled)):
        X.append(features_scaled[i - WINDOW_SIZE:i])
        y.append(target_scaled[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    split = int(len(X) * 0.8)
    return (
        X[:split], X[split:],
        y[:split], y[split:],
        scaler, target_scaler,
        df.index[WINDOW_SIZE:]
    )

class LSTMModel(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]   # take last timestep
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def train(df: pd.DataFrame, ticker: str) -> dict:
    X_train, X_test, y_train, y_test, scaler, target_scaler, dates = prepare_data(df)

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t  = torch.tensor(X_test)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False)

    model = LSTMModel(input_size=X_train.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # was 0.001
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    patience, patience_counter = 10, 0

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)}")

    for epoch in range(1, 101):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # save best weights
            torch.save(model.state_dict(), f"models/{ticker}_best.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0:
            print(f"Epoch {epoch}/100 — loss: {avg_loss:.6f} (best: {best_loss:.6f})")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # load best weights
    model.load_state_dict(torch.load(f"models/{ticker}_best.pt"))

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"models/{ticker}_model.pt")
    with open(f"models/{ticker}_scaler.pkl", "wb") as f:
        pickle.dump((scaler, target_scaler), f)

    print(f"Model saved to models/{ticker}_model.pt")
    return {
        "model": model,
        "X_test": X_test_t,
        "y_test": y_test,
        "target_scaler": target_scaler,
        "dates": dates
    }

def load(ticker: str):
    with open(f"models/{ticker}_scaler.pkl", "rb") as f:
        scaler, target_scaler = pickle.load(f)
    
    # rebuild the model architecture, then load weights into it
    import pandas as pd
    from src.features import FEATURE_COLS
    n_features = len(FEATURE_COLS)
    model = LSTMModel(input_size=n_features)
    model.load_state_dict(torch.load(f"models/{ticker}_model.pt", weights_only=True))
    model.eval()
    return model, scaler, target_scaler