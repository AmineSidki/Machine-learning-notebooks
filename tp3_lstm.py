# %% [markdown]
# # TP 3 - LSTM + Soft Attention : Prédiction Électrique
# 14 features | Split 70/15/15 | CosineAnnealingLR | EarlyStopping

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

torch.manual_seed(42)
np.random.seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
try:
    data = pd.read_csv('household_power_consumption.txt', sep=';',
                       parse_dates={'datetime': ['Date', 'Time']},
                       na_values=['?'], low_memory=False).dropna().reset_index(drop=True)
except FileNotFoundError:
    print("WARNING: household_power_consumption.txt not found. Generating dummy data.")
    dates = pd.date_range(start='2006-12-16', periods=3000, freq='T')
    data = pd.DataFrame(np.random.rand(3000, 7), columns=['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3'])
    data.insert(0, 'datetime', dates)

# === Feature Engineering Amélioré (14 features) ===
data['hour'] = data['datetime'].dt.hour
data['is_weekend'] = (data['datetime'].dt.dayofweek >= 5).astype(int)
data['hour_sin'] = np.sin(2*np.pi*data['hour']/24)
data['hour_cos'] = np.cos(2*np.pi*data['hour']/24)
data['dow_sin'] = np.sin(2*np.pi*data['datetime'].dt.dayofweek/7)
data['dow_cos'] = np.cos(2*np.pi*data['datetime'].dt.dayofweek/7)
# Using ffill/bfill to replace the deprecated 'method' arg in pandas
data['power_lag60'] = data['Global_active_power'].shift(60).bfill()
data['power_lag1440']= data['Global_active_power'].shift(1440).bfill()

feature_cols = ['Global_active_power','Global_reactive_power','Voltage',
                'Global_intensity','Sub_metering_1','Sub_metering_2',
                'Sub_metering_3','hour','is_weekend', 'hour_sin', 'hour_cos', 
                'dow_sin', 'dow_cos', 'power_lag60', 'power_lag1440']

# Scaling & Sequence creation (Fixing the missing loaders from your report)
scaler = MinMaxScaler()
dataset_scaled = scaler.fit_transform(data[feature_cols].values)

X, y = [], []
seq_len = 60
for i in range(len(dataset_scaled)-seq_len):
    X.append(dataset_scaled[i:i+seq_len])
    y.append(dataset_scaled[i+seq_len, 0])
    
X, y = np.array(X), np.array(y)

# Split 70/15/15
n_train = int(len(X) * 0.70)
n_val = int(len(X) * 0.85)

train_dataset = TensorDataset(torch.from_numpy(X[:n_train]).float(), torch.from_numpy(y[:n_train]).float().unsqueeze(-1))
val_dataset = TensorDataset(torch.from_numpy(X[n_train:n_val]).float(), torch.from_numpy(y[n_train:n_val]).float().unsqueeze(-1))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# %%
# === Modèle LSTM + Attention ===
class LSTMAttention(nn.Module):
    def __init__(self, input_size=15, hidden_size=256, num_layers=2, dropout=0.3): # input_size adjusted to 15 based on features above
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.attn_fc = nn.Linear(hidden_size, 1)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        scores = self.attn_fc(lstm_out) # (batch, 60, 1)
        weights = torch.softmax(scores, dim=1) # Normalisation
        context = (weights * lstm_out).sum(dim=1) # Vecteur de contexte
        return self.fc(self.norm(context))

# === EarlyStopping ===
class EarlyStopping:
    def __init__(self, patience=8, path='best.pt'):
        self.patience=patience
        self.best_loss=float('inf')
        self.wait=0
        self.stopped=False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - 1e-5:
            self.best_loss=val_loss
            self.wait=0
            torch.save(model.state_dict(), self.path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped=True

# %%
# === Boucle d'entraînement avec Scheduler + Clipping ===
model = LSTMAttention(input_size=15).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
early_stop = EarlyStopping(patience=8, path='best_attention.pt')
criterion = nn.MSELoss()

for epoch in range(1, 51):
    model.train()
    tr = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr += loss.item()
    tr /= len(train_loader)

    model.eval()
    vl = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            vl += criterion(model(Xb.to(device)), yb.to(device)).item()
        vl /= len(val_loader)

    scheduler.step()
    early_stop(vl, model)
    print(f'Epoch {epoch} | Train Loss: {tr:.6f} | Val Loss: {vl:.6f}')
    
    if early_stop.stopped:
        print(f'Early stop à epoch {epoch}')
        break