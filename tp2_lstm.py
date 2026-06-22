# %% [markdown]
# # TP2 — Prévision de la consommation électrique (Stacked LSTM & Attention)
# **Projet de fin de module — Deep Learning**
# 
# Ce notebook implémente l'étude comparative des réseaux récurrents pour la prédiction de la consommation électrique domestique, conformément au **Chapitre 3** du rapport universitaire.
# 
# ### Améliorations implémentées :
# 1. **Feature Engineering Avancé (9 -> 14 Features)** : 
#    - Encodage cyclique de l'heure (`hour_sin`, `hour_cos`)
#    - Encodage cyclique du jour de la semaine (`dow_sin`, `dow_cos`)
#    - Variables de Lag : contexte court terme (`power_lag1` / t-1) et contexte long terme (`power_lag24` / J-1)
# 2. **Architectures Modernisées** : 
#    - **`StackedLSTM_v2`** : 3 couches de LSTM (256 unités) avec `LayerNorm` et tête dense multicouche.
#    - **`LSTMAttention`** : Intégration d'un mécanisme de *Soft Attention* sur les états cachés temporels.
# 3. **Optimisation de l'Entraînement** :
#    - Planificateur de Learning Rate Cosinus (`CosineAnnealingLR`)
#    - Régularisation L2 (`weight_decay`)
#    - Rognage de gradient (`gradient clipping`)
#    - Arrêt précoce (`EarlyStopping`) basé sur la perte de validation

# %% [markdown]
# ## 0. Imports et Configuration

# %%
import os
import zipfile
import urllib.request
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

os.makedirs("figs", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device actif :", device)

# --- Hyperparamètres globaux ---
SEQ_LENGTH = 24      # Fenêtre d'observation (24 heures)
HIDDEN_SIZE = 256    # Capacité augmentée selon le rapport
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# %% [markdown]
# ## 1. Chargement et Préparation des Données

# %%
TXT_FILE = "household_power_consumption.txt"
ZIP_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"

if not os.path.exists(TXT_FILE):
    try:
        print("Téléchargement du dataset depuis UCI...")
        urllib.request.urlretrieve(ZIP_URL, "hpc.zip")
        with zipfile.ZipFile("hpc.zip") as z:
            z.extractall(".")
        print("Fichier extrait avec succès.")
    except Exception as e:
        print("Erreur lors du téléchargement :", e)
        print("Veuillez déposer le fichier household_power_consumption.txt manuellement.")

print("Lecture et nettoyage des données...")
data = pd.read_csv(TXT_FILE, sep=";", na_values=["?"], low_memory=False,
                   parse_dates={"datetime": ["Date", "Time"]},
                   dayfirst=True)
data = data.dropna().reset_index(drop=True)
data = data.set_index("datetime").sort_index()
print(f"Mesures brutes (minutes) lues : {len(data):,}")

# %% [markdown]
# ## 2. Ré-échantillonnage Horaire et Ingénierie des Caractéristiques

# %%
num_cols = ["Global_active_power", "Global_reactive_power", "Voltage",
            "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
data[num_cols] = data[num_cols].astype(float)

# Agrégation à l'échelle horaire (moyenne) pour éviter la saturation RAM
df = data[num_cols].resample("1h").mean().dropna()

# Extraction des composants temporels de base
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

# 1. Encodage cyclique de l'heure
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# 2. Encodage cyclique du jour de la semaine
df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

# 3. Features de Lag (Décalages temporels)
df["power_lag1"] = df["Global_active_power"].shift(1).bfill()
df["power_lag24"] = df["Global_active_power"].shift(24).bfill()

print(f"Lignes après agrégation et enrichissement (14 features) : {len(df):,}")
df.head()

# %% [markdown]
# ## 3. Préparation des Ensembles d'Apprentissage (Pas de Fuite d'Information)

# %%
feature_cols = ["Global_active_power", "Global_reactive_power", "Voltage",
                "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
                "power_lag1", "power_lag24"]

dataset = df[feature_cols].values.astype(np.float32)
n = len(dataset)

# Split chronologique 70% Train, 15% Val, 15% Test
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Normalisation StandardScaler ajustée UNIQUEMENT sur l'ensemble d'entraînement
scaler_X = StandardScaler().fit(dataset[:train_end])
dataset_scaled = scaler_X.transform(dataset)

# Scaler spécifique pour la cible afin de faciliter la dé-normalisation
scaler_y = StandardScaler().fit(dataset[:train_end, [0]])

# Génération des fenêtres temporelles (Many-to-One)
def create_windows(arr, seq_len):
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i + seq_len])
        y.append(arr[i + seq_len, 0])  # Cible : Global_active_power
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = create_windows(dataset_scaled, SEQ_LENGTH)

# Découpe des tenseurs
X_train = torch.from_numpy(X_all[:train_end - SEQ_LENGTH])
y_train = torch.from_numpy(y_all[:train_end - SEQ_LENGTH]).unsqueeze(-1)

X_val = torch.from_numpy(X_all[train_end - SEQ_LENGTH : val_end - SEQ_LENGTH])
y_val = torch.from_numpy(y_all[train_end - SEQ_LENGTH : val_end - SEQ_LENGTH]).unsqueeze(-1)

X_test = torch.from_numpy(X_all[val_end - SEQ_LENGTH:])
y_test = torch.from_numpy(y_all[val_end - SEQ_LENGTH:]).unsqueeze(-1)

print(f"Taille des ensembles : Train {X_train.shape[0]} | Val {X_val.shape[0]} | Test {X_test.shape[0]}")

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# %% [markdown]
# ## 4. Définition des Architectures (M2 & M3)

# %%
# --- Modèle M2 : Stacked LSTM v2 (Amélioré) ---
class StackedLSTMv2(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.ln(out[:, -1, :]) # On garde le dernier pas de temps
        return self.fc(out)


# --- Modèle M3 : LSTM + Soft Attention (Annexe C) ---
class LSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.attn_fc = nn.Linear(hidden_size, 1) # Calcule le score d'importance
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)             # (batch, seq_len, hidden_size)
        scores = self.attn_fc(lstm_out)        # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1) # Normalisation de l'importance par pas
        context = (weights * lstm_out).sum(dim=1) # Somme pondérée (Vecteur Contexte)
        return self.fc(self.norm(context))

# %% [markdown]
# ## 5. Classes Utilitaires (Early Stopping)

# %%
class EarlyStopping:
    def __init__(self, patience=8, path='best_model.pt'):
        self.patience = patience
        self.path = path
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - 1e-5:
            self.best_loss = val_loss
            self.wait = 0
            torch.save(model.state_dict(), self.path)
        else: 
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped = True

# %% [markdown]
# ## 6. Pipeline d'Entraînement Robuste

# %%
def train_model(model, name, train_loader, val_loader, epochs=50, lr=1e-3):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    early_stop = EarlyStopping(patience=8, path=f'best_{name}.pt')
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, epochs + 1):
        # --- Phase Entraînement ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient Clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        # --- Phase Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        
        scheduler.step()
        early_stop(epoch_val_loss, model)
        
        if epoch % 5 == 0 or epoch == 1 or early_stop.stopped:
            print(f"[{name}] Epoch {epoch:2d}/{epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
            
        if early_stop.stopped:
            print(f"-> Arrêt précoce déclenché à l'epoch {epoch}.")
            break
            
    # Restauration des meilleurs poids d'inférence
    model.load_state_dict(torch.load(f'best_{name}.pt'))
    return model, history

# %% [markdown]
# ## 7. Lancement des Entraînements

# %%
input_dim = X_train.shape[2] # 14 features

print("=== Entraînement du Modèle M2: Stacked LSTM v2 ===")
m2_model = StackedLSTMv2(input_size=input_dim)
m2_model, m2_history = train_model(m2_model, "StackedLSTMv2", train_loader, val_loader, epochs=EPOCHS, lr=LR)

print("\n=== Entraînement du Modèle M3: LSTM + Attention ===")
m3_model = LSTMAttention(input_size=input_dim)
m3_model, m3_history = train_model(m3_model, "LSTMAttention", train_loader, val_loader, epochs=EPOCHS, lr=LR)

# %% [markdown]
# ## 8. Évaluation Comparative sur l'Ensemble de Test (Échelle réelle en kW)

# %%
def evaluate_model(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)

# Prédictions brute (normalisées)
m2_preds_scaled = evaluate_model(m2_model, test_loader)
m3_preds_scaled = evaluate_model(m3_model, test_loader)

# Dénormalisation des prédictions et de la cible
y_true_kw = scaler_y.inverse_transform(y_test.numpy())
m2_preds_kw = scaler_y.inverse_transform(m2_preds_scaled)
m3_preds_kw = scaler_y.inverse_transform(m3_preds_scaled)

# Calcul des Métriques d'évaluation globales
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # Avoid divide-by-zero on MAPE
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 0.05, None))) * 100
    return mae, rmse, r2, mape

m2_metrics = compute_metrics(y_true_kw, m2_preds_kw)
m3_metrics = compute_metrics(y_true_kw, m3_preds_kw)

metrics_df = pd.DataFrame({
    "Modèle M2 : StackedLSTM_v2": m2_metrics,
    "Modèle M3 : LSTM + Attention": m3_metrics
}, index=["MAE (kW)", "RMSE (kW)", "R2 Score", "MAPE (%)"])

print("\n=== TABLEAU COMPARATIF DES RÉSULTATS (Sur Set de Test) ===")
print(metrics_df.round(4))

# %% [markdown]
# ## 9. Visualisations Graphiques

# %%
# --- Figure 1 : Comparaison des pertes (Train/Val) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(m2_history['train_loss'], label='Train Loss')
ax1.plot(m2_history['val_loss'], label='Val Loss')
ax1.set_title("M2 : StackedLSTM_v2 — MSE Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

ax2.plot(m3_history['train_loss'], label='Train Loss')
ax2.plot(m3_history['val_loss'], label='Val Loss')
ax2.set_title("M3 : LSTM + Attention — MSE Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("figs/loss_comparison.png", dpi=150)
plt.show()

# --- Figure 2 : Zoom sur les prédictions (Premières 168 heures / 1 Semaine) ---
plt.figure(figsize=(15, 6))
plt.plot(y_true_kw[:168], label="Consommation Réelle (kW)", color="black", linewidth=2)
plt.plot(m2_preds_kw[:168], label="M2: StackedLSTM_v2", color="blue", linestyle="--", alpha=0.8)
plt.plot(m3_preds_kw[:168], label="M3: LSTM + Attention", color="red", linestyle="-.", alpha=0.8)

plt.title("Comparaison des prédictions horaires sur 1 semaine de test (168h)")
plt.xlabel("Heures temporelles")
plt.ylabel("Global Active Power (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figs/predictions_comparison.png", dpi=150)
plt.show()