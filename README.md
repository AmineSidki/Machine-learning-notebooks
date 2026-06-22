# Machine Learning Notebooks

A collection of deep learning practical assignments (TPs) covering computer vision, time series forecasting with LSTMs, and TinyML deployment.

---

## Notebooks

### TP1 — FashionMNIST Image Classification with CNN
**File:** `tp1_fashion_cnn.py` → `tp1_fashion_cnn.ipynb`  
**Framework:** PyTorch | **Dataset:** FashionMNIST

Implements a convolutional neural network (CNN) to classify clothing images from the FashionMNIST dataset.

- Two convolutional blocks with BatchNorm, ReLU, and MaxPool
- Fully connected head with Dropout regularization
- Trained for 5 epochs with Adam optimizer and CrossEntropyLoss
- Reports test accuracy per epoch

---

### TP2 — Household Electric Power Consumption Forecasting (Stacked LSTM & Attention)
**File:** `tp2_lstm.py` → `tp2_lstm.ipynb`  
**Framework:** PyTorch | **Dataset:** [UCI Household Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

Comparative study of recurrent architectures for hourly electricity consumption prediction, with advanced feature engineering.

**Key features:**
- 14-feature engineering: cyclical time encoding (`hour_sin/cos`, `dow_sin/cos`), lag features (`power_lag1`, `power_lag24`)
- **Model M2 — StackedLSTM v2:** 3-layer LSTM (256 units) with LayerNorm and multi-layer dense head
- **Model M3 — LSTM + Soft Attention:** attention mechanism over hidden states for weighted temporal context
- Training pipeline: CosineAnnealingLR scheduler, L2 regularization, gradient clipping, EarlyStopping
- Evaluation metrics: MAE, RMSE, R², MAPE (on real-scale kW values)
- Visualization: loss curves and 1-week prediction comparison

---

### TP3 — LSTM + Soft Attention for Power Forecasting (Standalone)
**File:** `tp3_lstm.py` → `tp3_lstm.ipynb`  
**Framework:** PyTorch | **Dataset:** UCI Household Power Consumption

A focused, self-contained implementation of the LSTM + Soft Attention model from TP2, using minute-level data (60-step sequences).

- 15 input features with MinMaxScaler normalization
- 70/15/15 chronological train/val/test split
- CosineAnnealingLR + EarlyStopping + gradient clipping
- Includes fallback to synthetic dummy data if the dataset file is not found

---

### TP4 — TinyML LSTM: Fuel Consumption Prediction
**File:** `tp4_tinyml_fuel.py` → `tp4_tinyml_fuel.ipynb`  
**Framework:** TensorFlow/Keras | **Dataset:** FuelConsumption.csv

Trains a compact stacked LSTM to predict vehicle fuel consumption, then exports the model to C++ for embedded/TinyML deployment via `everywhereML`.

- Input features: engine size, cylinders, CO₂ emissions
- Architecture: LSTM(64) → LSTM(32) → Dense(64) → Dense(1)
- Callbacks: EarlyStopping + ReduceLROnPlateau
- EDA: pairplot and Spearman correlation heatmap
- C++ export to `.h` header file for microcontroller deployment (requires `everywhereml`)

---

## Setup

### Requirements

**PyTorch notebooks (TP1, TP2, TP3):**
```bash
pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn
```

**TensorFlow notebook (TP4):**
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
# Optional for C++ export:
pip install everywhereml
```

### Datasets

| Notebook | Dataset | Source |
|----------|---------|--------|
| TP1 | FashionMNIST | Auto-downloaded via `torchvision` |
| TP2 | Household Power Consumption | Auto-downloaded from UCI, or place `household_power_consumption.txt` in working directory |
| TP3 | Household Power Consumption | Place `household_power_consumption.txt` in working directory (falls back to dummy data) |
| TP4 | FuelConsumption.csv | Place in `./data/` directory (falls back to dummy data) |

---

## Project Structure

```
.
├── README.md
├── tp1_fashion_cnn.ipynb       # CNN image classification
├── tp2_lstm.ipynb              # Stacked LSTM & Attention comparison
├── tp3_lstm.ipynb              # LSTM + Attention standalone
└── tp4_tinyml_fuel.ipynb       # TinyML LSTM + C++ export
```
