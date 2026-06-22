# %% [markdown]
# # TP 4 - TinyML LSTM : Prédiction Consommation Carburant
# Framework: TensorFlow/Keras | Dataset: FuelConsumption.csv
# Déploiement C++ via everywhereML

# %%
# Install required packages if running in Colab
# !pip install eloquent-tensorflow everywhereml

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# %%
# Chargement + nettoyage
# Ensure you have the dataset in a 'data' folder
os.makedirs('./data', exist_ok=True)
os.makedirs('./figures', exist_ok=True)

try:
    df = pd.read_csv('./data/FuelConsumption.csv')
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    print(f'Lignes après nettoyage : {df.shape[0]}')
except FileNotFoundError:
    print("WARNING: FuelConsumption.csv not found in ./data/. Creating dummy dataset for execution.")
    df = pd.DataFrame({
        'ENGINE SIZE': np.random.uniform(1.0, 8.0, 1000),
        'CYLINDERS': np.random.randint(3, 12, 1000),
        'COEMISSIONS ': np.random.uniform(100, 600, 1000),
        'FUEL CONSUMPTION': np.random.uniform(5.0, 30.0, 1000)
    })

# EDA
sns.pairplot(df[['ENGINE SIZE','CYLINDERS','FUEL CONSUMPTION','COEMISSIONS ']])
plt.savefig('./figures/pairplot.png', dpi=300)
# plt.show() # Uncomment if running locally with UI

corr = df[['ENGINE SIZE','CYLINDERS','FUEL CONSUMPTION','COEMISSIONS ']].corr(method='spearman')
sns.heatmap(corr, cmap='coolwarm', annot=True)
plt.savefig('./figures/heatmap.png', dpi=300)
# plt.show()

# %%
# Préparation
X = df[['ENGINE SIZE','CYLINDERS', 'COEMISSIONS ']]
y = df[['FUEL CONSUMPTION']]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Normalisation StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)

# Reshape pour LSTM (samples, timesteps, features)
X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, 3)
X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, 3)

# %%
# Architecture améliorée
model = tf.keras.Sequential([
    layers.Input(shape=(1, 3)),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.1),
    layers.Dense(1, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Callbacks
callbacks_list = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                     patience=20, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.5, patience=10, min_lr=1e-6)
]

history = model.fit(X_train_lstm, y_train_scaled,
                    validation_split=0.2, epochs=300, batch_size=32,
                    callbacks=callbacks_list)

# Évaluation
y_pred_scaled = model.predict(X_test_lstm)
y_pred_real = scaler_y.inverse_transform(y_pred_scaled)
y_test_real = scaler_y.inverse_transform(y_test)
print(f'MAE Test: {np.mean(np.abs(y_test_real - y_pred_real)):.4f} L/100km')

# %%
# Export TinyML
try:
    from everywhereml.code_generators.tensorflow import tf_porter

    portable_model = tf.keras.Sequential([
        layers.Input(shape=(1, 3)),
        layers.LSTM(64, return_sequences=True, unroll=True),
        layers.LSTM(32, unroll=True),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1, activation='linear')
    ])

    portable_model.set_weights(model.get_weights())
    porter = tf_porter(portable_model, X_train_scaled, y_train_scaled)
    code = porter.to_cpp(instance_name='fuel_model', arena_size=40000)

    with open('model_fuel_final.h', 'w') as f:
        f.write(code)

    print('Export C++ réussi : model_fuel_final.h')
except ImportError:
    print("WARNING: everywhereml not installed. C++ export skipped. Run `pip install everywhereml` first.")