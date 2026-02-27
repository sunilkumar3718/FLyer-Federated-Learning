


import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Random import get_random_bytes
from Crypto.PublicKey import RSA


# ============================================================
# ============================================================
def preprocess_dataset(csv_path, num_edges=3):
    print("\nLoading and preprocessing dataset")

    columns = [
        'District_Name', 'Soil_color', 'Nitrogen', 'Phosphorus',
        'Potassium', 'pH', 'Rainfall', 'Temperature',
        'Crop', 'Fertilizer', 'Link'
    ]

    df = pd.read_csv(csv_path, header=None, names=columns)
    df = df[df['Nitrogen'] != 'Nitrogen']
    df.dropna(inplace=True)

    num_cols = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']
    df[num_cols] = df[num_cols].astype(float)

    soil_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()

    df['Soil_color'] = soil_encoder.fit_transform(df['Soil_color'])
    df['Crop'] = crop_encoder.fit_transform(df['Crop'])

    X = df[num_cols + ['Soil_color']]
    y = df['Crop']

    X = MinMaxScaler().fit_transform(X)

    indices = np.array_split(np.arange(len(X)), num_edges)
    edge_data = []

    for i, idx in enumerate(indices):
        edge_data.append((X[idx], y.iloc[idx].values))
        print(f" Edge-{i+1} samples: {len(idx)}")

    return edge_data, len(np.unique(y)), crop_encoder


# ============================================================
# ============================================================
def build_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================

# ============================================================
def generate_rsa_keys():
    key = RSA.generate(2048)
    print("[SECURITY] RSA keys generated")
    return key.export_key(), key.publickey().export_key()

def encrypt_weights(weights):
    aes_key = get_random_bytes(32)  # AES-256
    cipher = AES.new(aes_key, AES.MODE_GCM)
    data = pickle.dumps(weights)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return (cipher.nonce, ciphertext, tag), aes_key

def decrypt_weights(enc_data, aes_key):
    nonce, ciphertext, tag = enc_data
    cipher = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
    return pickle.loads(cipher.decrypt_and_verify(ciphertext, tag))

def encrypt_aes_key(aes_key, public_key):
    cipher = PKCS1_OAEP.new(RSA.import_key(public_key))
    return cipher.encrypt(aes_key)

def decrypt_aes_key(enc_key, private_key):
    cipher = PKCS1_OAEP.new(RSA.import_key(private_key))
    return cipher.decrypt(enc_key)


# ============================================================

# ============================================================
def local_training(edge_id, X, y, num_classes, global_weights=None):
    print(f"\nEdge-{edge_id}: Local training started")

    X = X.reshape((X.shape[0], 1, X.shape[1]))

    model = build_lstm_model((1, X.shape[2]), num_classes)

    if global_weights is not None:
        model.set_weights(global_weights)
        print(f"Edge-{edge_id}: Global model loaded")

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weights = dict(enumerate(class_weights))

    model.fit(
        X, y,
        epochs=15,
        batch_size=32,
        verbose=1,
        class_weight=class_weights
    )

    enc_weights, aes_key = encrypt_weights(model.get_weights())
    print(" Model encrypted using AES-256")

    return enc_weights, aes_key


# ============================================================
# ============================================================
def federated_average(weight_sets):
    avg_weights = []
    for layer in zip(*weight_sets):
        avg_weights.append(np.mean(layer, axis=0))
    return avg_weights

def global_update(encrypted_payloads, private_key):
    local_weights = []

    for i, (enc_weights, enc_key) in enumerate(encrypted_payloads):
        aes_key = decrypt_aes_key(enc_key, private_key)
        weights = decrypt_weights(enc_weights, aes_key)
        local_weights.append(weights)
        print(f" Decrypted Edge-{i+1} model")

    print(" Federated Averaging")
    return federated_average(local_weights)


# ============================================================
# ============================================================
print("\n========== FLYER FEDERATED LEARNING ==========")

edge_data, num_classes, crop_encoder = preprocess_dataset("cropdata.csv", 3)
private_key, public_key = generate_rsa_keys()

global_weights = None
ROUNDS = 6

for r in range(ROUNDS):
    print(f"\n========== FEDERATED ROUND {r+1} ==========")
    encrypted_updates = []

    for i, (X, y) in enumerate(edge_data):
        enc_weights, aes_key = local_training(
            i + 1, X, y, num_classes, global_weights
        )
        encrypted_updates.append(
            (enc_weights, encrypt_aes_key(aes_key, public_key))
        )

    global_weights = global_update(encrypted_updates, private_key)

print("\nFLYER TRAINING COMPLETED SUCCESSFULLY")


# ============================================================
# ============================================================
X_all = np.vstack([X for X, y in edge_data])
y_all = np.hstack([y for X, y in edge_data])

X_all = X_all.reshape((X_all.shape[0], 1, X_all.shape[1]))

global_model = build_lstm_model((1, X_all.shape[2]), num_classes)
global_model.set_weights(global_weights)

y_pred = np.argmax(global_model.predict(X_all, verbose=0), axis=1)

print("\n========== GLOBAL MODEL EVALUATION ==========")
print(f"Global Model Accuracy: {accuracy_score(y_all, y_pred):.4f}")
print("\nClassification Report:\n")
print(classification_report(y_all, y_pred))


# ============================================================
# ============================================================
print("\n========== SAMPLE CROP PREDICTION ==========")

# [N, P, K, pH, Rainfall, Temperature, Soil_color_encoded]
sample = np.array([[80, 40, 40, 6.5, 1200, 25, 1]])
sample = sample.reshape((1, 1, 7))

pred = global_model.predict(sample, verbose=0)
crop_name = crop_encoder.inverse_transform([np.argmax(pred)])

print("Predicted Crop for given soil & climate:", crop_name[0])