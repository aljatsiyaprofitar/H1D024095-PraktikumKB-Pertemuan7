import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
# from tf.keras.models import Sequential
# from tf.keras.layers import Dense, Input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


file_name = 'iris.data'

if os.path.exists(file_name):
    print(f"Memuat data dari file lokal: {file_name}")
    dataset = pd.read_csv(file_name, header=None, sep=',')
else:
    print(f"File '{file_name}' tidak ditemukan di folder lokal.")
    print("Mengunduh otomatis dari internet agar program tidak eror...")
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    dataset = pd.read_csv(url, header=None, sep=',')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_dim = X_train.shape[1]

model = Sequential([
    Input(shape=(input_dim,)),
    Dense(1000, activation='relu'),
    Dense(500, activation='relu'),
    Dense(300, activation='relu'),
    Dense(3, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Memulai Pelatihan Model ---")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

print("\n--- Evaluasi Model pada Data Uji ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss Akhir: {loss:.4f}")
print(f"Akurasi Akhir: {accuracy:.4f}\n")

pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.title('Metrik Grafik Pelatihan (Loss vs Accuracy)')
plt.ylabel('Nilai Performa')
plt.xlabel('Epoch')
plt.grid(True)
plt.show()

predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)

print("\nHasil Perbandingan Prediksi dan Target Aktual:")
print("Hasil Prediksi Model :", predicted_classes)
print("Label Sebenarnya (Asli):", y_test)

cm = confusion_matrix(y_test, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel('Spesies Prediksi')
plt.ylabel('Spesies Asli')
plt.title('Visualisasi Confusion Matrix')
plt.show()

def predict_new_data():
    print("\n--- Uji Coba Prediksi Data Bunga Baru ---")
    try:
        sepal_length = float(input("Masukkan sepal length (cm): "))
        sepal_width = float(input("Masukkan sepal width (cm): "))
        petal_length = float(input("Masukkan petal length (cm): "))
        petal_width = float(input("Masukkan petal width (cm): "))
        
        new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        prediction = model.predict(new_data, verbose=0)
        predicted_class = prediction.argmax(axis=1)
        
        predicted_label = label_encoder.inverse_transform(predicted_class)
        print(f"\nHasil Identifikasi Kelas Spesies: {predicted_label[0]}")
    except ValueError:
        print("Eror Input! Pastikan Anda mengetik angka desimal menggunakan pemisah titik (.)")

predict_new_data()