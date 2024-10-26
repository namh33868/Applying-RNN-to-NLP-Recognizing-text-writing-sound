# Nhập các thư viện cần thiết
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import EPOCHS, BATCH_SIZE
from utils import plot_training_history
import logging
from googletrans import Translator
import re
import numpy as np

# Xử lý văn bản thành vector số
def process_text(text):
    try:
        words = text.lower().split()
        vector = np.zeros(1000)
        for i, word in enumerate(words[:1000]):
            vector[i] = hash(word) % 100 / 100.0
        return vector
    except Exception as e:
        print(f"Lỗi xử lý văn bản: {str(e)}")
        return np.zeros(1000)

# Dịch văn bản sang tiếng Việt
def translate_text(text):
    try:
        translator = Translator()
        result = translator.translate(text, dest='vi')
        return result.text
    except:
        return text

# Làm sạch văn bản
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return ' '.join(text.split())

# Xây dựng mô hình nhận dạng văn bản
def build_text_recognition_model():
    try:
        model = models.Sequential([
            layers.Embedding(input_dim=10000, output_dim=256, mask_zero=True),
            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy', keras.metrics.AUC()])
        return model
    except Exception as e:
        logging.error(f"Lỗi khi xây dựng mô hình: {str(e)}")
        raise

# Đặt lại dữ liệu
def reset_data():
    global x_train, y_train, x_test, y_test
    x_train, y_train, x_test, y_test = None, None, None, None

# Tải dữ liệu văn bản
def load_text_data():
    try:
        reset_data()
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
        
        max_length = max(max(len(seq) for seq in x_train), max(len(seq) for seq in x_test))
        x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
        x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')
        
        return x_train, y_train, x_test, y_test
    except Exception as e:
        logging.error(f"Lỗi khi tải dữ liệu: {str(e)}")
        raise

# Huấn luyện mô hình
def train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    try:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001
        )
        
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            validation_data=(x_test, y_test),
            callbacks=[early_stopping, reduce_lr]
        )
        
        plot_training_history(history, "Text Recognition")
        return model
    except Exception as e:
        logging.error(f"Lỗi khi huấn luyện mô hình: {str(e)}")
        raise
