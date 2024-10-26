import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from config import EPOCHS, BATCH_SIZE
from utils import plot_training_history

def build_handwriting_model():
    # Xây dựng mô hình LSTM cho nhận dạng chữ viết tay
    return models.Sequential([
        layers.LSTM(64, input_shape=(28, 28)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ]).compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def load_handwriting_data():
    # Tải và tiền xử lý dữ liệu MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train / 255.0, to_categorical(y_train, 10), x_test / 255.0, to_categorical(y_test, 10)

def train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    # Huấn luyện mô hình và vẽ biểu đồ lịch sử
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    plot_training_history(history, "Handwriting Recognition")
    return model
