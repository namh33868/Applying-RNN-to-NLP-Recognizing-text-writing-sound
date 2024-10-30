import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import tensorflow_datasets as tfds
from config import EPOCHS, BATCH_SIZE
from utils import plot_training_history
import logging
import subprocess
import os

# Đường dẫn lưu trữ dữ liệu
DATA_DIR = os.path.join(os.getcwd(), 'tensorflow_datasets')

def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info("ffmpeg đã được cài đặt và có thể truy cập.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("ffmpeg không tìm thấy hoặc gặp lỗi. Vui lòng cài đặt ffmpeg và thêm vào PATH.")
        raise

check_ffmpeg()

def build_speech_model():
    model = models.Sequential([
        layers.Input(shape=(124, 129, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(35, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess(audio, label):
    # Chuẩn hóa âm thanh
    audio = tf.squeeze(audio)
    audio = tf.cast(audio, tf.float32) / tf.int16.max
    
    # Đảm bảo độ dài cố định
    target_length = 16000
    audio = tf.cond(tf.less(tf.shape(audio)[0], target_length),
                    lambda: tf.pad(audio, [[0, target_length - tf.shape(audio)[0]]]),
                    lambda: audio[:target_length])
    
    # Tính và xử lý spectrogram
    spectrogram = tf.abs(tf.signal.stft(audio, frame_length=255, frame_step=128))
    spectrogram = tf.math.log(spectrogram + 1e-6)
    spectrogram = (spectrogram - tf.reduce_mean(spectrogram)) / tf.math.reduce_std(spectrogram)
    
    # Điều chỉnh kích thước
    spectrogram = tf.image.resize_with_crop_or_pad(tf.expand_dims(spectrogram, 0), 124, 129)
    spectrogram = tf.expand_dims(tf.squeeze(spectrogram), -1)
    
    logging.info(f"Kích thước spectrogram: {spectrogram.shape}, kiểu dữ liệu: {spectrogram.dtype}")
    return spectrogram, label

def load_speech_data():
    try:
        # Tải dữ liệu
        (ds_train, ds_test), ds_info = tfds.load('speech_commands', 
                                                 split=['train', 'test'],
                                                 shuffle_files=True,
                                                 as_supervised=True,
                                                 with_info=True,
                                                 data_dir=DATA_DIR)
        
        # Xử lý dữ liệu huấn luyện
        ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = ds_train.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        # Xử lý dữ liệu kiểm tra
        ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
        
        return ds_train, ds_test
    except Exception as e:
        logging.error(f"Lỗi khi tải dữ liệu âm thanh: {e}")
        raise

def train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    plot_training_history(history, "Nhận dạng giọng nói")
    return model
