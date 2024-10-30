# Import các module cần thiết từ các file chứa các mô hình và dữ liệu riêng biệt
from src import text_recognition, handwriting_recognition, speech_recognition
import tensorflow as tf
import logging
import os
from config import EPOCHS, BATCH_SIZE, MODEL_DIR
import tensorflow_datasets as tfds

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Đường dẫn lưu mô hình
TEXT_MODEL_PATH = os.path.join(MODEL_DIR, 'text_recognition_model.keras')
HANDWRITING_MODEL_PATH = os.path.join(MODEL_DIR, 'handwriting_recognition_model.keras')
SPEECH_MODEL_PATH = os.path.join(MODEL_DIR, 'speech_recognition_model.keras')

# Tạo thư mục models nếu chưa tồn tại
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model(model_type, model_path, load_data_func, build_model_func, train_model_func):
    if os.path.exists(model_path):
        logging.info(f"Mô hình {model_type} đã tồn tại tại: {model_path}. Bỏ qua huấn luyện.")
        return

    logging.info(f"Đang huấn luyện mô hình {model_type}...")
    x_train, y_train, x_test, y_test = load_data_func()
    model = build_model_func()
    model = train_model_func(model, x_train, y_train, x_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE)
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    logging.info(f"Độ chính xác của mô hình {model_type} trên tập kiểm tra: {test_accuracy:.4f}")

    model.save(model_path)
    logging.info(f"Đã lưu mô hình {model_type} tại: {model_path}")

def train_speech_model():
    if os.path.exists(SPEECH_MODEL_PATH):
        logging.info(f"Mô hình nhận diện âm thanh đã tồn tại tại: {SPEECH_MODEL_PATH}. Bỏ qua huấn luyện.")
        return

    logging.info("Đang huấn luyện mô hình nhận diện âm thanh...")

    (ds_train, ds_test), ds_info = tfds.load('speech_commands', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True)

    def preprocess_audio(spectrogram, label):
        return tf.cast(spectrogram, tf.float32) / 255.0, label

    ds_train = ds_train.map(preprocess_audio, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(ds_info.splits['train'].num_examples).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess_audio, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

    model = speech_recognition.build_speech_model()
    model.summary()

    speech_model_dir = os.path.join(MODEL_DIR, 'speech_recognition_temp')
    os.makedirs(speech_model_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(speech_model_dir, 'speech_model_best.keras'), save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(speech_model_dir, 'logs'), histogram_freq=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(ds_train, epochs=EPOCHS, validation_data=ds_test, callbacks=callbacks)

    test_loss, test_accuracy = model.evaluate(ds_test)
    logging.info(f"Độ chính xác của mô hình nhận diện âm thanh trên tập kiểm tra: {test_accuracy:.4f}")

    model.save(SPEECH_MODEL_PATH, save_format='tf')
    logging.info(f"Đã lưu mô hình nhận diện âm thanh tại: {SPEECH_MODEL_PATH}")

    import pickle
    history_path = os.path.join(speech_model_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    logging.info(f"Đã lưu lịch sử huấn luyện tại: {history_path}")

    speech_recognition.plot_training_history(history)

if __name__ == "__main__":
    logging.info("Bắt đầu kiểm tra và huấn luyện các mô hình...")
    
    train_model("nhận diện văn bản", TEXT_MODEL_PATH, text_recognition.load_text_data, text_recognition.build_text_recognition_model, text_recognition.train_model)
    train_model("nhận diện chữ viết tay", HANDWRITING_MODEL_PATH, handwriting_recognition.load_handwriting_data, handwriting_recognition.build_handwriting_model, handwriting_recognition.train_model)
    train_speech_model()

    logging.info("Đã hoàn thành kiểm tra và huấn luyện tất cả các mô hình!")
