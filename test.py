import tensorflow as tf
from src import text_recognition, handwriting_recognition, speech_recognition
from config import TEXT_MODEL_PATH, HANDWRITING_MODEL_PATH, SPEECH_MODEL_PATH
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_model(model_path, load_data_func, model_name):
    try:
        start_time = time.time()
        model = tf.keras.models.load_model(model_path)
        _, _, x_test, y_test = load_data_func()
        
        loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        logging.info(f"{model_name}:")
        logging.info(f"  - Độ chính xác kiểm tra: {accuracy:.4f}")
        logging.info(f"  - Tổn thất kiểm tra: {loss:.4f}")
        logging.info(f"  - Thời gian thực thi: {execution_time:.2f} giây")
        
        # Thêm dự đoán mẫu
        sample_predictions = model.predict(x_test[:5])
        logging.info(f"  - Dự đoán mẫu: {sample_predictions.argmax(axis=1)}")
        logging.info(f"  - Nhãn thực tế: {y_test[:5].argmax(axis=1)}")
        
    except Exception as e:
        logging.error(f"Lỗi khi kiểm tra mô hình {model_name}: {str(e)}")

def main():
    models = [
        (TEXT_MODEL_PATH, text_recognition.load_text_data, "Nhận dạng văn bản"),
        (HANDWRITING_MODEL_PATH, handwriting_recognition.load_handwriting_data, "Nhận dạng chữ viết tay"),
        (SPEECH_MODEL_PATH, speech_recognition.load_speech_data, "Nhận dạng giọng nói")
    ]
    
    for model_path, load_data_func, model_name in models:
        test_model(model_path, load_data_func, model_name)
        print("-" * 50)  # Thêm dòng phân cách giữa các mô hình

if __name__ == "__main__":
    main()
