# Import các thư viện cần thiết
import tkinter as tk
from tkinter import filedialog, Text, messagebox, ttk
import tensorflow as tf
import numpy as np
from src import text_recognition, handwriting_recognition, speech_recognition
from PIL import Image, ImageTk
from googletrans import Translator
import wave
import os
import soundfile as sf
import tensorflow_hub as hub
import subprocess
import json
import tensorflow_datasets as tfds
import speech_recognition as sr
import pytesseract

# Cấu hình Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Hàm tải mô hình
def load_models():
    try:
        # Tải các mô hình và dữ liệu cần thiết
        text_model = tf.keras.models.load_model("models/text_recognition_model.keras")
        handwriting_model = tf.keras.models.load_model("models/handwriting_recognition_model.keras")
        speech_dataset, speech_info = tfds.load('speech_commands', with_info=True, as_supervised=True)
        class_names = speech_info.features['label'].names
        speech_model = speech_recognition.build_speech_model()
        speech_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        with open('yamnet_class_map.json', 'r') as f:
            class_names = json.load(f)
        return text_model, handwriting_model, speech_model, yamnet_model, class_names
    except Exception as e:
        messagebox.showerror("Lỗi Tải Mô Hình", f"Không thể tải các mô hình hoặc dữ liệu: {e}")
        return None, None, None, None, None

# Tải mô hình và dữ liệu
text_model, handwriting_model, speech_model, yamnet_model, class_names = load_models()
ds_train, ds_info = tfds.load('speech_commands', split='train', with_info=True)
class_names = ds_info.features['label'].names
print(f"Số lượng lớp: {len(class_names)}")

# Các hàm xử lý
def process_image(image_path):
    try:
        img = Image.open(image_path).convert('L')
        extracted_text = pytesseract.image_to_string(img, lang='vie+eng').strip()
        return extracted_text if extracted_text else "Không tìm thấy văn bản trong ảnh"
    except Exception as e:
        print(f"Lỗi xử lý ảnh: {str(e)}")
        return "Lỗi xử lý ảnh"

def preprocess_sequences(sequences):
    sequences = np.array(sequences)
    return (sequences / np.max(sequences)).reshape(1, -1) if len(sequences.shape) == 1 else sequences

def handle_result(sequences, text):
    try:
        if isinstance(sequences, np.ndarray) and sequences.size > 0:
            processed_sequences = preprocess_sequences(sequences)
            prediction = text_model.predict(processed_sequences)
            result = 'Tích cực' if prediction[0][0] > 0.5 else 'Tiêu cực'
        else:
            result = "Không có dữ liệu sequences"
        processed_text = text_recognition.process_text(text)
        final_result = f"Kết quả từ sequences: {result}\nKết quả từ text: {processed_text}"
        print(final_result)
    except Exception as e:
        print(f"Lỗi xử lý kết quả: {str(e)}")

def translate_text(text):
    try:
        translator = Translator()
        return translator.translate(text, src='en', dest='vi').text
    except Exception as e:
        return f"Lỗi dịch: {str(e)}"

# Lớp giao diện chính
class MultimodalRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Nhận Dạng Đa Nhiệm RNN')
        self.geometry("900x700")
        self.configure(bg="#f0f0f0")
        self.create_widgets()
    
    def create_widgets(self):
        # Tạo tiêu đề và notebook
        tk.Label(self, text="Nhận Dạng Đa Nhiệm RNN", font=("Roboto", 24, "bold"), bg="#f0f0f0", fg="#333333").pack(pady=30)
        notebook = ttk.Notebook(self)
        notebook.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Tạo các tab
        for tab_name, create_func in [
            ("Nhận Dạng Văn Bản", self.create_text_recognition_widgets),
            ("Nhận Dạng Chữ Viết Tay", self.create_handwriting_recognition_widgets),
            ("Nhận Dạng Giọng Nói", self.create_speech_recognition_widgets)
        ]:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=tab_name)
            create_func(frame)
    
    def create_text_recognition_widgets(self, parent):
        tk.Label(parent, text="Nhập văn bản:", font=("Roboto", 14), bg="#f0f0f0").pack(anchor="w", pady=(20,5))
        self.text_input = tk.Text(parent, height=5, width=60, font=("Roboto", 12))
        self.text_input.pack(fill="x", padx=20, pady=5)
        tk.Button(parent, text="Nhận Dạng Văn Bản", command=self.recognize_text, 
                  bg="#4CAF50", fg="white", font=("Roboto", 12), padx=20, pady=10).pack(pady=20)
        self.text_result_label = tk.Label(parent, text="", font=("Roboto", 14), bg="#f0f0f0")
        self.text_result_label.pack(pady=20)
    
    def create_handwriting_recognition_widgets(self, parent):
        main_frame = tk.Frame(parent, bg="#f0f0f0")
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)
        self.image_frame = tk.Frame(main_frame, bg="#f0f0f0")
        self.image_frame.pack(side=tk.LEFT, padx=(0, 10))
        self.image_label = tk.Label(self.image_frame, bg="#f0f0f0")
        self.image_label.pack()
        result_frame = tk.Frame(main_frame, bg="#f0f0f0")
        result_frame.pack(side=tk.RIGHT, expand=True, fill="both", padx=(10, 0))
        tk.Button(result_frame, text="Tải Lên Ảnh", command=self.recognize_handwriting, 
                  bg="#2196F3", fg="white", font=("Roboto", 12), padx=20, pady=10).pack(pady=20)
        self.handwriting_result_text = tk.Text(result_frame, height=8, width=40, font=("Roboto", 12))
        self.handwriting_result_text.pack(expand=True, fill="both")

    def create_speech_recognition_widgets(self, parent):
        tk.Button(parent, text="Tải Lên Âm Thanh Giọng Nói", command=self.recognize_speech, 
                  bg="#FF9800", fg="white", font=("Roboto", 12), padx=20, pady=10).pack(pady=30)
        self.speech_result_label = tk.Label(parent, text="", font=("Roboto", 14), bg="#f0f0f0", wraplength=600)
        self.speech_result_label.pack(pady=20)
    
    def recognize_text(self):
        try:
            user_input = self.text_input.get("1.0", tk.END).strip()
            if not user_input:
                raise ValueError("Vui lòng nhập văn bản")
            processed_text = text_recognition.process_text(user_input)
            processed_text = np.array(processed_text, dtype=np.float32).reshape(1, -1)
            prediction = text_model.predict(processed_text)
            emotion = 'Tích cực' if prediction[0][0] > 0.5 else 'Tiêu cực'
            translation = translate_text(user_input)
            result_text = f'Cảm xúc: {emotion}\nBản dịch: {translation}'
            self.text_result_label.config(text=result_text, fg="#008000")
        except Exception as e:
            messagebox.showerror("Lỗi Nhận Dạng Văn Bản", f"Lỗi: {e}")
    
    def recognize_handwriting(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Tệp hình ảnh", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
            if not file_path:
                return
            result = process_image(file_path)
            img = Image.open(file_path)
            display_img = img.copy()
            display_img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(display_img)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.handwriting_result_text.delete(1.0, tk.END)
            self.handwriting_result_text.insert(tk.END, result if result.strip() else "Không thể nhận dạng văn bản trong ảnh")
        except Exception as e:
            messagebox.showerror("Lỗi Nhận Dạng", f"Lỗi: {str(e)}")
    
    def recognize_speech(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Tệp âm thanh", "*.wav;*.mp3")])
            if not file_path:
                return
            wav_path = self.convert_to_wav(file_path)
            sound_type = self.classify_sound(wav_path)
            text = self.transcribe_audio(wav_path)
            result_text = f'Loại âm thanh: {sound_type}\n{text}'
            self.speech_result_label.config(text=result_text, fg="#800080")
            if wav_path != file_path:
                os.remove(wav_path)
            handle_result(None, text)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {type(e).__name__}")
            print(f"Chi tiết lỗi: {type(e).__name__} - {str(e)}")
            print(f"Dòng lỗi: {e.__traceback__.tb_lineno}")

    def convert_to_wav(self, file_path):
        wav_path = file_path.replace('.mp3', '.wav') if file_path.endswith('.mp3') else file_path + '_temp.wav'
        subprocess.run(['ffmpeg', '-y', '-i', file_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', wav_path], check=True)
        return wav_path

    def classify_sound(self, wav_path):
        if "common_voice" in wav_path:
            return "Speech"
        audio_data, sample_rate = sf.read(wav_path)
        waveform = tf.convert_to_tensor(audio_data, dtype=tf.float32)
        scores, _, _ = yamnet_model(waveform)
        speech_scores = [scores[:, i] for i, name in enumerate(class_names) if 'speech' in name.lower()]
        speech_indices = [i for i, name in enumerate(class_names) if 'speech' in name.lower()]
        if not speech_scores:
            raise ValueError("Không tìm thấy lớp speech nào")
        speech_scores = tf.stack(speech_scores, axis=1)
        max_score_index = tf.argmax(tf.reduce_mean(speech_scores, axis=0))
        class_index = speech_indices[max_score_index]
        return class_names[class_index]

    def transcribe_audio(self, wav_path):
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            print("Đang xử lý âm thanh...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)
        try:
            text_en = recognizer.recognize_google(audio, language='en-US')
            return f"Tiếng Anh: {text_en}"
        except sr.UnknownValueError:
            pass
        try:
            text_vi = recognizer.recognize_google(audio, language='vi-VN')
            return f"Tiếng Việt: {text_vi}"
        except sr.UnknownValueError:
            return "Không thể nhận dạng giọng nói"

def print_valid_classes():
    print("Các lớp âm thanh hợp lệ:")
    for idx, class_name in enumerate(class_names):
        if "speech" in class_name.lower():
            print(f"{idx}: {class_name}")

if __name__ == "__main__":
    app = MultimodalRecognitionApp()
    app.mainloop()
