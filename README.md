# Ứng dụng Nhận dạng Đa nhiệm RNN

## Giới thiệu

Ứng dụng desktop tiên tiến này tận dụng sức mạnh của mô hình RNN (Recurrent Neural Network) để thực hiện đa dạng các tác vụ nhận dạng phức tạp:

- 📝 Phân tích cảm xúc và nhận dạng văn bản
- ✍️ Nhận dạng chữ viết tay 
- 🗣️ Nhận dạng giọng nói
- 🌐 Dịch thuật văn bản tự động

## Chức năng nổi bật

🔍 Phân tích sắc thái cảm xúc trong văn bản (tích cực/tiêu cực)
🔄 Dịch thuật tự động giữa tiếng Anh và tiếng Việt
📸 Nhận dạng chữ viết tay từ hình ảnh
🎙️ Chuyển đổi giọng nói thành văn bản
🖥️ Giao diện người dùng trực quan, thân thiện được xây dựng bằng Tkinter

## Cấu trúc dự án
project/
│
├── src/ # Thư mục chứa mã nguồn chính
│ ├── text_recognition.py # Xử lý nhận dạng văn bản
│ ├── handwriting_recognition.py # Xử lý nhận dạng chữ viết tay
│ ├── speech_recognition.py # Xử lý nhận dạng giọng nói
│ └── utils.py # Các tiện ích
│
├── models/ # Thư mục chứa các mô hình đã train
├── app.py # File chính chạy ứng dụng
├── train.py # Script huấn luyện mô hình
├── test.py # Script kiểm thử
└── requirements.txt # Các thư viện yêu cầu

## Yêu cầu hệ thống

Để chạy ứng dụng này một cách hiệu quả, bạn cần đảm bảo hệ thống của mình đáp ứng các yêu cầu sau:

- 🐍 Python 3.6 trở lên
- 🔍 Tesseract OCR (cho nhận dạng văn bản)
- 🎵 ffmpeg (cho xử lý âm thanh)
- 🖥️ GPU (được khuyến nghị để tăng tốc quá trình huấn luyện mô hình)

Lưu ý: Việc sử dụng GPU sẽ giúp cải thiện đáng kể hiệu suất khi huấn luyện các mô hình phức tạp.

## Cài đặt

1. Clone repository:
```bash
   git clone <repository-url>
   cd <project-folder>
```

2. Tạo và kích hoạt môi trường ảo:
```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
```

3. Cài đặt các thư viện yêu cầu:
```bash
   pip install -r requirements.txt
```

4. Cài đặt Tesseract OCR:
   - Windows: Tải và cài đặt từ [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux:   `sudo apt-get install tesseract-ocr`
   - Mac:     `brew install tesseract`

5. Cài đặt ffmpeg:
   - Windows: Tải và cài đặt từ [trang chủ ffmpeg](https://ffmpeg.org/download.html)
   - Linux:   `sudo apt-get install ffmpeg`
   - Mac:     `brew install ffmpeg`

## Sử dụng

1. Chạy ứng dụng:
```bash
   python app.py
```

2. Huấn luyện mô hình (tùy chọn):
```bash
   python train.py
```

3. Kiểm thử mô hình (tùy chọn):
```bash
   python test.py
```

### Hướng dẫn sử dụng các tính năng:

1. **Nhận dạng văn bản**:
   - Nhập văn bản vào ô nhập liệu
   - Nhấn nút "Nhận dạng văn bản"
   - Kết quả sẽ hiển thị cảm xúc và bản dịch tương ứng

2. **Nhận dạng chữ viết tay**:
   - Tải lên hình ảnh chứa chữ viết tay
   - Hệ thống tự động phân tích và hiển thị kết quả nhận dạng

3. **Nhận dạng giọng nói**:
   - Tải lên tệp âm thanh (định dạng .wav hoặc .mp3)
   - Hệ thống sẽ xử lý và chuyển đổi thành văn bản tương ứng

## Giấy phép
Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.
