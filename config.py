import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

TEXT_MODEL_PATH = os.path.join(MODEL_DIR, 'text_recognition_model.keras')
HANDWRITING_MODEL_PATH = os.path.join(MODEL_DIR, 'handwriting_recognition_model.keras')
SPEECH_MODEL_PATH = os.path.join(MODEL_DIR, 'speech_recognition_model.keras')

EPOCHS = 10
BATCH_SIZE = 32
