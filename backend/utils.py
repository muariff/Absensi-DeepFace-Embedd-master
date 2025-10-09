# backend/utils.py
import numpy as np
import cv2 # PENTING: Diperlukan untuk membaca data 'bytes' menjadi array gambar (NumPy Array)
from deepface import DeepFace
import os

# --- KONFIGURASI PENTING ---
# Batas ambang jarak kosinus (Cosine Distance) untuk penentuan wajah dikenali (Threshold)
# Nilai default ini disetel ke 0.40 agar bisa diimpor oleh backend/main.py.
# Wajah dikenali jika jarak <= DISTANCE_THRESHOLD
DISTANCE_THRESHOLD = 0.40 

# --- FUNGSI EKSTRAKSI FITUR ---

def extract_face_features(image_bytes: bytes, model_name="ArcFace"):
    """
    Ekstraksi fitur wajah (embedding) menggunakan model DeepFace dari data bytes gambar.

    Args:
        image_bytes (bytes): Data gambar yang diunggah dari frontend.
        model_name (str): Nama model DeepFace yang akan digunakan (e.g., 'ArcFace').
        
    Returns:
        list of list[float]: List dari embedding wajah yang terdeteksi. 
                             Mengembalikan list kosong ([]) jika tidak ada wajah.
    """
    
    try:
        # 1. Konversi bytes (dari upload FastAPI) ke array numpy mentah
        np_array = np.frombuffer(image_bytes, np.uint8)
        # 2. Decode array bytes menjadi array gambar yang dapat dibaca OpenCV (cv2.imread)
        img_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if img_array is None:
             print("❌ Gagal membaca bytes gambar. Mungkin format file tidak didukung.")
             return []

        # 3. DeepFace.represent: menerima numpy array (img_array)
        results = DeepFace.represent(
            img_path=img_array, # Menerima NumPy array, bukan path file
            model_name=model_name,
            enforce_detection=True, # Memaksa deteksi wajah
            detector_backend='opencv' 
        )
    except ValueError as ve:
        # Menangani kesalahan spesifik DeepFace saat enforce_detection=True dan wajah tidak ditemukan
        print(f"⚠️ Peringatan: DeepFace gagal mendeteksi wajah atau membaca gambar. Detail: {ve}")
        return []
    except Exception as e:
        # Menangani error umum lainnya
        print(f"❌ ERROR Ekstraksi Fitur: {e}")
        return []


    if not results:
        return []

    # Ambil embedding dari semua wajah yang dideteksi (meskipun main.py hanya akan menggunakan yang pertama)
    embeddings_list = [res["embedding"] for res in results]
    
    # Kita mengembalikan list of list (Python list) agar mudah diproses di main.py 
    # sebelum dikonversi ke string vector PostgreSQL.
    return embeddings_list
