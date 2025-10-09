import cv2
import requests
import time
import os
import pygame
import tempfile
# Mengimpor exception spesifik untuk penanganan error yang lebih baik
from requests.exceptions import RequestException, JSONDecodeError 

# --- KONFIGURASI ---
SERVER_URL_BASE = "http://127.0.0.1:8000"
RECOGNIZE_URL = f"{SERVER_URL_BASE}/recognize"
# LOCAL_AUDIO_DIR dihapus karena kita akan selalu mengunduh dari server untuk memastikan konsistensi

def play_audio(filename: str):
    """
    Mengunduh file audio MP3 yang spesifik dari server (via static route baru /audio)
    dan memutarnya menggunakan file sementara (tempfile).
    """
    if not filename: return
    
    # --- PERBAIKAN KRITIS UNTUK JALUR AUDIO (HARUS /audio/ )---
    # Jalur ini HARUS menggunakan /audio/ karena telah diperbarui di main.py
    audio_url = f"{SERVER_URL_BASE}/audio/{filename}"
    
    try:
        print(f"🔊 Mengunduh audio dari server: {filename}...")
        
        # Permintaan GET ke server untuk mengunduh audio
        response = requests.get(audio_url, stream=True, timeout=10)
        response.raise_for_status() # Akan memunculkan RequestException jika status non-200
        
        # Simpan ke file temporer
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_file_path = tmp_file.name

        # Inisialisasi dan putar file temporer
        pygame.mixer.music.load(tmp_file_path)
        pygame.mixer.music.play()
        print(f"✅ Memutar audio: {filename}")
        
        while pygame.mixer.music.get_busy(): # Tunggu sampai selesai diputar
            pygame.time.Clock().tick(10)
        
        # Hapus file temporer setelah selesai
        os.remove(tmp_file_path)

    except RequestException as e:
        # Menampilkan pesan error yang spesifik dari server jika gagal
        print(f"❌ Gagal mengunduh atau terhubung ke server untuk audio {filename}: {e}")
    except Exception as e:
        print(f"❌ Error saat memutar audio {filename}: {e}")
        
def run_webcam_attendance():
    # Inisialisasi Pygame (untuk mixer)
    pygame.init()
    pygame.mixer.init()
    
    # Inisialisasi Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Tidak bisa membuka kamera.")
        return
    print("✅ Kamera siap. Tekan 'SPASI' untuk absen, 'Q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        cv2.imshow('Webcam Absensi (DeepFace)', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            print("\n📸 Mengambil gambar...")
            # Mengubah frame menjadi byte JPEG
            _, image_bytes = cv2.imencode('.jpg', frame)
            
            print("✈️  Mengirim gambar ke server...")
            result = {} # Inisialisasi result sebagai dict kosong untuk kasus error
            
            try:
                # Mengirim data sebagai multipart/form-data dengan field 'file'
                files = {
                    'file': ('webcam_capture.jpg', image_bytes.tobytes(), 'image/jpeg')
                }
                
                response = requests.post(
                    RECOGNIZE_URL, 
                    files=files,  
                    timeout=17
                )
                
                # Cek status HTTP. Ini akan memunculkan RequestException jika 4xx atau 5xx.
                response.raise_for_status()

                # Coba parse JSON. Gunakan nested try-except untuk menangkap JSONDecodeError.
                try:
                    result = response.json()
                    status = result.get('status', 'N/A')
                    name = result.get('name', '')
                    
                    if status == "duplicate":
                        print(f"💬 Server Status: {status} | Pesan: {name} sudah absen hari ini.")
                    elif status == "success":
                        print(f"💬 Server Status: {status} | Nama: {name}")
                    else:
                        # Pesan error atau unrecognized
                        print(f"💬 Server Status: {status} | Pesan: {result.get('message', 'N/A')}")
                        
                except JSONDecodeError:
                    # Gagal parse JSON (server mungkin mengembalikan HTML error atau teks)
                    print(f"❌ Gagal memproses JSON. Server mengembalikan status HTTP {response.status_code}.")
                    print(f"   Respons Mentah (Mungkin Error HTML/Teks): {response.text[:200]}...") # Cetak 200 karakter pertama
                    
                
                audio_file_name = None
                
                # --- LOGIKA PENGAMBILAN AUDIO: LANGSUNG DARI TRACK_ID SERVER ---
                # Server mengirimkan nama file audio LENGKAP di field 'track_id' untuk semua status.
                audio_file_name = result.get('track_id')

                # Memutar audio
                if audio_file_name:
                    play_audio(audio_file_name)
                else:
                    print("⚠️ Gagal menentukan file audio dari respons server.")

            except RequestException as e:
                # Menangkap error koneksi, timeout, dan status HTTP yang buruk (dari raise_for_status)
                print(f"❌ Gagal terhubung/merespons dari server: {e}")
            
            time.sleep(1) # Beri jeda singkat setelah pemrosesan
            print("\n✅ Kamera siap kembali...")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    run_webcam_attendance()
