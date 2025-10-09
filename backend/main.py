import time
import sys
import os
import sqlite3
import psycopg2
import shutil 
import uuid 
from pathlib import Path
from datetime import date, timedelta 
import io
import webbrowser 
from typing import List, Optional 

# Import gTTS library for automatic Text-to-Speech generation
from gtts import gTTS 
# BARU: Tambahkan Form untuk menerima data non-file dari form
from fastapi import FastAPI, File, UploadFile, HTTPException, Form 
from starlette.requests import Request
from starlette.staticfiles import StaticFiles
from starlette.status import HTTP_302_FOUND
# BARU: Impor RedirectResponse
from starlette.responses import RedirectResponse 

# Import DeepFace (pastikan sudah terinstal: pip install deepface)
try:
    from deepface import DeepFace
except ImportError:
    print("WARNING: DeepFace library not found. Registration API might fail.")
    DeepFace = None # Fallback jika DeepFace tidak terinstal

# --- PATH & KONFIGURASI ---
# KOREKSI KRITIS 1: PROJECT_ROOT diubah agar menunjuk ke direktori induk (Absensi_DeepFace_Embedd)
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
sys.path.insert(0, str(PROJECT_ROOT))

# Impor fungsi dan konfigurasi dari file lain (asumsi sudah ada)
# KOREKSI KRITIS 2: Menggunakan relative import karena main.py berada di dalam folder backend
try:
    from .utils import extract_face_features, DISTANCE_THRESHOLD
except ImportError:
    print("⚠️ Peringatan: Gagal mengimpor utilitas dari backend/utils.py. Pastikan file ini ada.")
    # Fallback/Dummy jika utilitas tidak ditemukan
    def extract_face_features(image_bytes): return []
    DISTANCE_THRESHOLD = 0.5

# Konfigurasi DB
DB_HOST = "localhost"
DB_NAME = "vector_db"
DB_USER = "macbookpro"  
DB_PASSWORD = "deepfacepass" 
DB_PATH = PROJECT_ROOT / "backend" / "attendance.db" # Database SQLite untuk log

# FOLDER UNTUK GAMBAR
CAPTURED_IMAGES_DIR = PROJECT_ROOT / "backend" / "captured_images" # Gambar hasil absensi
FACES_DIR = PROJECT_ROOT / "backend" / "faces" # Gambar sumber untuk indexing DeepFace
# PERBAIKAN KRITIS: Mengacu langsung ke PROJECT_ROOT agar sesuai dengan struktur yang diinginkan (main.html, data.html, settings.html di root)
FRONTEND_STATIC_DIR = PROJECT_ROOT / "frontend"  # Folder untuk file HTML (main.html, data.html, settings.html) di root proyek
AUDIO_FILES_DIR = PROJECT_ROOT / "backend" / "generated_audio"

# --- INISIALISASI APLIKASI ---
app = FastAPI(title="DeepFace Absensi API")

# Mount folder audio
app.mount("/audio", StaticFiles(directory=str(AUDIO_FILES_DIR), check_dir=True), name="generated_audio")

# Mount folder gambar absensi 
app.mount("/images", StaticFiles(directory=str(CAPTURED_IMAGES_DIR), check_dir=True), name="captured_images")

# Mount folder gambar sumber wajah (untuk indexing dan penghapusan)
app.mount("/faces_data", StaticFiles(directory=str(FACES_DIR), check_dir=True), name="faces_data")


# --- FUNGSI AUDIO GENERATION ---

def generate_audio_file(filename: str, text: str):
    """Menghasilkan dan menyimpan file audio MP3 menggunakan gTTS jika belum ada."""
    audio_path = AUDIO_FILES_DIR / filename
    os.makedirs(AUDIO_FILES_DIR, exist_ok=True)
    
    if audio_path.exists():
        return

    try:
        print(f"   -> 🔊 Generating TTS file: {filename} for text: '{text}'...")
        tts = gTTS(text=text, lang='id')
        tts.save(str(audio_path))
        print(f"   -> ✅ TTS file {filename} successfully generated.")
    except Exception as e:
        # Gagal generate audio jika tidak ada koneksi internet
        print(f"❌ ERROR: Gagal generate file audio {filename}. Pastikan Anda memiliki koneksi internet: {e}")
        
# --- FUNGSI DATABASE HELPERS ---

def initialize_sqlite_db():
    """Memastikan tabel interns dan attendance_logs ada di SQLite DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interns (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                instansi TEXT
            );
        """)
        
        # Contoh data yang dijamin ada
        cursor.execute(
            "INSERT OR IGNORE INTO interns (name, instansi) VALUES (?, ?)",
            ("Said", "Software Engineer")
        )

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                intern_id INTEGER,
                intern_name TEXT NOT NULL,
                instansi TEXT,
                image_url TEXT, 
                absent_at TEXT,
                FOREIGN KEY (intern_id) REFERENCES interns(id)
            );
        """)
        
        conn.commit()
        conn.close()
        print("✅ SQLite Database (attendance.db) berhasil diinisialisasi.")
        
        # Pastikan semua folder yang akan di-mount sudah ada
        os.makedirs(CAPTURED_IMAGES_DIR, exist_ok=True)
        os.makedirs(FACES_DIR, exist_ok=True) 
        os.makedirs(AUDIO_FILES_DIR, exist_ok=True)
        print(f"✅ Folder gambar absensi ({CAPTURED_IMAGES_DIR}) dan wajah ({FACES_DIR}) siap.")
        
    except Exception as e:
        print(f"❌ KRITIS: Gagal menginisialisasi tabel SQLite: {e}")

def get_or_create_intern(name: str, instansi: str = "Intern"):
    """Mendapatkan ID intern yang sudah ada atau membuat entri baru di SQLite."""
    try:
        conn = connect_sqlite_db()
        cursor = conn.cursor()
        
        # Cek apakah nama sudah ada
        cursor.execute("SELECT id FROM interns WHERE name = ?", (name,))
        result = cursor.fetchone()
        
        if result:
            intern_id = result[0]
            conn.close()
            return intern_id
        else:
            # Jika belum ada, buat entri baru
            cursor.execute(
                "INSERT INTO interns (name, instansi) VALUES (?, ?)",
                (name, instansi)
            )
            intern_id = cursor.lastrowid
            conn.commit()
            conn.close()
            print(f"✅ Intern baru '{name}' (ID: {intern_id}) berhasil ditambahkan ke SQLite.")
            return intern_id
             
    except Exception as e:
        print(f"❌ Gagal mendapatkan/membuat entri intern di SQLite: {e}")
        # Tetap raise Exception agar API tahu proses gagal
        raise Exception(f"Gagal mengelola data intern: {e}")

def connect_vector_db():
    """Membuat koneksi ke Database Vektor (PostgreSQL)."""
    try:
        return psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    except psycopg2.Error as e:
        print(f"❌ Gagal koneksi ke Database Vektor: {e}")
        # Mengganti raise HTTPException dengan pesan yang lebih informatif untuk logging
        raise Exception("Database Vektor tidak terhubung/konfigurasi salah.")

def connect_sqlite_db():
    """Helper untuk koneksi ke SQLite DB."""
    try:
        return sqlite3.connect(DB_PATH)
    except Exception as e:
        print(f"❌ Gagal koneksi ke SQLite: {e}")
        raise HTTPException(status_code=500, detail="Database SQLite tidak terhubung.")
        
def check_duplicate_attendance(intern_name: str) -> bool:
    """Memeriksa apakah intern sudah absen hari ini di SQLite."""
    today_date = date.today().strftime('%Y-%m-%d')
    try:
        conn = connect_sqlite_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM attendance_logs WHERE intern_name = ? AND absent_at LIKE ?",
            (intern_name, f"{today_date}%")
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    except Exception as e:
        print(f"❌ Gagal memeriksa duplikasi absensi: {e}")
        return False

def log_attendance(intern_name: str, instansi: str, image_url: str):
    """Mencatat log absensi ke database SQLite dan mengembalikan ID intern."""
    try:
        conn = connect_sqlite_db()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM interns WHERE name = ?", (intern_name,))
        intern_id_tuple = cursor.fetchone()
        
        intern_id = intern_id_tuple[0] if intern_id_tuple else None
        
        cursor.execute(
            "INSERT INTO attendance_logs (intern_id, intern_name, instansi, image_url, absent_at) VALUES (?, ?, ?, ?, datetime('now', 'localtime'))",
            (intern_id, intern_name, instansi, image_url)
        )
        conn.commit()
        conn.close()
        return intern_id
             
    except Exception as e:
        print(f"❌ Gagal mencatat log absensi: {e}")
        return None

def delete_face_files(name: str):
    """Menghapus folder dan semua file gambar wajah untuk nama tertentu."""
    face_folder = FACES_DIR / name
    if face_folder.exists() and face_folder.is_dir():
        try:
            shutil.rmtree(face_folder)
            return True
        except Exception as e:
            print(f"❌ Gagal menghapus folder file wajah: {e}")
            return False
    return False

# --- HOOK UNTUK MEMBUKA BROWSER OTOMATIS ---

@app.on_event("startup")
async def startup_event():
    """Melakukan inisialisasi DB dan membuka browser saat startup."""
    initialize_sqlite_db()
    
    # Membuka browser otomatis ke main.html
    try:
        # Menunggu sebentar untuk memastikan server siap
        time.sleep(1) 
        # Cukup buka root, nanti akan diredirect oleh endpoint baru di bawah
        webbrowser.open("http://127.0.0.1:8000/") 
        print("\n=============================================")
        print("🌐 Aplikasi DeepFace Absensi siap.")
        print("Akses di: http://127.0.0.1:8000/")
        print("=============================================\n")
    except Exception as e:
        print(f"⚠️ Gagal membuka browser otomatis: {e}")
        

# --- ENDPOINT UTAMA (PERBAIKAN 404) ---

@app.get("/")
async def redirect_to_main():
    """Mengalihkan permintaan root ke main.html untuk melayani halaman utama."""
    # Mengalihkan ke main.html secara eksplisit
    return RedirectResponse(url="/main.html", status_code=HTTP_302_FOUND)


# --- ENDPOINT BARU: REGISTRASI WAJAH (API JANGKA PANJANG) ---

@app.post("/api/register-face")
async def register_new_face(
    person_name: str = Form(..., description="Nama lengkap intern."),
    instansi: str = Form("Intern", description="Jabatan intern."),
    face_image: UploadFile = File(..., description="Gambar wajah yang jelas untuk registrasi.")
):
    """
    [Jangka Panjang] Mendaftarkan wajah baru ke dalam sistem secara dinamis.
    1. Menyimpan data intern ke SQLite (jika belum ada).
    2. Menyimpan gambar ke disk (backend/faces).
    3. Ekstrak Embedding (DeepFace.represent).
    4. Menyimpan Embedding ke PostgreSQL/pgvector.
    """
    
    if DeepFace is None:
        raise HTTPException(status_code=500, detail="DeepFace tidak terinstal. Registrasi tidak dapat dilakukan.")

    print(f"\n[API] Menerima permintaan registrasi untuk: {person_name} ({instansi})")
    
    # 1. Sanitize Nama dan Tentukan Path Penyimpanan
    # Hapus spasi dan ganti dengan underscore untuk nama folder
    safe_name = person_name.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
    if not safe_name:
        raise HTTPException(status_code=400, detail="Nama orang tidak boleh kosong.")
        
    person_dir = FACES_DIR / safe_name
    person_dir.mkdir(parents=True, exist_ok=True)
    
    # Dapatkan ID Intern dari SQLite (atau buat baru)
    try:
        intern_id = get_or_create_intern(person_name, instansi)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    # 2. Simpan File Gambar ke Lokasi Operasional (backend/faces)
    file_extension = Path(face_image.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path_on_disk = person_dir / unique_filename
    
    try:
        # Pindahkan file yang diupload ke lokasi permanen
        with open(file_path_on_disk, "wb") as buffer:
            shutil.copyfileobj(face_image.file, buffer)
            
        full_path_str = str(file_path_on_disk)
        print(f"[FILE] Gambar disimpan di: {full_path_str}")

    except Exception as e:
        print(f"[ERROR] Gagal menyimpan file: {e}")
        raise HTTPException(status_code=500, detail="Gagal menyimpan file gambar di server.")

    # 3. Ekstrak Embedding Wajah
    try:
        # Menggunakan DeepFace.represent untuk memproses gambar yang baru disimpan
        embedding_objs = DeepFace.represent(
            img_path=full_path_str, 
            model_name="Facenet512", 
            enforce_detection=True
        )
        
        # Ambil vektor embedding pertama (asumsi satu wajah per gambar)
        embedding_vector = embedding_objs[0]["embedding"]
        
        print(f"[DEEPFACE] Sukses mendapatkan embedding.")

    except Exception as e:
        # Jika DeepFace gagal mendeteksi wajah, hapus file yang tadi disimpan
        os.remove(file_path_on_disk)
        print(f"[ERROR] DeepFace gagal mendeteksi wajah: {e}")
        raise HTTPException(
            status_code=422, 
            detail="Wajah tidak terdeteksi dalam gambar yang diupload. Pastikan gambar jelas dan hanya berisi satu wajah."
        )

    # 4. Simpan Data ke Database Vektor (PostgreSQL)
    try:
        conn = connect_vector_db()
        cursor = conn.cursor()
        
        # Konversi array float Python menjadi string array PostgreSQL
        vector_string = "[" + ",".join(map(str, embedding_vector)) + "]"
        
        cursor.execute("""
            INSERT INTO intern_embeddings (intern_id, name, instansi, embedding, file_path)
            VALUES (%s, %s, %s, %s::vector, %s)
        """, (intern_id, person_name, instansi, vector_string, full_path_str))
        
        conn.commit()
        conn.close()
        
        print(f"[DB] Sukses menyimpan data embedding untuk ID: {intern_id}")

    except Exception as e:
        # Jika database gagal, hapus juga file yang tadi disimpan
        os.remove(file_path_on_disk)
        print(f"[ERROR] Gagal menyimpan ke database vektor: {e}")
        raise HTTPException(status_code=500, detail="Gagal menyimpan data embedding ke database.")
        
    # Tentukan URL agar frontend dapat melihat gambar
    relative_url_path = f"{safe_name}/{unique_filename}"
    file_url = f"/faces_data/{relative_url_path}"

    return {
        "status": "success",
        "message": f"Wajah '{person_name}' sukses didaftarkan dan di-index.",
        "intern_id": intern_id,
        "user_name": person_name,
        "file_url": file_url
    }

# --- ENDPOINTS ABSENSI (main.html) ---

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """Endpoint utama untuk deteksi wajah dan pencocokan cepat."""
    start_time = time.time()
    image_bytes = await file.read() 

    image_url_for_db = ""
    
    # 1. EKSTRAKSI VEKTOR WAJAH BARU
    emb_list = extract_face_features(image_bytes) 
    
    if not emb_list:
        generate_audio_file("S002.mp3", "Wajah tidak terdeteksi. Silakan coba lagi.")
        return {"status": "error", "message": "Wajah tidak terdeteksi.", "track_id": "S002.mp3", "image_url": image_url_for_db}
    
    new_embedding = emb_list[0] 

    # 2. PENCARIAN VEKTOR DI DATABASE VEKTOR
    try:
        conn = connect_vector_db()
        cursor = conn.cursor()
        
        # Konversi array float Python menjadi string array PostgreSQL
        vector_string = "[" + ",".join(map(str, new_embedding)) + "]"

        # Menggunakan operator <=> (jarak kosinus) dari ekstensi pgvector
        cursor.execute(f"""
            SELECT name, instansi, embedding <=> '{vector_string}'::vector AS distance
            FROM intern_embeddings
            ORDER BY distance ASC
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        conn.close()

        if result:
            name, instansi, distance = result
            elapsed_time = time.time() - start_time
            
            # 3. VERIFIKASI AMBANG BATAS AKURASI
            if distance <= DISTANCE_THRESHOLD:
                
                # Check duplikasi absensi
                if check_duplicate_attendance(name):
                    print(f"✅ DUPLIKAT ABSENSI: {name} | Latensi: {elapsed_time:.2f}s")
                    audio_filename = f"duplicate_{name.replace(' ', '_')}.mp3"
                    generate_audio_file(audio_filename, f"{name}, Anda sudah absen hari ini. Selamat bekerja.")
                    
                    return {"status": "duplicate", "name": name, "instansi": instansi, "distance": f"{distance:.4f}", "latency": f"{elapsed_time:.2f}s", "track_id": audio_filename, "image_url": image_url_for_db} 
                
                # --- LOGIKA PENYIMPANAN GAMBAR ABSENSI ---
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                # Menggunakan nama yang sudah dibersihkan
                clean_name = name.replace(' ', '_').replace('.', '').lower()
                image_filename = f"{timestamp}_{clean_name}.jpg"
                image_path = CAPTURED_IMAGES_DIR / image_filename
                
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                image_url_for_db = f"/images/{image_filename}"
                # --- END LOGIKA PENYIMPANAN GAMBAR ABSENSI ---
                
                # Absensi Berhasil: Catat ke DB
                log_attendance(name, instansi, image_url_for_db) 
                print(f"✅ DETEKSI BERHASIL: {name} | Jarak: {distance:.4f} | Latensi: {elapsed_time:.2f}s | Gambar disimpan: {image_filename}")
                
                audio_filename = f"welcome_{clean_name}.mp3"
                generate_audio_file(audio_filename, f"Selamat datang, {name}. Absensi berhasil dicatat.")
                
                # Mengembalikan image_url dan jarak yang sudah diformat
                return {"status": "success", "name": name, "instansi": instansi, "distance": f"{distance:.4f}", "latency": f"{elapsed_time:.2f}s", "track_id": audio_filename, "image_url": image_url_for_db}
            else:
                # ⚠️ Tidak Dikenali (Jarak Terlalu Jauh)
                print(f"❌ DETEKSI GAGAL: Jarak Terlalu Jauh ({distance:.4f}) | Latensi: {elapsed_time:.2f}s")
                generate_audio_file("S003.mp3", "Data wajah Anda belum terdaftar di sistem. Mohon hubungi admin.")
                return {"status": "unrecognized", "message": "Data Wajah Anda Belum Terdaftar Di Sistem", "track_id": "S003.mp3", "image_url": image_url_for_db}

        else:
            # Database Vektor kosong
            generate_audio_file("S003.mp3", "Data wajah Anda belum terdaftar di sistem. Mohon hubungi admin.")
            return {"status": "error", "message": "Sistem kosong, lakukan indexing.", "track_id": "S003.mp3", "image_url": image_url_for_db}

    except Exception as e:
        print(f"❌ ERROR PENCARIAN/ABSENSI: {e}")
        # Jika koneksi DB vektor gagal, akan ada pesan error yang lebih umum
        generate_audio_file("S004.mp3", "Kesalahan server terjadi. Mohon hubungi admin.")
        return {"status": "error", "message": f"Kesalahan server: {str(e)}", "track_id": "S004.mp3", "image_url": image_url_for_db}

# --- ENDPOINTS DATA (data.html) ---

@app.get("/attendance/today") # Digunakan oleh data.html
async def get_today_attendance():
    """Mendapatkan daftar log absensi lengkap hari ini (untuk data.html)."""
    today_date = date.today().strftime('%Y-%m-%d')
    
    try:
        conn = connect_sqlite_db()
        cursor = conn.cursor()
        
        # Mengambil semua log absensi hari ini, diurutkan berdasarkan waktu terbaru
        cursor.execute("""
            SELECT intern_name, instansi, absent_at, image_url
            FROM attendance_logs 
            WHERE absent_at LIKE ?
            ORDER BY absent_at DESC
        """, (f"{today_date}%",))
        
        results = cursor.fetchall()
        conn.close()

        attendance_list = []
        for name, instansi, time_str, image_url in results: 
            # Mengembalikan list yang sesuai dengan format yang diharapkan data.html
            attendance_list.append({
                "name": name,
                "instansi": instansi,
                "timestamp": time_str,
                "distance": 0.0000, # Placeholder (tidak disimpan di log DB)
                "image_path": image_url 
            })
            
        return attendance_list # Return list directly

    except Exception as e:
        print(f"❌ Error mengambil daftar absensi hari ini: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINTS PENGATURAN (settings.html) ---

@app.post("/reload_db") # Digunakan oleh settings.html
async def reload_db():
    """Mensimulasikan operasi indexing/reload database wajah."""
    try:
        # ASUMSI: Proses indexing (memindai FACES_DIR, menghitung vektor, dan
        # menyimpannya ke intern_embeddings) dilakukan di sini.
        
        conn = connect_vector_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT name) FROM intern_embeddings")
        total_unique_faces = cursor.fetchone()[0]
        conn.close()

        print(f"✅ RELOAD SIMULASI BERHASIL. Total {total_unique_faces} wajah unik terindeks.")

        return {"status": "success", "message": "Database wajah berhasil dimuat ulang/disinkronisasi (Simulasi)", "total_faces": total_unique_faces}

    except Exception as e:
        print(f"❌ Error saat simulasi reload database: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal reload database: {e}")

@app.get("/list_faces") # Digunakan oleh settings.html
async def list_registered_faces():
    """Mendapatkan daftar wajah yang terdaftar di database vektor."""
    try:
        conn = connect_vector_db()
        cursor = conn.cursor()
        
        # Mengambil nama unik dan jumlah foto yang diwakilinya
        cursor.execute("""
            SELECT name, COUNT(*) 
            FROM intern_embeddings
            GROUP BY name
            ORDER BY name ASC
        """)
        
        results = cursor.fetchall()
        conn.close()

        # Jumlah foto di sini adalah jumlah vektor yang terindeks untuk nama tersebut
        faces_list = [{"name": name, "count": count} for name, count in results]
            
        return {"status": "success", "faces": faces_list}

    except Exception as e:
        print(f"❌ Error mengambil daftar wajah terdaftar: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal mengambil daftar wajah: {e}")

@app.delete("/delete_face/{name}") # Digunakan oleh settings.html
async def delete_face(name: str):
    """Menghapus data wajah dari database vektor dan file dari disk."""
    try:
        conn = connect_vector_db()
        cursor = conn.cursor()
        
        # 1. Hapus dari Database Vektor
        cursor.execute("DELETE FROM intern_embeddings WHERE name = %s", (name,))
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        # 2. Hapus file gambar dari folder FACES_DIR
        file_deleted = delete_face_files(name) 
        
        if deleted_count > 0 or file_deleted:
            print(f"✅ Hapus Wajah Berhasil: {name}. Vektor dihapus: {deleted_count}. File dihapus: {file_deleted}")
            
            # Memanggil reload DB (simulasi) setelah penghapusan penting agar sistem segera sinkron
            await reload_db() 
            
            return {"status": "success", "message": f"Data wajah '{name}' berhasil dihapus. Vektor: {deleted_count} dihapus."}
        else:
            return {"status": "error", "message": f"Data wajah '{name}' tidak ditemukan di database atau folder file."}

    except Exception as e:
        print(f"❌ Error menghapus data wajah: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal menghapus data wajah: {e}")


# --- ENDPOINTS LAMA (Dipertahankan untuk kompatibilitas data) ---

@app.get("/api/system-start-date")
async def get_system_start_date():
    """Mendapatkan tanggal log absensi pertama dan tanggal hari ini."""
    try:
        conn = connect_sqlite_db()
        cursor = conn.cursor()
        
        cursor.execute("SELECT absent_at FROM attendance_logs ORDER BY absent_at ASC LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        
        start_date = "N/A"
        if result:
            start_date = result[0].split(' ')[0]
            
        current_date_str = date.today().isoformat()
            
        return {"system_start_date": start_date, "current_date": current_date_str}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error mengambil tanggal mulai sistem: {e}")
        return {"system_start_date": "N/A", "current_date": date.today().isoformat(), "error": str(e)}
        
@app.get("/api/attendance-dates-with-range")
async def get_attendance_dates_with_range():
    """Mendapatkan daftar semua tanggal dari tanggal mulai sistem hingga hari ini."""
    try:
        start_info = await get_system_start_date()
        start_date_str = start_info.get("system_start_date")
        current_date_str = start_info.get("current_date")
        
        if start_date_str == "N/A":
            return {"date_range": []}

        start_date = date.fromisoformat(start_date_str)
        today = date.fromisoformat(current_date_str)

        conn = connect_sqlite_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT SUBSTR(absent_at, 1, 10) 
            FROM attendance_logs
        """)
        dates_with_logs = {row[0] for row in cursor.fetchall()}
        conn.close()

        date_range = []
        current_day = start_date
        while current_day <= today:
            date_str = current_day.isoformat()
            date_range.append({
                "date": date_str,
                "has_attendance": date_str in dates_with_logs
            })
            current_day = current_day + timedelta(days=1)
            
        return {"date_range": date_range}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error mengambil tanggal absensi dalam rentang: {e}")
        return {"date_range": [], "error": str(e)}


@app.get("/api/attendance-by-date/{date}")
async def get_attendance_by_date(date: str):
    """Mendapatkan log absensi unik berdasarkan tanggal tertentu."""
    if not date:
        raise HTTPException(status_code=400, detail="Parameter tanggal (date) diperlukan.")
    
    try:
        conn = connect_sqlite_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT T1.intern_name, T1.instansi, MAX(T1.absent_at), T1.image_url
            FROM attendance_logs T1
            WHERE T1.absent_at LIKE ?
            GROUP BY T1.intern_name, T1.instansi, T1.image_url
            ORDER BY MAX(T1.absent_at) DESC
        """, (f"{date}%",))
        
        results = cursor.fetchall()
        conn.close()

        attendance_list = []
        for name, instansi, time_str, image_url in results:
            attendance_list.append({
                "name": name,
                "instansi": instansi,
                "recognition_time": time_str,
                "status": "Hadir",
                "photo": image_url # Kunci disesuaikan dengan frontend
            })
            
        return {"date": date, "attendees": attendance_list, "total_unique": len(attendance_list)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error mengambil log absensi per tanggal: {e}")
        return {"date": date, "attendees": [], "total_unique": 0, "error": str(e)}

@app.get("/api/monthly-attendance/{year}/{month}")
async def get_monthly_attendance(year: int, month: int):
    """Mendapatkan statistik dan detail absensi bulanan."""
    try:
        conn = connect_sqlite_db()
        cursor = conn.cursor()
        
        search_pattern = f"{year}-{str(month).zfill(2)}%"
        
        # 1. Total Attendance and Unique Days
        cursor.execute("""
            SELECT 
                COUNT(*), 
                COUNT(DISTINCT SUBSTR(absent_at, 1, 10))
            FROM attendance_logs 
            WHERE absent_at LIKE ?
        """, (search_pattern,))
        
        total_attendance, unique_days = cursor.fetchone()
        
        avg_daily_attendance = round(total_attendance / unique_days, 2) if unique_days > 0 else 0
        
        # 2. Daily Stats (for Weekly Modal): Get unique attendance for each day
        cursor.execute("""
            SELECT 
                SUBSTR(absent_at, 1, 10) AS log_date, 
                intern_name, 
                instansi
            FROM attendance_logs 
            WHERE absent_at LIKE ?
            GROUP BY log_date, intern_name
            ORDER BY log_date ASC
        """, (search_pattern,))
        
        daily_log_results = cursor.fetchall()
        
        daily_stats_map = {}
        for log_date, intern_name, instansi in daily_log_results:
            if log_date not in daily_stats_map:
                daily_stats_map[log_date] = []
            daily_stats_map[log_date].append({"name": intern_name, "instansi": instansi})

        daily_stats = [{"date": d, "attendees": a} for d, a in daily_stats_map.items()]
        
        conn.close()
        
        return {
            "total_attendance": total_attendance,
            "unique_days": unique_days,
            "avg_daily_attendance": avg_daily_attendance,
            "daily_stats": daily_stats
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error mengambil data bulanan: {e}")
        return {"error": str(e), "total_attendance": 0, "unique_days": 0, "avg_daily_attendance": 0, "daily_stats": []}


# --- KRITIS: APP.MOUNT INI HARUS DI POSISI TERAKHIR (FALLBACK) ---

# Mount folder frontend utama (menangani main.html, data.html, settings.html)
# Ini berfungsi sebagai rute fallback untuk semua file statis yang belum dihandle oleh API.
app.mount("/", StaticFiles(directory=str(FRONTEND_STATIC_DIR)), name="frontend")
