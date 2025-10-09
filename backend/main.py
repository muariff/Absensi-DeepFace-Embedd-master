import time
import sys
import os
import sqlite3
import psycopg2
from pathlib import Path
from datetime import date 
import io
import webbrowser 

# Import gTTS library for automatic Text-to-Speech generation
from gtts import gTTS 
from fastapi import FastAPI, File, UploadFile, HTTPException
# Mengubah import Response: Menambahkan RedirectResponse
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.staticfiles import StaticFiles
from starlette.status import HTTP_302_FOUND

# --- PATH & KONFIGURASI ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Impor fungsi dan konfigurasi dari file lain (asumsi sudah ada)
from backend.utils import extract_face_features, DISTANCE_THRESHOLD
# Konfigurasi DB (harus disamakan dengan train.py)
# NOTE PENTING: Kredensial ini disesuaikan dengan yang berhasil di train.py
DB_HOST = "localhost"
DB_NAME = "vector_db"
DB_USER = "macbookpro"  
DB_PASSWORD = "deepfacepass" 
DB_PATH = PROJECT_ROOT / "backend" / "attendance.db" # Database SQLite untuk log
FRONTEND_STATIC_DIR = PROJECT_ROOT / "frontend" / "static" # Path ke folder static (untuk main.html, CSS, JS)
AUDIO_FILES_DIR = PROJECT_ROOT / "backend" / "generated_audio" # Path FISIK ke folder audio

# --- INISIALISASI APLIKASI ---
app = FastAPI(title="DeepFace Absensi API")

# Mount folder static utama (untuk mengakses main.html, CSS, JS)
app.mount("/static", StaticFiles(directory=str(FRONTEND_STATIC_DIR)), name="static")

# Mount folder audio spesifik menggunakan jalur baru "/audio"
# Klien (client_webcam.py) HARUS disesuaikan untuk menggunakan jalur ini.
app.mount("/audio", StaticFiles(directory=str(AUDIO_FILES_DIR), check_dir=True), name="generated_audio")

# --- FUNGSI AUDIO GENERATION ---

def generate_audio_file(filename: str, text: str):
    """Menghasilkan dan menyimpan file audio MP3 menggunakan gTTS jika belum ada."""
    audio_path = AUDIO_FILES_DIR / filename

    # 1. Pastikan folder audio ada
    os.makedirs(AUDIO_FILES_DIR, exist_ok=True)
    
    # 2. Hanya buat jika file belum ada untuk efisiensi dan kecepatan
    if audio_path.exists():
        return

    try:
        print(f"   -> 🔊 Generating TTS file: {filename} for text: '{text}'...")
        # Buat objek gTTS dengan bahasa Indonesia
        tts = gTTS(text=text, lang='id')
        
        # Simpan ke disk
        tts.save(str(audio_path))
        print(f"   -> ✅ TTS file {filename} successfully generated.")
    except Exception as e:
        print(f"❌ ERROR: Gagal generate file audio {filename}. Pastikan Anda memiliki koneksi internet: {e}")
        # Gagal generate TTS tidak boleh menghentikan server, hanya mencatat error
        
# --- FUNGSI DATABASE ---

def initialize_sqlite_db():
    """Memastikan tabel interns dan attendance_logs ada di SQLite DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 1. Buat tabel interns (jika belum ada)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interns (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                jobdesk TEXT
            );
        """)
        
        # 2. Tambahkan entri test untuk 'Said' (jika belum ada)
        # Ini penting agar log absensi pertama bisa dicatat dan duplikasi bisa dicek.
        cursor.execute(
            "INSERT OR IGNORE INTO interns (name, jobdesk) VALUES (?, ?)",
            ("Said", "Software Engineer")
        )

        # 3. Buat tabel attendance_logs (jika belum ada)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                intern_id INTEGER,
                intern_name TEXT NOT NULL,
                jobdesk TEXT,
                image_url TEXT,
                absent_at TEXT,
                FOREIGN KEY (intern_id) REFERENCES interns(id)
            );
        """)
        
        conn.commit()
        conn.close()
        print("✅ SQLite Database (attendance.db) berhasil diinisialisasi.")
        
    except Exception as e:
        print(f"❌ KRITIS: Gagal menginisialisasi tabel SQLite: {e}")
        # Jangan raise HTTPException di sini, biarkan server tetap berjalan
        # walaupun fungsi absensi sementara tidak bekerja sampai DB diperbaiki.

def connect_vector_db():
    """Membuat koneksi ke Database Vektor (PostgreSQL)."""
    try:
        return psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    except psycopg2.Error as e:
        print(f"❌ Gagal koneksi ke Database Vektor: {e}")
        raise HTTPException(status_code=500, detail="Database Vektor tidak terhubung.")

def check_duplicate_attendance(intern_name: str) -> bool:
    """Memeriksa apakah intern sudah absen hari ini di SQLite."""
    today_date = date.today().strftime('%Y-%m-%d')
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM attendance_logs WHERE intern_name = ? AND absent_at LIKE ?",
            (intern_name, f"{today_date}%")
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    except Exception as e:
        # Menangkap error jika tabel belum dibuat (sekarang seharusnya teratasi oleh initialize_sqlite_db)
        print(f"❌ Gagal memeriksa duplikasi absensi: {e}")
        return False

def log_attendance(intern_name: str, jobdesk: str, image_url: str):
    """Mencatat log absensi ke database SQLite dan mengembalikan ID intern."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM interns WHERE name = ?", (intern_name,))
        intern_id_tuple = cursor.fetchone()
        
        if intern_id_tuple:
            intern_id = intern_id_tuple[0]
            cursor.execute(
                "INSERT INTO attendance_logs (intern_id, intern_name, jobdesk, image_url, absent_at) VALUES (?, ?, ?, ?, datetime('now', 'localtime'))",
                (intern_id, intern_name, jobdesk, image_url)
            )
            conn.commit()
            conn.close()
            return intern_id
        else:
             conn.close()
             print(f"⚠️ Peringatan: Nama '{intern_name}' tidak ditemukan di tabel 'interns' SQLite.")
             return None
             
    except Exception as e:
        print(f"❌ Gagal mencatat log absensi: {e}")
        return None

# --- HOOK UNTUK MEMBUKA BROWSER OTOMATIS ---

@app.on_event("startup")
async def open_browser_on_startup():
    """Melakukan inisialisasi DB dan membuka browser setelah FastAPI selesai startup."""
    # PENTING: Inisialisasi DB harus dilakukan sebelum fungsi lain berjalan
    initialize_sqlite_db()
    
    import threading
    
    # URL yang akan dibuka (sesuai dengan default Uvicorn)
    url = "http://127.0.0.1:8000/"
    
    def open_url():
        # Beri sedikit delay agar server benar-benar siap
        time.sleep(1) 
        # Buka URL di browser default
        print(f"🌐 Membuka aplikasi absensi di: {url}")
        webbrowser.open(url) # KODE SUDAH DI-UNCOMMENT
        
    # Jalankan fungsi open_url di thread terpisah agar tidak memblokir server utama
    threading.Thread(target=open_url).start() # KODE SUDAH DI-UNCOMMENT


# --- ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Mengalihkan ke file HTML utama yang di-mount secara statis."""
    # PERBAIKAN: Mengalihkan permintaan root ("/") langsung ke file di path statis
    return RedirectResponse(url="/static/main.html", status_code=HTTP_302_FOUND)
    # Ini memastikan bahwa browser diarahkan ke path yang benar dan menghilangkan 
    # kebutuhan untuk membaca file secara manually, menghindari masalah path relatif.

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """Endpoint utama untuk deteksi wajah dan pencocokan cepat."""
    start_time = time.time()
    # image_bytes = await file.read() # Mengganti ini dengan baris di bawah
    # Pastikan data file sudah dibaca sebelum diproses
    image_bytes = await file.read() 

    # 1. EKSTRAKSI VEKTOR WAJAH BARU (GPU CRITICAL STEP)
    emb_list = extract_face_features(image_bytes) 
    
    if not emb_list:
        # ⚠️ Jika wajah tidak terdeteksi / ekstaksi fitur gagal
        generate_audio_file("S002.mp3", "Wajah tidak terdeteksi. Silakan coba lagi.")
        # MENGEMBALIKAN NAMA FILE LENGKAP DI TRACK_ID
        return {"status": "error", "message": "Wajah tidak terdeteksi.", "track_id": "S002.mp3"}
    
    new_embedding = emb_list[0] 

    # 2. PENCARIAN VEKTOR DI DATABASE VEKTOR
    try:
        conn = connect_vector_db()
        cursor = conn.cursor()
        
        # NOTE: Formatting vector string untuk query PGSQL
        vector_string = "[" + ",".join(map(str, new_embedding)) + "]"

        cursor.execute(f"""
            SELECT name, jobdesk, embedding <=> '{vector_string}'::vector AS distance
            FROM intern_embeddings
            ORDER BY distance ASC
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        conn.close()

        if result:
            name, jobdesk, distance = result
            elapsed_time = time.time() - start_time
            
            # 3. VERIFIKASI AMBANG BATAS AKURASI
            if distance <= DISTANCE_THRESHOLD:
                
                # Check duplikasi absensi
                if check_duplicate_attendance(name):
                    # ⚠️ Status Duplikat
                    print(f"✅ DUPLIKAT ABSENSI: {name} | Latensi: {elapsed_time:.2f}s")
                    
                    # MENGATUR NAMA FILE AUDIO KUSTOM UNTUK DUPLIKAT
                    audio_filename = f"duplicate_{name}.mp3"
                    generate_audio_file(audio_filename, f"{name}, Anda sudah absen hari ini. Selamat bekerja.")
                    
                    # MENGEMBALIKAN NAMA FILE LENGKAP DI TRACK_ID
                    return {"status": "duplicate", "name": name, "jobdesk": jobdesk, "distance": f"{distance:.3f}", "latency": f"{elapsed_time:.2f}s", "track_id": audio_filename}
                
                # Absensi Berhasil
                # NOTE: image_url di log_attendance bisa diganti jika Anda menyimpan foto absensi
                log_attendance(name, jobdesk, "N/A") 
                print(f"✅ DETEKSI BERHASIL: {name} | Jarak: {distance:.3f} | Latensi: {elapsed_time:.2f}s")
                
                # MENGATUR NAMA FILE AUDIO KUSTOM UNTUK SUKSES
                audio_filename = f"welcome_{name}.mp3"
                generate_audio_file(audio_filename, f"Selamat datang, {name}. Absensi berhasil dicatat.")
                
                # MENGEMBALIKAN NAMA FILE LENGKAP DI TRACK_ID
                return {"status": "success", "name": name, "jobdesk": jobdesk, "distance": f"{distance:.3f}", "latency": f"{elapsed_time:.2f}s", "track_id": audio_filename}
            else:
                # ⚠️ Tidak Dikenali (Jarak Terlalu Jauh)
                print(f"❌ DETEKSI GAGAL: Jarak Terlalu Jauh ({distance:.3f}) | Latensi: {elapsed_time:.2f}s")
                # Generate audio: S003.mp3
                generate_audio_file("S003.mp3", "Data wajah Anda belum terdaftar di sistem. Mohon hubungi admin.")
                # MENGEMBALIKAN NAMA FILE LENGKAP DI TRACK_ID
                return {"status": "unrecognized", "message": "Data Wajah Anda Belum Terdaftar Di Sistem", "track_id": "S003.mp3"}
        else:
            # Database Vektor kosong
            # Generate audio: S003.mp3
            generate_audio_file("S003.mp3", "Data wajah Anda belum terdaftar di sistem. Mohon hubungi admin.")
            # MENGEMBALIKAN NAMA FILE LENGKAP DI TRACK_ID
            return {"status": "error", "message": "Sistem kosong, lakukan indexing.", "track_id": "S003.mp3"}

    except HTTPException:
        # Menangkap HTTPException dari connect_vector_db
        raise
    except Exception as e:
        print(f"❌ ERROR PENCARIAN/ABSENSI: {e}")
        # Generate audio: S002.mp3
        generate_audio_file("S002.mp3", "Wajah tidak terdeteksi. Silakan coba lagi.")
        # MENGEMBALIKAN NAMA FILE LENGKAP DI TRACK_ID
        return {"status": "error", "message": "Kesalahan server saat pencarian/absensi.", "track_id": "S002.mp3"}

# --- ENDPOINT BARU UNTUK DATA FRONTEND ---

def connect_sqlite_db():
    """Helper untuk koneksi ke SQLite DB."""
    try:
        return sqlite3.connect(DB_PATH)
    except Exception as e:
        print(f"❌ Gagal koneksi ke SQLite: {e}")
        raise HTTPException(status_code=500, detail="Database SQLite tidak terhubung.")


@app.get("/api/system-start-date")
async def get_system_start_date():
    """Mendapatkan tanggal log absensi pertama (untuk main.html)."""
    try:
        conn = connect_sqlite_db()
        cursor = conn.cursor()
        # Mengambil tanggal absensi tertua
        cursor.execute("SELECT absent_at FROM attendance_logs ORDER BY absent_at ASC LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        
        start_date = "N/A"
        if result:
            # Format: '2025-10-09 01:53:27.262' -> ambil bagian tanggal
            start_date = result[0].split(' ')[0]
            
        return {"start_date": start_date}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error mengambil tanggal mulai sistem: {e}")
        return {"start_date": "N/A", "error": str(e)}

@app.get("/api/today-active-interns")
async def get_today_active_interns():
    """Mendapatkan daftar unik intern yang absen hari ini (untuk data.html)."""
    today_date = date.today().strftime('%Y-%m-%d')
    
    try:
        conn = connect_sqlite_db()
        cursor = conn.cursor()
        
        # Mengambil daftar unik intern yang absen hari ini
        # KODE INI MENGAMBIL LOG ABSENSI TERAKHIR per intern hari ini.
        cursor.execute("""
            SELECT T1.intern_name, T1.jobdesk, MAX(T1.absent_at)
            FROM attendance_logs T1
            WHERE T1.absent_at LIKE ?
            GROUP BY T1.intern_name, T1.jobdesk
            ORDER BY MAX(T1.absent_at) DESC
        """, (f"{today_date}%",))
        
        results = cursor.fetchall()
        conn.close()

        interns = []
        for name, jobdesk, time_str in results:
            # Hanya ambil waktu (HH:MM:SS)
            time_only = time_str.split(' ')[1].split('.')[0]
            interns.append({
                "name": name,
                "jobdesk": jobdesk,
                "time": time_only,
                "status": "Hadir" # Asumsi semua yang ada di log adalah Hadir
            })
            
        return {"active_interns": interns, "count": len(interns)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error mengambil daftar intern hari ini: {e}")
        return {"active_interns": [], "count": 0, "error": str(e)}
        
@app.get("/api/attendance-dates")
async def get_attendance_dates():
    """Mendapatkan daftar semua tanggal unik absensi yang tersedia (untuk daftarhadir.html)."""
    try:
        conn = connect_sqlite_db()
        cursor = conn.cursor()
        
        # Mengambil bagian tanggal saja dari kolom absent_at, dan mengembalikannya secara unik
        cursor.execute("""
            SELECT DISTINCT SUBSTR(absent_at, 1, 10) 
            FROM attendance_logs 
            ORDER BY 1 DESC
        """)
        
        dates = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return {"available_dates": dates}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error mengambil tanggal absensi: {e}")
        return {"available_dates": [], "error": str(e)}

@app.get("/api/attendance-by-date/{date}")
async def get_attendance_by_date(date: str):
    """Mendapatkan log absensi unik berdasarkan tanggal tertentu (untuk daftarhadir.html)."""
    if not date:
        raise HTTPException(status_code=400, detail="Parameter tanggal (date) diperlukan.")
    
    try:
        conn = connect_sqlite_db()
        cursor = conn.cursor()
        
        # Dapatkan log absensi terakhir per intern untuk tanggal tersebut
        cursor.execute("""
            SELECT T1.intern_name, T1.jobdesk, MAX(T1.absent_at)
            FROM attendance_logs T1
            WHERE T1.absent_at LIKE ?
            GROUP BY T1.intern_name, T1.jobdesk
            ORDER BY MAX(T1.absent_at) DESC
        """, (f"{date}%",))
        
        results = cursor.fetchall()
        conn.close()

        attendance_list = []
        for name, jobdesk, time_str in results:
            # Hanya ambil waktu (HH:MM:SS)
            time_only = time_str.split(' ')[1].split('.')[0]
            attendance_list.append({
                "name": name,
                "jobdesk": jobdesk,
                "time": time_only,
                "status": "Hadir" # Asumsi semua yang ada di log adalah Hadir
            })
            
        return {"date": date, "attendance_logs": attendance_list, "total_unique": len(attendance_list)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error mengambil log absensi per tanggal: {e}")
        return {"date": date, "attendance_logs": [], "total_unique": 0, "error": str(e)}
