import os
import csv 
import sys
import psycopg2
from pathlib import Path
from deepface import DeepFace
# from datetime import date # Tidak digunakan, dapat dihapus

# Pastikan DeepFace sudah terinstal: pip install deepface
# Pastikan psycopg2 sudah terinstal: pip install psycopg2-binary
# Pastikan pgvector sudah terinstal di PostgreSQL server Anda

# --- KONFIGURASI PROYEK ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Path ke file CSV Master di root proyek
CSV_MASTER_PATH = PROJECT_ROOT / "interns.csv" 
# Path ke folder dataset Anda (ASUMSI STRUKTUR: data/dataset/<Nama Intern>/<Gambar>.jpg)
DATASET_PATH = PROJECT_ROOT / "data" / "dataset"
# Model yang digunakan (Misalnya: ArcFace, VGG-Face, dll.)
MODEL = "ArcFace" 

# --- KONFIGURASI DATABASE VEKTOR (PostgreSQL + pgvector) ---
DB_HOST = "localhost"
DB_NAME = "vector_db"
DB_USER = "admin"
DB_PASSWORD = "deepfacepass" 
DB_TABLE_EMBEDDINGS = "intern_embeddings"

# --- FUNGSI DATABASE VEKTOR ---

def create_embeddings_table(conn):
    """Memastikan tabel intern_embeddings ada dan skemanya benar."""
    try:
        cur = conn.cursor()
        print("    -> Memastikan skema database: Menghapus tabel lama jika ada...")
        # Hapus tabel tabel lama, ini akan menghapus semua data 
        cur.execute(f"DROP TABLE IF EXISTS {DB_TABLE_EMBEDDINGS};")
        conn.commit()
        
        # NOTE: Ukuran dimensi vector ArcFace adalah 512
        cur.execute(f"""
            CREATE TABLE {DB_TABLE_EMBEDDINGS} (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                instansi VARCHAR(100),
                kategori VARCHAR(100),
                image_path VARCHAR(255) NOT NULL,
                embedding vector(512) NOT NULL
            );
        """)
        conn.commit()
        print("    -> Tabel 'intern_embeddings' berhasil dibuat/dibuat ulang dengan skema yang benar.")
    except Exception as e:
        print(f"❌ ERROR: Gagal membuat/memperbarui tabel database: {e}")
        sys.exit(1)

def connect_vector_db():
    """Membuat koneksi ke Database Vektor (PostgreSQL)."""
    try:
        # NOTE: Ganti dengan 'psycopg2-binary' jika Anda tidak menginstall kompilator C++
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        return conn
    except psycopg2.Error as e:
        print(f"❌ FATAL: Gagal koneksi ke Database Vektor: {e}")
        print("   -> Pastikan PostgreSQL server berjalan, database 'vector_db' ada, dan pgvector diaktifkan.")
        sys.exit(1)

# --- FUNGSI MEMBACA DATA CSV ---

def load_master_data():
    """Memuat interns.csv Master Data (Nama, Instansi, dan Kategori) ke dalam dictionary."""
    master_data = {}
    try:
        with open(CSV_MASTER_PATH, mode='r', encoding='utf-8') as file:
            # Gunakan csv.DictReader untuk memudahkan akses kolom berdasarkan nama
            reader = csv.DictReader(file)
            for row in reader:
                # Kunci dictionary adalah kolom 'Name' di CSV (cocok dengan nama folder)
                master_data[row['Name']] = {
                    'instansi': row['Instansi'], 
                    'kategori': row['Kategori']
                }
        print("    -> Master data interns.csv berhasil dimuat.")
        return master_data
    except FileNotFoundError:
        print(f"❌ ERROR: File Master CSV tidak ditemukan di: {CSV_MASTER_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Gagal memproses CSV: {e}")
        sys.exit(1)


# --- FUNGSI UTAMA INDEXING (Koreksi Logic Instansi & Database INSERT) ---

def index_dataset():
    conn = connect_vector_db()
    
    # Membuat/memastikan skema tabel benar (termasuk kolom 'kategori')
    create_embeddings_table(conn) 
    cur = conn.cursor()
    
    # MUAT DATA MASTER DARI CSV
    master_data = load_master_data() 
    
    print("\n🧠 Memulai proses indexing fitur (DeepFace/{})...".format(MODEL))
    
    total_indexed_count = 0
    
    # Iterasi melalui setiap folder (nama orang) di DATASET_PATH
    for person_name in os.listdir(DATASET_PATH):
        person_dir = DATASET_PATH / person_name
        if not os.path.isdir(person_dir) or person_name.startswith('.'):
            continue
            
        embeddings_to_insert = []
        person_success_count = 0
        
        # Cek dan Ambil Data Instansi dan Kategori dari CSV Master
        if person_name not in master_data:
            print(f"   ⚠️ PERINGATAN: Nama folder '{person_name}' tidak ditemukan di interns.csv. Folder diabaikan.")
            continue
            
        instansi_value = master_data[person_name]['instansi']
        kategori_value = master_data[person_name]['kategori'] 

        print(f"\n   -> Memproses {person_name} (Instansi: {instansi_value}, Kategori: {kategori_value})...")

        # Iterasi melalui setiap gambar di folder orang tersebut
        for filename in sorted(os.listdir(person_dir)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            filepath = str(person_dir / filename)
            
            try:
                # 1. GENERASI EMBEDDING WAJAH
                print(f"   -> Memproses {person_name} ({filename})...")
                
                representations = DeepFace.represent(
                    img_path=filepath, 
                    model_name=MODEL,
                    enforce_detection=False 
                )
                
                if representations:
                    # Ambil embedding vektor pertama
                    embedding_vector = representations[0]["embedding"]
                    # Format vektor ke string yang sesuai untuk PostgreSQL
                    vector_string = "[" + ",".join(map(str, embedding_vector)) + "]"
                    
                    # Tambahkan data ke batch untuk insertion. Jumlah item = 5.
                    embeddings_to_insert.append((person_name, instansi_value, kategori_value, filepath, vector_string)) 
                    person_success_count += 1
                else:
                    print(f"   ⚠️ PERINGATAN: Tidak ada representasi yang dibuat untuk {filename}. Gambar diabaikan.")
                    
            except Exception as e:
                # Cek jika errornya terkait DeepFace (misalnya, wajah tidak terdeteksi)
                if 'Face could not be detected' in str(e):
                    print(f"   ⚠️ PERINGATAN: Wajah tidak terdeteksi di {filename}. Gambar diabaikan.")
                else:
                    print(f"   ❌ Gagal memproses {filename}. Detail: {e}")

        # 2. INSERT BATCH KE DATABASE SETELAH SELESAI SATU ORANG
        if embeddings_to_insert:
            # Query untuk batch insertion. HARUS ADA 5 %s SESUAI DENGAN JUMLAH KOLOM
            # name: %s, instansi: %s, kategori: %s, image_path: %s, embedding: %s::vector
            insert_query = f"INSERT INTO {DB_TABLE_EMBEDDINGS} (name, instansi, kategori, image_path, embedding) VALUES (%s, %s, %s, %s, %s::vector)"
            
            # Data sudah disiapkan dalam embeddings_to_insert dengan 5 elemen per tuple
            
            try:
                # Menggunakan executemany untuk batch insert
                cur.executemany(insert_query, embeddings_to_insert)
                conn.commit()
                total_indexed_count += person_success_count
                print(f"✅ Selesai indexing {person_name}. Total {person_success_count} embeddings disimpan.")
            except Exception as db_e:
                conn.rollback()
                # Print baris data yang gagal untuk debugging
                print(f"❌ FATAL ERROR DB: Gagal menyimpan data untuk {person_name}. Detail: {db_e}")
                
        else:
            print(f"✅ Selesai indexing {person_name}. Total 0 embeddings disimpan.")
            
    conn.close()
    
    print("\n" + "="*50)
    if total_indexed_count > 0:
        print(f"🎉 INDEXING LENGKAP! Total {total_indexed_count} wajah berhasil disimpan.")
    else:
        print("⚠️ INDEXING SELESAI, tetapi tidak ada wajah yang berhasil di-index.")
    print("="*50)

if __name__ == "__main__":
    print("="*50)
    print("🤖 SCRIPT INDEXING & AUDIO GENERATION")
    print("="*50)
    # Panggil fungsi untuk indexing
    index_dataset()
    
    # TODO: Tambahkan fungsi generate_audio() di sini jika Anda memilikinya.
    # Misalnya: generate_audio()