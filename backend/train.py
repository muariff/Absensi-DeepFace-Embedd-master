import os
import sys
import sqlite3
import psycopg2
from pathlib import Path
from deepface import DeepFace
from datetime import date # Import date jika ada tabel lain yang memerlukannya

# Pastikan DeepFace sudah terinstal: pip install deepface
# Pastikan psycopg2 sudah terinstal: pip install psycopg2-binary
# Pastikan pgvector sudah terinstal di PostgreSQL server Anda

# --- KONFIGURASI PROYEK ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Path ke folder dataset Anda (ASUMSI STRUKTUR: data/dataset/<Nama Intern>/<Gambar>.jpg)
DATASET_PATH = PROJECT_ROOT / "data" / "dataset"
# Model yang digunakan (Misalnya: ArcFace, VGG-Face, dll.)
MODEL = "ArcFace" 

# --- KONFIGURASI DATABASE VEKTOR (PostgreSQL + pgvector) ---
DB_HOST = "localhost"
DB_NAME = "vector_db"
DB_USER = "macbookpro"
DB_PASSWORD = "deepfacepass" 
DB_TABLE_EMBEDDINGS = "intern_embeddings"

# --- FUNGSI DATABASE VEKTOR ---

def create_embeddings_table(conn):
    """Memastikan tabel intern_embeddings ada dan skemanya benar."""
    try:
        cur = conn.cursor()
        print("    -> Memastikan skema database: Menghapus tabel lama jika ada...")
        # Hapus tabel lama untuk memulai dari awal (pilihan, bisa dihilangkan jika tidak ingin reset)
        cur.execute(f"DROP TABLE IF EXISTS {DB_TABLE_EMBEDDINGS};")
        conn.commit()
        
        # NOTE: Ukuran dimensi vector ArcFace adalah 512
        cur.execute(f"""
            CREATE TABLE {DB_TABLE_EMBEDDINGS} (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                instansi VARCHAR(100),
                image_path VARCHAR(255) NOT NULL,
                embedding vector(512) NOT NULL
            );
        """)
        conn.commit()
        print("    -> Tabel 'intern_embeddings' berhasil dibuat/dibuat ulang dengan skema yang benar.")
    except Exception as e:
        print(f"❌ ERROR: Gagal membuat/memperbarui tabel database: {e}")
        # Jika gagal, pastikan ekstensi 'vector' sudah terinstal di PostgreSQL
        sys.exit(1)

def connect_vector_db():
    """Membuat koneksi ke Database Vektor (PostgreSQL)."""
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        return conn
    except psycopg2.Error as e:
        print(f"❌ FATAL: Gagal koneksi ke Database Vektor: {e}")
        print("   -> Pastikan PostgreSQL server berjalan, database 'vector_db' ada, dan pgvector diaktifkan.")
        sys.exit(1) # Keluar dari skrip jika koneksi gagal

# --- FUNGSI UTAMA INDEXING ---

def index_dataset():
    conn = connect_vector_db()
    create_embeddings_table(conn)
    cur = conn.cursor()
    
    print("\n🧠 Memulai proses indexing fitur (DeepFace/{})...".format(MODEL))
    
    total_indexed_count = 0
    
    # Iterasi melalui setiap folder (nama orang) di DATASET_PATH
    for person_name in os.listdir(DATASET_PATH):
        person_dir = DATASET_PATH / person_name
        if not os.path.isdir(person_dir) or person_name.startswith('.'):
            continue
            
        embeddings_to_insert = []
        person_success_count = 0
        instansi = "Intern" # Asumsi instansi default, ganti jika Anda punya tabel instansi terpisah
        
        print(f"\n   -> Memproses {person_name}...")
        
        # Iterasi melalui setiap gambar di folder orang tersebut
        for filename in sorted(os.listdir(person_dir)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            filepath = str(person_dir / filename)
            
            try:
                # 1. GENERASI EMBEDDING WAJAH
                # enforce_detection=False: MEMPERBAIKI MASALAH WAJAH TIDAK TERDETEKSI
                print(f"   -> Memproses {person_name} ({filename})...")
                
                # DeepFace.represent akan mengembalikan list of dicts. 
                # Kita hanya mengambil wajah pertama yang terdeteksi/diwakili.
                representations = DeepFace.represent(
                    img_path=filepath, 
                    model_name=MODEL,
                    enforce_detection=False # <--- PERBAIKAN DARI MASALAH SEBELUMNYA
                )
                
                if representations:
                    # Ambil embedding vektor pertama
                    embedding_vector = representations[0]["embedding"]
                    # Format vektor ke string yang sesuai untuk PostgreSQL
                    vector_string = "[" + ",".join(map(str, embedding_vector)) + "]"
                    
                    # Tambahkan data ke batch untuk insertion
                    embeddings_to_insert.append((person_name, instansi, filepath, vector_string))
                    person_success_count += 1
                else:
                    # Ini terjadi jika DeepFace benar-benar gagal mengekstrak fitur meskipun enforce_detection=False
                    print(f"   ⚠️ PERINGATAN: Tidak ada representasi yang dibuat untuk {filename}. Gambar diabaikan.")
                
            except Exception as e:
                # Menangani kegagalan DeepFace.represent (misal: gambar rusak)
                print(f"   ❌ Gagal memproses {filename}. Detail: {e}")

        # 2. INSERT BATCH KE DATABASE SETELAH SELESAI SATU ORANG
        if embeddings_to_insert:
            # Query untuk batch insertion
            insert_query = f"INSERT INTO {DB_TABLE_EMBEDDINGS} (name, instansi, image_path, embedding) VALUES (%s, %s, %s, %s::vector)"
            
            # Prepare data: hilangkan format vector_string karena sudah di-cast di query
            data_to_insert = [(name, instansi, path, vector_str) 
                              for name, instansi, path, vector_str in embeddings_to_insert]

            try:
                cur.executemany(insert_query, data_to_insert)
                conn.commit() # <<< COMMIT INI SANGAT PENTING
                total_indexed_count += person_success_count
                print(f"✅ Selesai indexing {person_name}. Total {person_success_count} embeddings disimpan.")
            except Exception as db_e:
                conn.rollback()
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
