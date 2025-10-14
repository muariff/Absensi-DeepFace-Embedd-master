import cv2
import os
from pathlib import Path

# --- Konfigurasi ---
# Tentukan path ke folder dataset utama Anda
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"

# Jumlah gambar yang akan diambil per orang
NUM_IMAGES_TO_COLLECT = 15

def collect_new_person():
    """
    Fungsi utama untuk membuka webcam, mengambil 15 gambar,
    dan menyimpannya ke folder dataset baru.
    """
    # Inisialisasi webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Error: Tidak bisa membuka kamera.")
        return

    captured_images = []
    print("✅ Kamera siap.")
    print(f"Tekan 'SPASI' untuk mengambil gambar. Butuh {NUM_IMAGES_TO_COLLECT} gambar.")
    print("Tekan 'Q' untuk keluar jika belum selesai.")

    while len(captured_images) < NUM_IMAGES_TO_COLLECT:
        ret, frame = cap.read()
        if not ret:
            print("❌ Gagal mengambil frame dari kamera.")
            break

        # Tampilkan frame webcam ke pengguna
        # Buat salinan frame agar teks tidak ikut tersimpan
        display_frame = frame.copy()
        
        # Tambahkan teks informasi ke frame yang ditampilkan
        counter_text = f"Terkumpul: {len(captured_images)}/{NUM_IMAGES_TO_COLLECT}"
        cv2.putText(display_frame, counter_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Tekan 'SPASI' untuk capture", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Dataset Collector", display_frame)

        # Tunggu input dari keyboard
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            # Simpan frame asli (tanpa teks)
            captured_images.append(frame)
            print(f"✅ Gambar ke-{len(captured_images)} berhasil diambil!")
        
        elif key == ord('q'):
            print("🛑 Proses dihentikan oleh pengguna.")
            break
    
    # Matikan webcam dan tutup jendela
    cap.release()
    cv2.destroyAllWindows()

    # Lanjutkan ke proses penyimpanan jika gambar sudah terkumpul 15
    if len(captured_images) == NUM_IMAGES_TO_COLLECT:
        print(f"\n✅ Pengambilan {NUM_IMAGES_TO_COLLECT} gambar selesai.")
        
        # Minta nama untuk folder dataset
        person_name = input("➡️  Masukkan nama untuk folder dataset (contoh: Budi_Santoso): ").strip()

        if not person_name:
            print("❌ Nama tidak boleh kosong. Proses penyimpanan dibatalkan.")
            return

        # Ganti spasi dengan underscore untuk nama folder yang aman
        safe_folder_name = person_name.replace(" ", "_")
        target_dir = DATASET_DIR / safe_folder_name
        
        # Buat folder baru
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"\n💾 Menyimpan gambar ke folder '{target_dir}'...")
        for i, img in enumerate(captured_images):
            # Simpan setiap gambar dengan nama file yang unik
            filename = target_dir / f"{safe_folder_name}_{i+1}.jpg"
            cv2.imwrite(str(filename), img)
        
        print(f"\n🎉 Berhasil! {NUM_IMAGES_TO_COLLECT} gambar untuk '{person_name}' telah disimpan.")
        print("\nlangkah selanjutnya: Jalankan 'python -m backend.train_model' untuk melatih data baru ini.")
    
    else:
        print("\n❌ Pengambilan gambar tidak selesai. Tidak ada gambar yang disimpan.")

if __name__ == "__main__":
    collect_new_person()