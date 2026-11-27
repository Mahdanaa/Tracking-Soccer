<div align="center">
  <h1>⚽ Soccer Tracking</h1>
  <p>Deteksi pemain, pelacakan bola, dan minimap pertandingan sepak bola. Aplikasi Streamlit ini menjalankan tiga model ONNX secara lokal dengan deteksi otomatis GPU (CUDA / ROCm / Apple Silicon / Windows ML) atau fallback ke CPU.</p>
</div>

---

## 🔗 Referensi proyek

Proyek ini mengadaptasi Roboflow Sports untuk menganalisis rekaman pertandingan dan mengekspor hasilnya sebagai video MP4. Kode pendukung Roboflow ada di `sports/`, sedangkan alur analisis aplikasi ada di `src/`. [Roboflow Sports](https://github.com/roboflow/sports)

---

## 📂 Struktur folder

```text ad
Tracking-Soccer/
├── app.py                                    # Antarmuka Streamlit
├── requirements.txt                          # Versi dependensi aplikasi
├── setup.py                                  # Metadata paket Python sports
├── README.md                                 # Panduan instalasi dan penggunaan
├── LICENSE                                   # Lisensi kode proyek
├── .gitignore                                # Aturan untuk berkas lokal dan hasil analisis
├── models/
│   └── .gitkeep                              # Tempat tiga model ONNX dari tim
├── samples/
│   └── .gitkeep                              # Tempat video MP4 untuk pilihan Sample
├── outputs/
│   └── .gitkeep                              # Tempat hasil MP4 dan berkas sementara
├── src/
│   ├── __init__.py
│   ├── config.py                             # Path, pilihan mode, warna, dan ambang
│   ├── device.py                             # Deteksi otomatis GPU (CUDA/ROCm/MPS/Windows ML) atau CPU
│   ├── models.py                             # Memuat dan memvalidasi model ONNX
│   ├── processor.py                          # Menjalankan analisis dan melaporkan progres
│   ├── video.py                              # Membaca frame dan menulis output H.264
│   ├── core/
│   │   ├── __init__.py
│   │   ├── player.py                         # Deteksi manusia dan ID ByteTrack
│   │   ├── ball.py                           # Deteksi, pelacakan, dan jejak bola
│   │   ├── pitch.py                          # Deteksi landmark dan proyeksi minimap
│   │   ├── team.py                           # Pengelompokan pemain dan kiper berdasarkan tim
│   │   └── annotations.py                    # Kotak, label, ID, dan overlay pada video
│   └── modes/
│       ├── __init__.py                       # Menghubungkan nama mode ke fungsi pemrosesan
│       ├── context.py                        # Data dan peringatan untuk satu analisis
│       ├── pitch_detection.py
│       ├── player_detection.py
│       ├── ball_detection.py
│       ├── player_tracking.py
│       ├── team_classification.py
│       ├── radar.py
│       └── full_match_analysis.py
├── sports/
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── ball.py                           # Tracker dan jejak bola
│   │   ├── team.py                           # Ekstraksi fitur jersey dengan SigLIP dan clustering
│   │   └── view.py                           # Transformasi perspektif kamera
│   ├── configs/
│   │   ├── __init__.py
│   │   └── soccer.py                         # Dimensi lapangan dan definisi 32 landmark
│   └── annotators/
│       ├── __init__.py
│       └── soccer.py                         # Menggambar lapangan dan titik di minimap
├── scripts/
│   ├── setup_env.py                          # Instalasi dependensi otomatis sesuai hardware (GPU/CPU)
│   └── check_models.py                       # Diagnostik model dengan satu frame video
└── examples/
    └── soccer/
        └── main.py                           # Menjalankan analisis lewat terminal
```

Git membawa `.gitkeep` agar folder `models/`, `samples/`, dan `outputs/` tersedia setelah clone. Simpan model, video masukan, dan hasil analisis di folder lokal masing-masing. Langkah instalasi di bawah akan membuat `.venv/` untuk Python dan library proyek.

---

## 💻 Instalasi setelah clone

### 1. Siapkan Git dan Python

Gunakan **Windows x64 dan Python 3.14** untuk mengikuti perintah PowerShell di bawah. Setup proyek telah diuji pada Windows 11 dengan Python 3.14.5. Koneksi internet dibutuhkan untuk mengunduh dependensi dan menyiapkan cache SigLIP.

```powershell
git --version
py -3.14 --version
```

Jika `py` belum tersedia, jalankan `python --version`. Jika hasilnya Python 3.14, gunakan `python` untuk membuat virtual environment pada langkah 2. Unduh Git atau Python jika belum terpasang. [Git](https://git-scm.com/downloads) · [Python](https://www.python.org/downloads/)

### 2. Clone dan buat virtual environment

```powershell
git clone --branch Mode https://github.com/Mahdanaa/Tracking-Soccer.git
cd Tracking-Soccer
py -3.14 -m venv .venv
```

Jika memakai `python` langsung, perintah untuk langkah terakhir adalah `python -m venv .venv`. Jalankan perintah selanjutnya dari folder yang berisi `app.py`.

### 3. Instal dependensi (otomatis menyesuaikan GPU/CPU)

Jalankan skrip setup berikut agar paket disesuaikan otomatis dengan perangkat Anda:

```powershell
.\.venv\Scripts\python.exe -m scripts.setup_env
```

Skrip ini mendeteksi hardware secara otomatis:
- **NVIDIA GPU (Windows/Linux):** memasang PyTorch CUDA dan `onnxruntime-gpu[cuda]`.
- **AMD GPU (Linux ROCm):** memasang PyTorch ROCm dan ONNX Runtime.
- **Apple Silicon (macOS arm64):** memasang PyTorch MPS dan ONNX Runtime CoreML.
- **Windows 11 24H2+ (AMD/Intel/Qualcomm GPU):** memasang `onnxruntime-windowsml` dan PyTorch CPU.
- **Perangkat tanpa GPU yang didukung:** memasang konfigurasi CPU universal.

Jika ingin memasang versi **CPU saja** secara manual, jalankan:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Perintah di atas langsung memakai Python dari `.venv`, jadi bisa dijalankan tanpa mengaktifkan environment. Untuk mengaktifkannya di PowerShell, jalankan `.\.venv\Scripts\Activate.ps1`. Setelah itu, Anda bisa memakai `python` sebagai pengganti `.\.venv\Scripts\python.exe`.

Dependensi proyek sudah mencakup Streamlit, ONNX Runtime, OpenCV, dan `imageio-ffmpeg` untuk ekspor MP4.

### 4. Masukkan model ONNX dan siapkan video

Salin tiga berkas ONNX yang disediakan oleh tim proyek ke `models/`. Pertahankan nama berkas agar aplikasi dapat menemukannya.

| Berkas | Fungsi |
|---|---|
| `football-player-detection.onnx` | Mendeteksi pemain, kiper, wasit, dan bola. |
| `football-ball-detection.onnx` | Detektor khusus bola. |
| `football-pitch-detection.onnx` | Mendeteksi 32 landmark lapangan. |

Ketiga model menggunakan input 640 × 640. Siapkan satu video pertandingan dalam format MP4. Anda bisa mengunggahnya dari komputer saat aplikasi terbuka atau menyalinnya ke `samples/` agar muncul sebagai pilihan **Sample**. Perintah contoh di README ini memakai nama `pertandingan.mp4`.

### 5. Unduh SigLIP untuk mode yang memakai warna tim

Jalankan sekali untuk memakai **Team Classification**, **Radar**, atau **Full Match Analysis**. Gunakan akun Windows yang sama dengan akun yang menjalankan aplikasi agar cache dapat dibaca. [Model SigLIP](https://huggingface.co/google/siglip-base-patch16-224)

```powershell
.\.venv\Scripts\hf.exe download google/siglip-base-patch16-224 config.json preprocessor_config.json model.safetensors
```

**Player Tracking**, **Player Detection**, **Ball Detection**, dan **Pitch Detection** bisa langsung dijalankan dengan model ONNX yang sesuai.

---

## 🚀 Cara menjalankan Streamlit

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

1. Buka alamat lokal yang tercetak di terminal, biasanya `http://localhost:8501`.
2. Pilih **Upload** untuk mengunggah MP4 atau **Sample** untuk memilih video dari `samples/`. Batas unggahan adalah **200 MB**.
3. Aplikasi otomatis memilih device terbaik (`Auto (GPU ...)` atau `Auto (CPU)`) tanpa perlu dipilih manual. Untuk percobaan pertama, pilih **Player Tracking** dan gunakan klip pendek. Klik **Start Analysis**.
4. Setelah proses selesai, putar hasilnya di Streamlit atau klik **Download result** untuk menyimpan MP4. Video hasil memakai H.264 tanpa audio.
5. Tekan **Ctrl+C** di terminal untuk menghentikan server.

### Pilihan mode

| Mode | Hasil |
|---|---|
| **Pitch Detection** | Menggambar landmark dan garis acuan lapangan. |
| **Player Detection** | Menggambar kotak, kelas, dan confidence untuk pemain, kiper, wasit, serta bola. |
| **Ball Detection** | Menampilkan posisi bola yang terdeteksi dan jejaknya. |
| **Player Tracking** | Menampilkan ID pemain, kiper, dan wasit dengan ByteTrack. |
| **Team Classification** | Mengelompokkan pemain berdasarkan jersey dan memberi warna tim. |
| **Radar** | Menampilkan ID dan warna tim, serta memetakan posisi pemain ke minimap. |
| **Full Match Analysis** | Menjalankan Radar sekaligus menampilkan jejak bola dan posisi bola pada minimap. |

Jika kalibrasi lapangan gagal pada satu frame, Radar dan Full Match Analysis menyembunyikan minimap untuk frame itu. Ringkasan Streamlit mencatat jumlah frame yang terdampak.

### Alternatif melalui terminal

Simpan MP4 sebagai `samples/pertandingan.mp4`, atau ganti nilai `--source_video_path` dengan lokasi video Anda.

```powershell
.\.venv\Scripts\python.exe -m examples.soccer.main `
  --source_video_path samples/pertandingan.mp4 `
  --target_video_path outputs/pertandingan-tracking.mp4 `
  --mode PLAYER_TRACKING `
  --no_preview
```

Gunakan nama output baru setiap analisis. Secara default CLI memakai `--device auto` (deteksi GPU/CPU otomatis); tambahkan `--device cpu` jika ingin memaksa CPU. Hapus `--no_preview` untuk membuka preview setelah proses selesai. Untuk melihat nama mode dan argumen lainnya, jalankan `.\.venv\Scripts\python.exe -m examples.soccer.main --help`.

---

## 🧪 Pemeriksaan cepat

Jika video sudah ada di `samples/pertandingan.mp4`, perintah berikut memuat ketiga model dan menjalankan inferensi pada satu frame. Hasilnya menampilkan metadata, checksum, dan deteksi yang ditemukan.

```powershell
.\.venv\Scripts\python.exe -m scripts.check_models --video samples/pertandingan.mp4
```

| Kendala | Langkah pemeriksaan |
|---|---|
| Model ONNX gagal dimuat | Cek nama berkas di `models/`. Diagnosis akan menunjukkan lokasi berkas dan konfigurasi yang diharapkan. |
| Cache SigLIP belum lengkap | Ulangi perintah `hf download` dan pastikan cache diisi oleh akun Windows yang menjalankan aplikasi. |
| Analisis terasa lambat / masih berjalan di CPU | Pastikan paket GPU terpasang dengan menjalankan `python -m scripts.setup_env`. Pada perangkat tanpa GPU, mulai dari klip pendek dengan Player Tracking karena Full Match Analysis menjalankan 3 model dan beberapa potongan bola pada setiap frame. |
| Video gagal diputar atau disimpan | Pastikan MP4 masukan bisa didekode dan ada ruang tulis untuk berkas sementara di `outputs/`. |

---

## 📚 Dataset, lisensi, dan kredit

Model pemain, bola, dan lapangan merupakan hasil fine-tuning tim dari dataset Roboflow Universe. [Players v10](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) · [Ball v2](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg) · [Field v12](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)

Atribusi Roboflow Sports dan lisensi MIT kode proyek tercantum di `LICENSE`. Model YOLOv8s tim mencantumkan AGPL-3.0 dalam metadata. SigLIP memakai Apache-2.0. Periksa lisensi dataset dan hak distribusi rekaman sebelum membagikan video. [Lisensi proyek](LICENSE) · [Ultralytics](https://www.ultralytics.com/license) · [SigLIP](https://huggingface.co/google/siglip-base-patch16-224)
  