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

```bash
git clone https://github.com/Mahdanaa/soccer-cihuy.git
cd soccer-cihuy
pip install -e .
pip install -r examples/soccer/requirements.txt
```

> [!NOTE]
> Jika ingin menjalankan `main_onnx.py`, pastikan `onnxruntime` sudah terpasang (`pip install onnxruntime`).

### Cek Kesiapan Environment

Jalankan skrip pengecekan sebelum memproses video:

```bash
python check_env.py
```

---

## 🚀 Cara Menjalankan

> [!IMPORTANT]
> Folder `examples/soccer/data/` tidak ikut di-commit ke GitHub karena ukuran file model `.pt` (`> 130 MB`) dan video sampel cukup besar. Unduh model bawaan melalui `examples/soccer/setup.sh` atau latih model `.onnx` sendiri mengikuti [PANDUAN_RETRAINING_KAGGLE_ONNX.md](PANDUAN_RETRAINING_KAGGLE_ONNX.md).

### 1. Mode GPU Teroptimasi (`main_fast.py` — `.pt`)

Cocok untuk laptop/PC dengan GPU NVIDIA VRAM ~4 GB:

```bash
python examples/soccer/main_fast.py `
  --source_video_path examples/soccer/data/08fd33_4.mp4 `
  --target_video_path examples/soccer/data/out-fast-radar.mp4 `
  --device cuda `
  --mode RADAR `
  --imgsz 960 `
  --no_preview
```

### 2. Mode Ringan CPU / ONNX (`main_onnx.py` — `.onnx`)

Membutuhkan 3 file model di `examples/soccer/data/`:

- `football-player-detection.onnx`
- `football-pitch-detection.onnx`
- `football-ball-detection.onnx`

Jalankan dengan:

```bash
python examples/soccer/main_onnx.py `
  --source_video_path examples/soccer/data/2e57b9_0.mp4 `
  --target_video_path examples/soccer/data/out-onnx-radar.mp4 `
  --device cpu `
  --mode RADAR `
  --no_preview
```

Pilihan `--mode` yang tersedia pada kedua skrip:

- `PITCH_DETECTION`
- `PLAYER_DETECTION`
- `BALL_DETECTION`
- `PLAYER_TRACKING`
- `TEAM_CLASSIFICATION`
- `RADAR`

---

## ⚽ Datasets (Roboflow Universe)

| Use Case                           | Dataset                                                                                                                                                          |
| :--------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ⚽ Soccer player detection         | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) |
| ⚽ Soccer ball detection           | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg)    |
| ⚽ Soccer pitch keypoint detection | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)   |

---

## 📄 Lisensi & Kredit

Mengikuti lisensi repository asli **[roboflow/sports](https://github.com/roboflow/sports/blob/main/LICENSE)**. Terima kasih kepada tim Roboflow atas library dan dataset open-source-nya.
