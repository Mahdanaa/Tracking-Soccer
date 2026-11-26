<div align="center">

  <h1>⚽ Soccer Tracking</h1>

  <p>
    Kustomisasi dan optimasi modul analisis sepak bola berbasis
    <a href="https://github.com/roboflow/sports"><strong>roboflow/sports</strong></a>
    untuk GPU VRAM terbatas (4 GB) dan CPU‑only (ONNX Runtime).
  </p>

  <!-- License badge -->
  ![License](https://img.shields.io/github/license/Mahdanaa/soccer-cihuy)

</div>

---

## 🔗 Referensi Project Awal

Repository ini merupakan pengembangan/turunan dari project open‑source **[roboflow/sports](https://github.com/roboflow/sports)** oleh **Roboflow**. Seluruh core library (`sports/`), konfigurasi lapangan, dan dataset awal merujuk pada repository upstream tersebut.

---

## 📂 Folder Structure

```text
Tracking-Soccer/
├─ main.py                     # Entry point utama (CLI) – contoh A & B
├─ sports/                     # Core library (fork dari roboflow/sports)
├─ examples/
│   └─ soccer/
│       ├─ main_fast.py        # Runner teroptimasi untuk GPU (VRAM ≈ 4 GB)
│       ├─ main_onnx.py        # Runner ringan untuk CPU/ONNX
│       ├─ setup.sh            # Skrip download model .onnx/.pt
│       ├─ requirements.txt    # Dependensi khusus contoh
│       ├─ config.py           # Parameter default (path data, threshold, dll.)
│       ├─ README.md           # Penjelasan singkat contoh & cara pakai
│       ├─ data/               # Video contoh & model (git‑ignored)
│       └─ notebooks/          # Jupyter notebooks untuk training model
├─ check_env.py                # Skrip diagnostik environment lokal
├─ PANDUAN_RETRAINING_KAGGLE_ONNX.md
├─ setup.py                    # File setup standar setuptools
├─ LICENSE                     # Lisensi MIT (lihat bagian Lisensi)
├─ README.md
└─ .gitignore
```

Folder `examples/soccer/data/` tidak termasuk dalam Git karena berisi file model (.pt > 130 MB) dan video berukuran besar. Unduh model melalui `setup.sh` atau latih model Anda sendiri.

---

## 💻 Instalasi

Gunakan lingkungan **Python ≥ 3.8**:

```bash
# Clone repo
git clone https://github.com/Mahdanaa/soccer-cihuy.git
cd soccer-cihuy

# Optional: buat virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Instalasi paket utama (memanggil setup.py)
pip install -e .
# atau alternatif klasik
# python setup.py install

# Instalasi dependensi contoh
pip install -r examples/soccer/requirements.txt
```

> **Catatan**
> Jika ingin menjalankan `main_onnx.py`, pastikan `onnxruntime` sudah terpasang (`pip install onnxruntime`).

### Cek Kesiapan Environment

`check_env.py` memverifikasi:
- Versi Python dan pip
- Ketersediaan driver NVIDIA (`nvidia-smi`)
- Instalasi PyTorch & CUDA (jika ada GPU)
- Keberadaan model contoh & video sampel

Jalankan skrip diagnostik sebelum memproses video:

```bash
python check_env.py
```

---

## 🚀 Cara Menjalankan

> **Penting**
> Folder `examples/soccer/data/` tidak ikut di‑commit ke GitHub karena ukuran file model `.pt` (> 130 MB) dan video sampel cukup besar. Unduh model bawaan melalui `examples/soccer/setup.sh` atau latih model `.onnx` sendiri mengikuti [PANDUAN_RETRAINING_KAGGLE_ONNX.md](PANDUAN_RETRAINING_KAGGLE_ONNX.md).

### 1. `main.py` – Entry point utama

`main.py` menyediakan antarmuka CLI yang menerima argumen `--source_video_path`, `--target_video_path`, `--mode`, dan opsi tambahan seperti `--no_preview`. Dua mode yang paling sering dipakai adalah:

#### A. Mode 1 – **Tracking + Warna Tim** (`TEAM_CLASSIFICATION`)

```bash
python main.py \
  --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/out-team.mp4 \
  --mode TEAM_CLASSIFICATION \
  --no_preview
```
*Menandai pemain berdasarkan tim (warna seragam) dan menuliskan label tim pada tiap pemain dalam video output.*

#### B. Mode 2 – **Tracking + Warna Tim + Radar 2D** (`RADAR`)

```bash
python main.py \
  --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/out-radar.mp4 \
  --mode RADAR \
  --no_preview
```
*Selain menandai tim, menambahkan visualisasi radar 2‑dimensi yang menunjukkan posisi relatif pemain di lapangan.*

---

### 2. Mode GPU Teroptimasi (`main_fast.py` — `.pt`)

Cocok untuk laptop/PC dengan GPU NVIDIA VRAM ≈ 4 GB:

```bash
python examples/soccer/main_fast.py \
  --source_video_path examples/soccer/data/08fd33_4.mp4 \
  --target_video_path examples/soccer/data/out-fast-radar.mp4 \
  --device cuda \
  --mode RADAR \
  --imgsz 960 \
  --no_preview
```

### 3. Mode Ringan CPU / ONNX (`main_onnx.py` — `.onnx`)

Membutuhkan tiga file model di `examples/soccer/data/`:
- `football-player-detection.onnx`
- `football-pitch-detection.onnx`
- `football-ball-detection.onnx`

```bash
python examples/soccer/main_onnx.py \
  --source_video_path examples/soccer/data/2e57b9_0.mp4 \
  --target_video_path examples/soccer/data/out-onnx-radar.mp4 \
  --device cpu \
  --mode RADAR \
  --no_preview
```

---

## 📂 Additional Resources

* **`examples/soccer/README.md`** – Menjelaskan struktur contoh, cara menyiapkan data, dan perintah cepat untuk menjalankan demo.
* **`examples/soccer/config.py`** – Menyimpan parameter default (mis. `DATA_ROOT`, `CONFIDENCE_THRESHOLD`). Anda dapat meng‑override dengan argumen CLI bila diperlukan.
* **`examples/soccer/setup.sh`** – Skrip bash/PowerShell yang mengunduh model `.onnx` dan `.pt` ke folder `data/`. Jalankan dengan:
  ```bash
  bash examples/soccer/setup.sh   # Linux/macOS
  # atau pada PowerShell
  .\examples\soccer\setup.sh
  ```
* **Jupyter notebooks** (`examples/soccer/notebooks/…`) – Notebook untuk melatih masing‑masing detektor (player, ball, pitch). Buka dengan `jupyter notebook` dan pilih notebook yang diinginkan.

---

## 🧪 Testing / Quick Verification

Tidak ada suite tes terintegrasi, namun Anda dapat memverifikasi instalasi dengan satu perintah sederhana yang memanggil library:

```bash
python -c "import sports, importlib; print('sports version:', sports.__version__)"
```
Jika tidak terjadi error, instalasi berhasil. Untuk menjalankan contoh cepat (tanpa training), gunakan perintah `python main.py` sebagaimana dijelaskan pada bagian *Cara Menjalankan*.

---

## 📂 Pilihan `--mode`

- `PITCH_DETECTION`
- `PLAYER_DETECTION`
- `BALL_DETECTION`
- `PLAYER_TRACKING`
- `TEAM_CLASSIFICATION`
- `RADAR`

---

## 📚 Datasets (Roboflow Universe)

| Use Case | Dataset |
| :--- | :--- |
| ⚽ Soccer player detection | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) |
| ⚽ Soccer ball detection | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg) |
| ⚽ Soccer pitch keypoint detection | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi) |

---

## 📄 Lisensi & Kredit

Lisensi MIT (lihat file `LICENSE`).

Mengikuti lisensi repository asli **[roboflow/sports](https://github.com/roboflow/sports/blob/main/LICENSE)**. Terima kasih kepada tim Roboflow atas library dan dataset open‑source‑nya.

---
