<div align="center">

  <h1>⚽ soccer-cihuy</h1>

  <p>
    Kustomisasi dan optimasi modul analisis sepak bola berbasis
    <a href="https://github.com/roboflow/sports"><strong>roboflow/sports</strong></a>
    untuk GPU VRAM terbatas (4 GB) dan CPU-only (ONNX Runtime).
  </p>

</div>

---

## 🔗 Referensi Project Awal

Repository ini merupakan pengembangan/turunan dari project open-source **[roboflow/sports](https://github.com/roboflow/sports)** oleh **Roboflow**. Seluruh core library (`sports/`), konfigurasi lapangan, dan dataset awal merujuk pada repository upstream tersebut.

---

## ✨ Tambahan & Perubahan di Repository Ini

Dibandingkan repository asli `roboflow/sports`, terdapat beberapa file dan fitur tambahan:

| File | Deskripsi |
| :--- | :--- |
| `examples/soccer/main_fast.py` | Runner teroptimasi untuk GPU VRAM 4 GB (mis. RTX 2050) dengan dukungan `--imgsz` (default `960`), `--no_preview`, dan `HomographySmoother` (EMA) untuk menstabilkan radar lapangan. |
| `examples/soccer/main_onnx.py` | Runner ringan menggunakan model `YOLOv8s` format `.onnx` (`imgsz=640`) via ONNX Runtime agar bisa dijalankan di PC/CPU tanpa GPU. |
| `PANDUAN_RETRAINING_KAGGLE_ONNX.md` | Panduan lengkap retraining 3 model bawaan (`YOLOv8x` `.pt`) menjadi `YOLOv8s` `.onnx` menggunakan Kaggle GPU. |
| `check_env.py` | Skrip diagnosa environment lokal (Python, `nvidia-smi`, dependensi paket, PyTorch CUDA, serta pengecekan model & video sampel di `examples/soccer/data/`). |

---

## 💻 Instalasi

Gunakan lingkungan [**Python >= 3.8**](https://www.python.org/):

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

| Use Case | Dataset |
| :--- | :--- |
| ⚽ Soccer player detection | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) |
| ⚽ Soccer ball detection | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg) |
| ⚽ Soccer pitch keypoint detection | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi) |

---

## 📄 Lisensi & Kredit

Mengikuti lisensi repository asli **[roboflow/sports](https://github.com/roboflow/sports/blob/main/LICENSE)**. Terima kasih kepada tim Roboflow atas library dan dataset open-source-nya.
