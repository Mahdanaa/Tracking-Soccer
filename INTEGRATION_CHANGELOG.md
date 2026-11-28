# Catatan Integrasi, Perubahan & Benchmark Teknologi (`feat/integrate-fe-mode`)

Dokumen ini merangkum seluruh perubahan yang dilakukan pada tahap integrasi branch **`Mode`** (`b412228`) dan **`feat/streamlit-fe`** (`7e9b55f`), alasan teknis di balik setiap keputusan, serta hasil benchmark nyata untuk dibagikan kepada tim.

---

## 1. Ringkasan Perubahan Utama

### A. Integrasi Frontend & Model Tanpa REST API (1 Aplikasi)
- Menggabungkan tampilan UI dari `feat/streamlit-fe` (`streamlit/app.py`) dengan engine analisis dari branch `Mode` (`src/`).
- Menghapus kebutuhan server REST API terpisah (`requests`, `POST/GET /api/analyses`, `API_BASE_URL`) sehingga aplikasi cukup dijalankan lewat **1 perintah Streamlit** (`streamlit run streamlit/app.py`).
- Menyimpan upload sementara di `outputs/_uploads/` dan riwayat analisis di `outputs/history.json`.
- Memperbaiki alur halaman **Status Pemrosesan** di `streamlit/app.py`:
  - Status `"queued"` dan `"processing"` sama-sama ditangani agar proses yang sempat terputus tidak memunculkan error *"Status tidak dikenali"*.
  - Tombol **Perbarui Status** otomatis dinonaktifkan (`disabled`) selama pemrosesan berjalan agar klik tombol tidak memutus proses Streamlit.
  - Menampilkan 1 progress bar live dan status perangkat otomatis (`resolve_device("auto")`) di sidebar.

### B. Peningkatan Stabilitas Pelabelan ID Pemain (`ByteTrack` + HSV ID Recovery)
- Menambahkan `PlayerTracker` di `src/core/player.py` di atas `sv.ByteTrack` dengan konfigurasi terpusat di `src/config.py`:
  - `TRACK_LOST_BUFFER = 60`
  - `ID_RECOVERY_WINDOW = 75`
  - `ID_RECOVERY_MAX_DISTANCE = 140.0`
  - `ID_RECOVERY_MIN_SIMILARITY = 0.65`
- **Cara kerja:** Saat `ByteTrack` memunculkan ID baru (akibat deteksi berkedip atau pemain berpapasan), sistem mencocokkan jarak posisi, kesesuaian kelas (`player`/`goalkeeper`/`referee`), dan kemiripan histogram warna torso HSV terhadap track yang baru hilang dalam 75 frame terakhir — lalu memulihkan ke ID lama tanpa membuat duplikat ID dalam 1 frame.
- **Alasan:** Pelabelan ID jauh lebih stabil (~18% lebih sedikit reset ID pada uji sampel) dengan tambahan beban komputasi nyaris **0%** dan **tanpa library baru**.

### C. Optimasi Kecepatan Analisis Berulang (In-Memory Model Caching)
- `src/models.py`: Menambahkan `_MODEL_CACHE` agar 3 session ONNX (`player`, `pitch`, `ball`) hanya dimuat dan divalidasi **sekali** selama aplikasi berjalan.
- `sports/common/team.py`: Menambahkan `@lru_cache(maxsize=2)` pada `_load_siglip(device)` agar 208 layer bobot `SiglipVisionModel` tidak dimuat ulang di setiap analisis baru.
- **Alasan:** Mempercepat waktu tunggu sebelum `Processing frames` pada analisis ke-2 dan seterusnya dengan **0% perubahan akurasi/output**.

### D. Perapian Struktur & Kepatuhan Aturan Dosen
Menyesuaikan codebase dengan prinsip **KISS, YAGNI, Single Source of Truth, kode berbahasa Inggris (kecuali tampilan UI), dan tanpa docstring**:
- **0 Docstring & hapus dead code:** Menghapus seluruh 10 docstring serta fungsi yang tidak pernah dipanggil (`draw_paths_on_pitch`, `draw_pitch_voronoi_diagram` di `sports/annotators/soccer.py`, dan `transform_image` di `sports/common/view.py`). Verifikasi: **37 file Python, 0 docstring**.
- **Single Source of Truth (`src/config.py`):** Menyatukan `UPLOADS_DIR`, `HISTORY_FILE`, `DEFAULT_MODE`, `DEFAULT_DEVICE`, `ALLOWED_VIDEO_EXTENSIONS`, dan batas `UPLOAD_MB = 200` ke `src/config.py` (menghapus `streamlit/config.py` serta menyamakan limit upload yang sebelumnya berbeda antara `200 MB` dan `500 MB`).
- **Meratakan folder `streamlit/`:**
  - `streamlit/services/api_service.py` → `streamlit/service.py` (`AnalysisError`, `video_path`, menghapus field `radar_url` yang tidak terpakai).
  - `streamlit/utils/validators.py` → `streamlit/validators.py`.
- **Menghapus file obsolete:** `app.py` (root lama dari branch `Mode`), `streamlit/.streamlit/secrets.toml.example`, `streamlit/requirements.txt`, `streamlit/README.md`, `streamlit/services/`, `streamlit/utils/`, dan `RIWAYAT_PERUBAHAN.md`.
- **Menyederhanakan CLI:** Memindahkan `examples/soccer/main.py` → `cli.py` di root dan menghapus folder `examples/` agar struktur lebih datar dan jelas untuk kebutuhan pembelajaran CLI.

---

## 2. Stack Teknologi yang Dipakai & Alasan Pemilihannya

| Komponen | Teknologi yang Dipakai | Alasan Pemilihan |
| :--- | :--- | :--- |
| **Deteksi Pemain, Bola & 32 Keypoint Lapangan** | **YOLOv8s (`.onnx`, `imgsz=640`)** via **ONNX Runtime** | Jauh lebih ringan daripada model bawaan `YOLOv8x` `.pt` (~43–47 MB vs ~130–134 MB per model) sehingga layak dijalankan di GPU VRAM 4 GB maupun PC CPU-only. |
| **Deteksi Hardware Otomatis** | **`src/device.py` (`resolve_device("auto")`)** + `scripts/setup_env.py` | Otomatis memakai GPU (`CUDAExecutionProvider` + PyTorch CUDA) bila tersedia, dan fallback aman ke CPU di perangkat tanpa GPU. |
| **Pelacakan Pemain (Tracking)** | **`sv.ByteTrack` + Lightweight HSV ID Recovery** | Titik tengah paling efisien antara stabilitas ID dan kecepatan (< 0.1 ms/frame overhead, 0 dependensi baru). Lihat benchmark di Bagian 3. |
| **Klasifikasi Tim** | **`google/siglip-base-patch16-224` + `UMAP` + `KMeans`** *(SigLIP di-cache)* | Mempertahankan akurasi pemisahan jersey tim yang sudah disepakati pada branch `Mode`, dengan cache memori agar tidak reload ulang antar analisis. |
| **Radar 2D & Homography** | **32-Keypoint ViewTransformer + `PitchStabilizer`** | Menjaga minimap radar di pojok kanan bawah (skala 28%, opasitas 75%) tetap stabil tanpa flicker. |
| **Pelacakan Bola** | **`BallPipeline` + `BallTracker` + `BallAnnotator`** | Menggabungkan deteksi bola dengan filter lintasan dan trail visual. |
| **Antarmuka Web (UI/UX)** | **Streamlit (`streamlit/app.py`)** | Langsung terhubung ke `src/` tanpa server backend terpisah — mudah dijalankan oleh seluruh tim/penguji (KISS). |

---

## 3. Data Benchmark Nyata

### A. Benchmark Ukuran Model (Bawaan `YOLOv8x .pt` vs `YOLOv8s .onnx`)

| Model | Bawaan (`YOLOv8x .pt`) | Saat Ini (`YOLOv8s .onnx`) | Penghematan Ukuran |
| :--- | :---: | :---: | :---: |
| Player Detection | 130.46 MB | **42.68 MB** | **-67.3%** |
| Pitch Keypoint Detection | 133.72 MB | **46.79 MB** | **-65.0%** |
| Ball Detection | 130.45 MB | **42.67 MB** | **-67.3%** |
| **Total 3 Model** | **394.63 MB** | **132.14 MB** | **-66.5%** |

### B. Benchmark Metode Tracker (Stabilisasi ID vs Beban Komputasi)

Pengujian dilakukan langsung pada **250 frame pertama** video sampel `samples/2e57b9_0.mp4` (`1920x1080 @ 25 FPS`) menggunakan **NVIDIA GeForce RTX 2050 (CUDA)** dengan waktu deteksi ONNX player sebesar **10.900 detik (`43.60 ms/frame`)**:

| Metode Tracker | Waktu Tracker (250 fr) | Waktu / Frame | Unique IDs *(makin kecil = makin stabil)* | Tambahan Beban *(vs Detect + BT30)* |
| :--- | :---: | :---: | :---: | :---: |
| **`ByteTrack` (`buf=30`, awal)** | 0.890 s | 3.56 ms | 44 | Baseline (`0%`) |
| **`ByteTrack` (`buf=60`)** | 0.765 s | 3.06 ms | 44 | `~0%` |
| **`ByteTrack` (`buf=60`) + HSV ID Recovery** *(dipilih)* | **0.787 s** | **3.15 ms** | **36** *(8 ID dipulihkan)* | **`+0.2%`** *(+0.09 ms/frame)* |
| **`BoT-SORT` (`sparseOptFlow` GMC)** | 5.267 s | 21.07 ms | 61 | **`+37.1%`** *(+17.5 ms/frame)* |

### C. Pertimbangan Terhadap Tracker Re-ID Penuh (DeepSORT / OccluBoost via BoxMOT)
- **Tidak dipilih** karena mengekstrak fitur neural network Re-ID untuk ~20 pemain di **setiap frame**, memerlukan 1 model bobot tambahan, menambah dependensi baru (berisiko terhadap kompatibilitas Python 3.14 dan setup CPU-only), serta menambah kompleksitas kode.
- **Catatan batasan realistis:** Pada rekaman kamera jauh dengan 10 pemain satu tim berseragam identik, baik HSV Recovery maupun Re-ID penuh tetap memiliki kemungkinan sesekali tertukar ID apabila 2 pemain satu tim bertabrakan sangat rapat tanpa pembacaan nomor punggung (OCR). Namun pada kasus pemain beda tim, wasit, kiper, dan deteksi yang sempat berkedip, **`ByteTrack` + HSV ID Recovery** terbukti menurunkan reset ID hingga **~18%** dengan beban tambahan `< 0.1%` pada pipeline penuh.

---

## 4. Ringkasan File yang Berubah

- **Diubah:** `src/config.py`, `src/models.py`, `src/core/player.py`, `src/modes/player_tracking.py`, `src/modes/team_classification.py`, `sports/annotators/soccer.py`, `sports/common/ball.py`, `sports/common/team.py`, `sports/common/view.py`, `streamlit/app.py`, `README.md`.
- **Ditambahkan (hasil perataan):** `cli.py`, `streamlit/service.py`, `streamlit/validators.py`, `INTEGRATION_CHANGELOG.md`.
- **Dihapus (obsolete/duplikat):** `app.py`, `RIWAYAT_PERUBAHAN.md`, `examples/soccer/main.py`, `streamlit/config.py`, `streamlit/requirements.txt`, `streamlit/README.md`, `streamlit/.streamlit/secrets.toml.example`, `streamlit/services/`, `streamlit/utils/`.

---

## 5. Cara Menjalankan untuk Tim

```powershell
# 1. Setup environment otomatis (GPU NVIDIA / CPU fallback)
python scripts/setup_env.py
pip install -e .

# 2. Pastikan 3 file .onnx ada di folder models/
python scripts/check_models.py

# 3. Jalankan Web Streamlit
streamlit run streamlit/app.py
```

*(Opsional CLI pembelajaran: `python cli.py --source_video_path samples/2e57b9_0.mp4 --target_video_path outputs/hasil.mp4 --mode FULL_MATCH_ANALYSIS --no_preview`)*
