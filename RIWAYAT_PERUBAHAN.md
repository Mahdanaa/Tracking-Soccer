# Riwayat Perubahan Soccer Tracking

Catatan ini merangkum pekerjaan yang dibahas dalam percakapan, mulai dari audit awal hingga usulan pesan commit. Nama file dan status Git menggambarkan kondisi saat masing-masing langkah dibahas. Perubahan yang dibuat sendiri setelah pembahasan terakhir berada di luar cakupan catatan ini.

## 1. Kondisi awal

- Pekerjaan dimulai di branch `Mode` pada repo turunan Roboflow Sports. Alur ONNX saat itu berada di `examples/soccer/config.py`, `examples/soccer/pipeline.py`, dan `examples/soccer/main.py`.
- Working tree sudah punya perubahan pada README, konfigurasi, dan pipeline enam mode. Tiga model ONNX asli, dua video sumber, dan beberapa hasil analisis lama tersimpan lokal di `examples/soccer/data/`.
- Audit memeriksa struktur repo, dependensi, metadata ketiga model, dan perbedaan dengan upstream. Baseline Player Detection berhasil memproses satu frame video nyata dari `0bfacc_0.mp4`.

## 2. Refactor dan integrasi aplikasi

| Lokasi | Perubahan |
|---|---|
| `src/config.py` | Menyatukan lokasi file, nama mode, warna anotasi, dan ambang deteksi. |
| `src/models.py` | Memuat model sesuai mode, memvalidasi input dan output ONNX, serta menjalankan inferensi di CPU. |
| `src/video.py` | Membaca frame, menulis video sementara, meng-encode MP4 H.264, dan memvalidasi hasil. |
| `src/processor.py` | Menyatukan alur pemrosesan, progres, peringatan, serta hasil untuk Streamlit dan CLI. |
| `src/core/` | Memisahkan deteksi dan tracking pemain, deteksi dan tracking bola, landmark lapangan, klasifikasi tim, serta anotasi. |
| `src/modes/` | Menyediakan daftar mode dan fungsi pemrosesan untuk setiap mode. |
| `sports/` | Mempertahankan helper Roboflow. Klasifikasi tim memakai SigLIP lokal dengan UMAP dan K-Means. |
| `app.py` | Menambahkan Streamlit untuk memilih sumber, mengunggah MP4, memilih mode, melihat progres dan preview, serta mengunduh hasil. |
| `examples/soccer/main.py` | Mengubah CLI lama agar memakai `process_video()` yang sama dengan Streamlit. |
| `scripts/check_models.py` | Menambahkan pemeriksaan metadata, checksum, dan inferensi pada satu frame nyata. |
| `requirements.txt` | Menyatukan dependensi yang diuji pada Windows x64 dengan Python 3.14.5. |

Ketiga model tim dipindahkan ke `models/` dengan nama `football-player-detection.onnx`, `football-ball-detection.onnx`, dan `football-pitch-detection.onnx`. Ketiganya memakai input statis 640 × 640 dan berjalan dengan ONNX Runtime `CPUExecutionProvider`. Model lapangan menghasilkan 32 landmark. Video sumber `0bfacc_0.mp4` dan `contoh1.mp4` dipindahkan ke `samples/`, sedangkan output dan berkas sesi ditempatkan di `outputs/`.

Enam mode disatukan dalam aplikasi, yaitu Pitch Detection, Player Detection, Ball Detection, Player Tracking, Team Classification, dan Radar. Mode ketujuh, **Full Match Analysis**, memakai alur Radar yang sama dengan tambahan detektor bola khusus, jejak bola pada video, dan titik bola pada minimap. Player Tracking memakai ByteTrack untuk pemain, kiper, dan wasit.

Konfigurasi dan pipeline lama di `examples/soccer/` digantikan oleh `src/`. Skrip diagnostik lama, requirements khusus contoh, skrip setup lama, dan README contoh dihapus saat refactor.

## 3. Pengujian awal dan pembersihan

- Ketujuh mode diuji dengan model asli pada klip pertandingan dan menghasilkan MP4 H.264. Microsoft Edge memutar dan mengunduh hasil setiap mode melalui Streamlit. Klip kedua juga dipakai untuk smoke test.
- Rangkaian awal meluluskan 30 tes unit atau komponen dan tujuh smoke test dengan model asli. Benchmark CPU dijalankan pada klip 32 frame. Angka waktu dari benchmark tersebut dipakai untuk menjelaskan perbedaan beban antar-mode.
- Setelah diminta membersihkan repo, folder `tests/` dan `benchmark/` dihapus bersama klip uji, hasil render lama, log, dan cache proyek. Dependensi tes dilepas dari konfigurasi instalasi.
- Tiga model ONNX, dua video sumber, `.venv`, CLI, dan diagnostik model tetap tersedia secara lokal. Setelah pembersihan, Streamlit masih menampilkan tujuh mode dan Player Tracking berhasil mengekspor klip uji pendek. Berkas uji sementara dibersihkan setelah pengecekan.

## 4. Revisi Radar dan minimap

1. **Tampilan Radar disesuaikan dengan versi lama.** Label di atas video kembali menampilkan ID. Warna menunjukkan tim dan wasit. Overlay kembali menempati separuh lebar dan tinggi frame dengan opasitas 50%. Radius titik pemain di minimap dikembalikan dari 13 ke 20 piksel pada kanvas minimap.
2. **Pemetaan diperbaiki pada frame yang sulit.** Toleransi dikaitkan dengan resolusi input model. Jika RANSAC awal gagal, percobaan kedua memakai landmark dari frame yang sama. Saat `contoh1.mp4` dirender ulang, jumlah frame dengan pemetaan gagal turun dari 36 menjadi 6 dari 720 frame. Pada detik 6,71, seluruh 24 track kembali tampil di minimap.
3. **Teks peringatan di dalam video dihapus.** Untuk Radar dan Full Match Analysis, overlay minimap disembunyikan hanya pada frame yang gagal dipetakan dan kembali muncul saat pemetaan valid. Anotasi pemain dan bola tetap mengikuti mode yang dipilih. Jumlah kegagalan tampil di ringkasan Streamlit. Pada cuplikan Radar yang diuji, overlay tersembunyi tepat pada 4 dari 14 frame.

Dalam pembahasan performa, Radar pernah menyelesaikan video `contoh1.mp4` berdurasi 30 detik dalam 5 menit 38 detik pada CPU. Full Match Analysis lebih berat karena menambah inferensi bola per potongan frame. Opsi menurunkan resolusi atau FPS dibahas sebagai cara mempercepat percobaan.

## 5. Perbaikan deteksi dan tracking bola

- Pemeriksaan `contoh1.mp4` menemukan bahwa model kadang menganggap sepatu putih atau marka lapangan sebagai bola. Tracker lama ikut memasukkan semua kandidat ke rata-rata riwayat, sehingga kandidat palsu bisa menarik penanda menjauh dari bola.
- Tracker baru menilai confidence, kontras lokal, dan konsistensi gerak dengan memperhitungkan pergeseran kamera. Kandidat ber-confidence rendah bisa melanjutkan jalur yang sudah meyakinkan jika posisinya dekat dengan prediksi. Frame yang meragukan bisa dilewati, dan riwayat direset saat terdeteksi pergantian adegan.
- Perubahan dipakai bersama oleh **Ball Detection** dan **Full Match Analysis**. Bobot ketiga ONNX tetap memakai berkas asli tim. Ekspor MP4 diuji pada kedua mode.

Pada detik 1,13 di `contoh1.mp4`, penanda yang sebelumnya berpindah ke sepatu kembali mengikuti bola. Perbandingan manual pada 46 frame dari dua pertandingan menghasilkan angka berikut.

| Video | Penanda tepat sebelum | Penanda tepat sesudah |
|---|---:|---:|
| `contoh1.mp4` | 8 dari 26 | 24 dari 26 |
| `0bfacc_0.mp4` | 17 dari 20 | 18 dari 20 |

## 6. Dokumentasi dan persiapan commit

- README disusun ulang mengikuti referensi yang dikirim. Isinya mencakup struktur folder, clone branch `Mode`, pembuatan `.venv`, instalasi dependensi, penempatan tiga ONNX, penyiapan cache SigLIP, penggunaan Streamlit hingga unduhan hasil, contoh CLI, dan diagnostik model.
- `.gitignore` diperbarui untuk mengabaikan virtual environment, model lokal, video sampel, dan output. `.gitkeep` dipertahankan agar struktur folder ikut saat clone.
- `FOOTBALL_TRACKING_CODEX_IMPLEMENTATION.md`, `PANDUAN_RETRAINING_KAGGLE_ONNX.md`, dan `examples/soccer/notebooks/` dikeluarkan dari struktur README dan ditambahkan ke `.gitignore`. Panduan retraining dan tiga notebook yang sebelumnya dilacak Git dilepas dari indeks dan perubahannya di-stage. Semua berkas asli tetap ada secara lokal saat pengecekan.
- Pesan commit yang diusulkan pada pembahasan terakhir adalah `feat: add Streamlit soccer tracking with ONNX`.

## 7. Deteksi otomatis GPU dan setup environment lintas perangkat

Agar pemrosesan video berjalan jauh lebih cepat di perangkat ber-GPU namun tetap otomatis aman di CPU pada perangkat tanpa GPU yang didukung, aplikasi diubah dari CPU-only menjadi deteksi hardware otomatis tanpa pilihan manual di UI.

| Lokasi | Perubahan |
|---|---|
| `src/device.py` | Menambahkan `resolve_device()` untuk mendeteksi runtime terbaik yang aktif secara otomatis (`NVIDIA CUDA`, `AMD ROCm`, `Apple Silicon CoreML/MPS`, `Windows ML`, atau fallback `CPU`), `preload_gpu_libraries()` untuk memuat library CUDA/cuDNN di Windows/Linux, serta petunjuk ringan jika GPU NVIDIA terdeteksi tetapi environment masih memakai paket CPU. |
| `scripts/setup_env.py` | Menambahkan skrip instalasi otomatis satu perintah (`python -m scripts.setup_env`) yang memeriksa OS dan tipe GPU, melepas paket ONNX Runtime yang bentrok, memasang kombinasi PyTorch + ONNX Runtime yang sesuai (`cu130`/`cu128`/`cu126` + `onnxruntime-gpu[cuda]` untuk NVIDIA, ROCm untuk Linux AMD, MPS/CoreML untuk Apple Silicon, `onnxruntime-windowsml` untuk Windows 11 24H2+ non-NVIDIA, atau paket CPU universal), lalu menjalankan `pip check`. |
| `src/models.py` | Memperbarui `ModelStore` agar menerima `RuntimeDevice`, meneruskan `onnx_device` ke predictor Ultralytics (tetap mempertahankan `BALL_CONFIDENCE`), dan memverifikasi provider aktual pada `validate_session()` agar fallback CPU tidak pernah diklaim sebagai GPU. |
| `src/core/team.py` | Meneruskan `torch_device` ke `TeamClassifier` (`batch_size=32` pada GPU, `16` pada CPU). |
| `src/processor.py` | Mengubah default `device="auto"` dan menambahkan `device`, `device_label`, serta `backend` aktual pada `ProcessingResult`. |
| `app.py` | Menjalankan deteksi device otomatis sekali per sesi server memakai `@st.cache_resource` dan menampilkan status aktual pada UI (mis. `Device · Auto (GPU (CUDA — NVIDIA GeForce RTX 4060 Laptop GPU))`). |
| `examples/soccer/main.py` | Mengubah default `--device auto` (tetap mendukung `--device cpu`, `cuda`, atau `mps`). |
| `scripts/check_models.py` | Menambahkan argumen `--device` (default `auto`) dan mencetak informasi runtime yang terdeteksi. |
| `README.md` | Memperbarui struktur folder, langkah instalasi otomatis memakai `scripts.setup_env`, serta keterangan auto-detect device pada Streamlit dan CLI. |

### Pengujian dan hasil yang dijalankan

Pengujian dilakukan pada laptop lokal (**NVIDIA GeForce RTX 4060 Laptop GPU 8 GB**, Driver `617.14` / CUDA `13.4`, Windows 11 Build `26300`, Python `3.14.5`):

- Menjalankan `.\.venv\Scripts\python.exe -m scripts.setup_env`: mengganti `torch 2.15.0` (CPU) dan `onnxruntime 1.32.0` (CPU) dengan `torch 2.14.1+cu130`, `torchvision 0.29.1+cu130`, dan `onnxruntime-gpu 1.32.0` beserta library CUDA 13; `pip check` bersih tanpa konflik.
- Menjalankan `scripts.check_models` pada `--device auto` (`CUDAExecutionProvider`) dan `--device cpu` (`CPUExecutionProvider`): ketiga model ONNX lolos dengan hasil deteksi dan 9 landmark lapangan yang identik.
- Menguji simulasi matrix `resolve_device()` dan `setup_env.build_plan()` lintas hardware (CUDA 13, CUDA 12, ROCm, Apple MPS/CoreML, Windows ML, CPU fallback + validasi strict/fallback) serta Streamlit `AppTest`: seluruh pemeriksaan lolos.
- Perbandingan kecepatan loop processing pada klip 32 frame 1080p (`0bfacc_0`):

| Mode | Loop CPU sebelumnya | Loop GPU RTX 4060 | Peningkatan |
|---|---:|---:|---:|
| `BALL_DETECTION` | 18,83 dtk (1,70 FPS) | 5,05 dtk (6,33 FPS) | ~3,7× lebih cepat |
| `FULL_MATCH_ANALYSIS` | 28,11 dtk (1,14 FPS) | 7,97 dtk (4,02 FPS) | ~3,5× lebih cepat |

## 8. Stabilisasi minimap dan penyesuaian tata letak overlay

1. **Menghilangkan efek kedip-kedip minimap (`src/config.py`, `src/core/pitch.py`, `src/modes/radar.py`, `README.md`):**
   - Sebelumnya, setiap kali kalibrasi homography gagal pada satu frame (misalnya saat kamera pan atau blur sesaat), minimap disembunyikan pada frame tersebut sehingga terlihat muncul-hilang.
   - `detect_pitch_transform()` kini memakai validasi dua lapis pada output model pitch yang sama (1× inferensi per frame): threshold ketat (`KEYPOINT_CONFIDENCE = 0.5`, toleransi `5 px`) dan fallback lebih longgar (`0.3`, toleransi `8 px`) jika lapis pertama gagal.
   - `PitchStabilizer` menambahkan *smoothing* matriks homography antarframe (`HOMOGRAPHY_SMOOTHING = 0.65`, langsung *snap* saat pergantian adegan) dan mempertahankan homography terakhir dengan kompensasi pergeseran kamera (`cv2.phaseCorrelate`) saat frame gagal kalibrasi.
   - `run_radar()` menambahkan buffer awal hingga `PITCH_STARTUP_BUFFER = 30` frame agar frame paling awal tetap mendapat proyeksi dari kalibrasi valid pertama, sehingga minimap tampil mulus dari awal hingga akhir tanpa teks `"Minimap unavailable"`.
   - Hasil verifikasi: uji kompensasi pan mencatat error proyeksi 1,2 cm pada lapangan 120 m × 70 m; uji nyata `FULL_MATCH_ANALYSIS` GPU pada 24 frame `0bfacc_0.mp4` dan `contoh1.mp4` lulus dengan `warnings=()` dan minimap aktif pada 100% frame (`contoh1.mp4` berjalan 5,66 FPS).
2. **Memperkecil dan memindahkan posisi minimap (`src/core/annotations.py`):**
   - Sebelumnya overlay minimap berukuran 50% × 50% frame (`960 × 540 px` pada 1080p) di bagian tengah bawah.
   - `overlay_minimap()` diubah agar menjaga rasio aspek lapangan (~28% tinggi frame, yakni **`491 × 302 px`** pada 1080p), ditempatkan di **pojok kanan bawah** dengan margin rapi (`22 px` di 1080p, koordinat `x = 1407..1897`, `y = 756..1057`), dan opasitas `0.75` agar titik pemain serta bola tetap jelas.

## 9. Langkah singkat untuk rekan tim setelah pull/clone

1. Perbarui dependensi sesuai perangkat masing-masing (otomatis GPU atau CPU):
   ```powershell
   .\.venv\Scripts\python.exe -m scripts.setup_env
   ```
2. Pastikan tiga model ONNX ada di `models/` dan cache SigLIP sudah diunduh sekali (`hf download google/siglip-base-patch16-224 config.json preprocessor_config.json model.safetensors`).
3. Jalankan aplikasi (device dipilih otomatis; jika server Streamlit lama masih menyala, hentikan dulu dengan `Ctrl + C` lalu jalankan ulang):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python -m streamlit run app.py
   ```
4. Alternatif melalui CLI:
   ```powershell
   python -m examples.soccer.main --source_video_path samples/0bfacc_0.mp4 --target_video_path outputs/hasil_full.mp4 --mode FULL_MATCH_ANALYSIS --no_preview
   ```
