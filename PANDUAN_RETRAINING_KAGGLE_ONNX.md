# Panduan Retraining 3 Model `roboflow/sports` ke `YOLOv8s` + Export `ONNX` (via Kaggle GPU)

Panduan ini disusun langkah demi langkah dari **0 sampai siap pakai** untuk mengganti 3 model bawaan `roboflow/sports` (`YOLOv8x`, ~130–140 MB/model) menjadi **`YOLOv8s` format `.onnx` (`imgsz=640`)** agar jauh lebih ringan dan layak dijalankan di PC sekolah/lab berbasis CPU (seperti Intel Celeron / AMD setara) tanpa GPU.

---

## Ringkasan Target Model

| No | Tugas Model | Dataset Roboflow (`workspace/project/ver`) | Base Model | `imgsz` | File Output Akhir |
| :-: | :--- | :--- | :--- | :-: | :--- |
| **1** | Deteksi Pemain, Kiper, Wasit, Bola | `roboflow-jvuqo/football-players-detection-3zvbc` (`v10`) | `yolov8s.pt` | `640` | `football-player-detection.onnx` |
| **2** | Deteksi 32 Keypoint Lapangan (Radar) | `roboflow-jvuqo/football-field-detection-f07vi` (`v12`) | `yolov8s-pose.pt` | `640` | `football-pitch-detection.onnx` |
| **3** | Deteksi Bola Jarak Jauh | `roboflow-jvuqo/football-ball-detection-rejhg` (`v2`) | `yolov8s.pt` | `640` | `football-ball-detection.onnx` |

> [!TIP]
> Disarankan membuat **3 Notebook terpisah** di Kaggle (1 notebook per model) atau menjalankannya satu per satu agar tidak kehabisan kuota disk/RAM sesi Kaggle.

---

## Tahap 1: Persiapan Akun (Sekali Saja)

> [!NOTE]
> **Kapan `ROBOFLOW_API_KEY` Dipakai?**
> - `ROBOFLOW_API_KEY` (akun gratis) **hanya dipakai di Kaggle untuk mengunduh 3 dataset latihan secara otomatis lewat kode Python**.
> - Setelah model `.onnx` selesai dibuat dan dipindahkan ke PC lokal/sekolah, program **`examples/soccer/main.py` berjalan 100% offline tanpa API Key**.
> - **Alternatif tanpa API Key:** Anda juga bisa mengunduh ketiga dataset secara manual lewat browser dari link di `README.md` (pilih format **YOLOv8** `.zip`), lalu meng-upload-nya ke menu **Datasets** di Kaggle dan membaca path-nya dari `/kaggle/input/...`.

### 1.1 Buat API Key Roboflow (Gratis — Untuk Cara Otomatis)
1. Daftar / login ke [Roboflow](https://app.roboflow.com/).
2. Klik ikon profil di pojok kanan atas → **Settings** → **Roboflow API**.
3. Salin **Private API Key** Anda (akan dipakai untuk mengunduh 3 dataset di Kaggle).

### 1.2 Siapkan Kaggle Notebook
1. Login ke [Kaggle](https://www.kaggle.com/) (pastikan akun sudah verifikasi nomor HP agar fitur **Internet** dan **GPU** aktif).
2. Klik **Create** → **New Notebook**.
3. Di panel kanan (**Notebook options / Settings**):
   - **Accelerator**: pilih **`GPU P100`** atau **`GPU T4 x2`** (jangan *None*).
   - **Internet**: ubah ke **`On`** (wajib agar bisa install library dan download dataset).
   - **Persistence**: opsional, bisa set *Files only* agar output aman jika sesi terputus.
4. Simpan API Key Roboflow dengan aman di Kaggle:
   - Pada menu atas Notebook, klik **Add-ons** → **Secrets** → **Add a new secret**.
   - **Label**: `ROBOFLOW_API_KEY`
   - **Value**: tempel API Key Roboflow Anda → centang kotak aktif di sebelahnya.

---

## Tahap 2: Setup Environment di Kaggle (Cell 1)

Buat **Cell 1** di Notebook Kaggle dan jalankan kode berikut untuk menginstal `ultralytics`, `roboflow`, `onnx`, dan `onnxslim`, serta mematikan prompt interaktif `wandb` agar training tidak macet menunggu input:

```python
!pip install -q ultralytics roboflow onnx onnxruntime onnxslim

import os
from kaggle_secrets import UserSecretsClient
from roboflow import Roboflow

# Matikan W&B agar tidak meminta login di tengah training
!yolo settings wandb=False

ROBOFLOW_API_KEY = UserSecretsClient().get_secret("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)

print("Setup selesai!")
```

---

## Tahap 3: Training & Export ONNX (3 Model)

Jalankan bagian **3A**, **3B**, dan **3C** berikut (boleh di 3 notebook terpisah atau berurutan di notebook yang sama).

### 3A. Model 1 — Player Detection (`football-player-detection.onnx`)

**Cell 2A — Download Dataset Pemain:**
```python
os.makedirs("/kaggle/working/datasets/player", exist_ok=True)
%cd /kaggle/working/datasets/player

project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
dataset_player = project.version(10).download("yolov8")
print("Dataset player di:", dataset_player.location)
```

**Cell 3A — Training `YOLOv8s` (`imgsz=640`, `50 epochs`):**
```python
%cd /kaggle/working
!yolo task=detect mode=train model=yolov8s.pt data={dataset_player.location}/data.yaml batch=16 epochs=50 imgsz=640 name=player_v8s plots=True
```

**Cell 4A — Validasi & Export ke ONNX:**
```python
!yolo task=detect mode=val model=/kaggle/working/runs/detect/player_v8s/weights/best.pt data={dataset_player.location}/data.yaml imgsz=640
!yolo export model=/kaggle/working/runs/detect/player_v8s/weights/best.pt format=onnx imgsz=640

import shutil
shutil.copy("/kaggle/working/runs/detect/player_v8s/weights/best.onnx", "/kaggle/working/football-player-detection.onnx")
print("Siap download: /kaggle/working/football-player-detection.onnx")
```

---

### 3B. Model 2 — Pitch Keypoint Detection (`football-pitch-detection.onnx`)

> [!IMPORTANT]
> Model lapangan memakai `task=pose` (`yolov8s-pose.pt`) dan **wajib** menyertakan `mosaic=0.0` (sesuai notebook resmi Roboflow) agar 32 titik garis lapangan tidak terpotong oleh augmentasi *mosaic*.

**Cell 2B — Download Dataset Lapangan:**
```python
os.makedirs("/kaggle/working/datasets/pitch", exist_ok=True)
%cd /kaggle/working/datasets/pitch

project = rf.workspace("roboflow-jvuqo").project("football-field-detection-f07vi")
dataset_pitch = project.version(12).download("yolov8")
print("Dataset pitch di:", dataset_pitch.location)
```

**Cell 3B — Training `YOLOv8s-pose` (`imgsz=640`, `100 epochs`):**
```python
%cd /kaggle/working
!yolo task=pose mode=train model=yolov8s-pose.pt data={dataset_pitch.location}/data.yaml batch=16 epochs=100 imgsz=640 mosaic=0.0 name=pitch_v8s_pose plots=True
```

**Cell 4B — Validasi & Export ke ONNX:**
```python
!yolo task=pose mode=val model=/kaggle/working/runs/pose/pitch_v8s_pose/weights/best.pt data={dataset_pitch.location}/data.yaml imgsz=640
!yolo export model=/kaggle/working/runs/pose/pitch_v8s_pose/weights/best.pt format=onnx imgsz=640

import shutil
shutil.copy("/kaggle/working/runs/pose/pitch_v8s_pose/weights/best.onnx", "/kaggle/working/football-pitch-detection.onnx")
print("Siap download: /kaggle/working/football-pitch-detection.onnx")
```

---

### 3C. Model 3 — Ball Detection (`football-ball-detection.onnx`)

**Cell 2C — Download Dataset Bola:**
```python
os.makedirs("/kaggle/working/datasets/ball", exist_ok=True)
%cd /kaggle/working/datasets/ball

project = rf.workspace("roboflow-jvuqo").project("football-ball-detection-rejhg")
dataset_ball = project.version(2).download("yolov8")
print("Dataset ball di:", dataset_ball.location)
```

**Cell 3C — Training `YOLOv8s` (`imgsz=640`, `50 epochs`):**
```python
%cd /kaggle/working
!yolo task=detect mode=train model=yolov8s.pt data={dataset_ball.location}/data.yaml batch=16 epochs=50 imgsz=640 name=ball_v8s plots=True
```

**Cell 4C — Validasi & Export ke ONNX:**
```python
!yolo task=detect mode=val model=/kaggle/working/runs/detect/ball_v8s/weights/best.pt data={dataset_ball.location}/data.yaml imgsz=640
!yolo export model=/kaggle/working/runs/detect/ball_v8s/weights/best.pt format=onnx imgsz=640

import shutil
shutil.copy("/kaggle/working/runs/detect/ball_v8s/weights/best.onnx", "/kaggle/working/football-ball-detection.onnx")
print("Siap download: /kaggle/working/football-ball-detection.onnx")
```

---

## Tahap 4: Unduh Hasil dari Kaggle

1. Di panel kanan Kaggle, buka tab **Output** → folder **`/kaggle/working`**.
2. Unduh ketiga file berikut (masing-masing berukuran sekitar ~43 MB):
   - `football-player-detection.onnx`
   - `football-pitch-detection.onnx`
   - `football-ball-detection.onnx`
3. *(Opsional)* Simpan juga file `best.pt` masing-masing sebagai cadangan jika suatu saat ingin export ulang ke format lain (misal OpenVINO).

---

## Tahap 5: Integrasi ke Repo `roboflow/sports` (Lokal / PC Sekolah)

### 5.1 Pasang Dependensi ONNX di PC Lokal
Aktifkan `.venv` di folder `sports`, lalu instal `onnxruntime`:
```powershell
cd D:\semester_5\project-senin\sports
.\.venv\Scripts\pip.exe install onnx onnxruntime
```

### 5.2 Letakkan File Model `.onnx`
Salin ketiga file `.onnx` hasil unduhan ke folder:
`sports\examples\soccer\data\`

Sehingga di dalamnya terdapat:
- `examples/soccer/data/football-player-detection.onnx`
- `examples/soccer/data/football-pitch-detection.onnx`
- `examples/soccer/data/football-ball-detection.onnx`

### 5.3 Penyesuaian di `examples/soccer/main.py`
Buka [examples/soccer/main.py](file:///D:/semester_5/project-senin/sports/examples/soccer/main.py) dan lakukan 3 penyesuaian berikut:

1. **Arahkan path model ke file `.onnx`** (baris 19–21):
   ```python
   PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.onnx')
   PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.onnx')
   BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.onnx')
   ```

2. **Ubah seluruh `imgsz=1280` menjadi `imgsz=640`** agar cocok dengan input model ONNX yang diekspor (terdapat di fungsi `run_player_detection`, `run_player_tracking`, `run_team_classification`, dan `run_radar`):
   ```python
   # Sebelum:
   result = player_detection_model(frame, imgsz=1280, verbose=False)[0]

   # Sesudah:
   result = player_detection_model(frame, imgsz=640, verbose=False)[0]
   ```

3. **Wajib untuk CPU/Celeron (jika memakai mode `TEAM_CLASSIFICATION` atau `RADAR`):**
   - Model `YOLO` sudah ringan dengan ONNX, **tetapi** `TeamClassifier` di [`sports/common/team.py`](file:///D:/semester_5/project-senin/sports/sports/common/team.py) masih memanggil **`SigLIP` + `UMAP`** di setiap frame (~4,6 detik/frame di CPU!).
   - Agar mode `RADAR` tidak macet di PC Celeron, tambahkan **cache tim berdasarkan `tracker_id`** di loop `run_radar` (prediksi tim hanya dijalankan 1× saat `tracker_id` baru pertama kali muncul, tidak diulang tiap frame) atau ganti `TeamClassifier` dengan metode warna baju K-Means sederhana.

### 5.4 Uji Jalankan di CPU Lokal
```powershell
cd D:\semester_5\project-senin\sports
.\.venv\Scripts\python.exe examples\soccer\main.py `
  --source_video_path examples\soccer\data\2e57b9_0.mp4 `
  --target_video_path examples\soccer\data\out-onnx-test.mp4 `
  --device cpu `
  --mode PLAYER_TRACKING
```

---

## Troubleshooting Umum di Kaggle
- **Error `CUDA out of memory` saat training**: Turunkan `batch=16` menjadi `batch=8`.
- **Sesi Kaggle terputus karena batas waktu**: Jalankan 1 model per notebook; estimasi waktu per model di GPU T4/P100 sekitar 15–35 menit (tergantung jumlah epoch dan ukuran dataset).
- **Error saat `YOLO("...onnx")` di PC lokal**: Pastikan `onnxruntime` sudah terinstal di `.venv` yang dipakai dan `imgsz` saat inferensi sama dengan `imgsz` saat export (`640`).
