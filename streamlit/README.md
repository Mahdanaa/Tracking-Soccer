# Football Player Tracking - Frontend

Frontend aplikasi untuk tracking pemain sepak bola menggunakan Streamlit.

## Instalasi

1. Pastikan Python 3.9+ sudah terinstal.
2. Masuk ke folder `streamlit`:
   ```bash
   cd streamlit
   ```
3. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## Menjalankan Aplikasi

Jalankan perintah berikut:
```bash
streamlit run app.py
```

Aplikasi akan berjalan di `http://localhost:8502`.

## Konfigurasi Backend

Aplikasi membutuhkan backend API untuk memproses video. Konfigurasi dapat dilakukan melalui file `.env` di folder `streamlit` atau via Streamlit Secrets.

**Variabel Lingkungan:**
- `API_BASE_URL`: URL dasar backend (contoh: `http://localhost:8000`).
- `API_TIMEOUT`: Timeout request ke API dalam detik (default: `30`).

## Endpoint yang Dibutuhkan Backend

Frontend mengharapkan endpoint berikut tersedia di backend:

1. `POST /api/analyses` — Upload file video, respons `{"analysis_id": "string", "status": "string"}`.
2. `GET /api/analyses/{id}/status` — Ambil status proses, respons `{"status": "string", "progress": number}`.
3. `GET /api/analyses/{id}/results` — Ambil hasil analisis, respons `{"video_url": "string", "radar_url": "string", "summary": {}}`.
4. `GET /api/analyses` — Ambil riwayat analisis, respons `{"analyses": [{"analysis_id", "filename", "date", "status"}]}`.

## Cara Menggunakan

1. Pastikan `API_BASE_URL` dikonfigurasi.
2. Buka aplikasi di browser.
3. Di Dashboard, klik atau seret video ke area upload.
4. Pastikan file format MP4, AVI, atau MOV dan ukuran ≤ 500 MB.
5. Klik "Mulai Analisis" untuk mengirim video dan memulai proses analisis.
6. Ikuti status proses di halaman Processing.
7. Lihat hasil di halaman Results.
8. Lihat riwayat di halaman History.