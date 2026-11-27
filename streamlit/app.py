import time
import streamlit as st
from config import API_BASE_URL
from services.api_service import (
    BackendError,
    download_file,
    get_history,
    get_results,
    get_status,
    submit_video,
)
from utils.validators import validate_video

st.set_page_config(
    page_title="Football Player Tracking",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
section[data-testid="stSidebar"] {
  background-color: #0F1830;
  border-right: 1px solid #26335A;
}
.sb-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 4px 14px 4px;
  border-bottom: 1px solid #26335A;
  margin-bottom: 14px;
}
.sb-brand-title {
  font-size: 1.05rem;
  font-weight: 700;
  color: #E8ECF6;
  line-height: 1.2;
}
.sb-brand-sub {
  font-size: 0.72rem;
  color: #8D9AB8;
}
.sb-section-label {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #8D9AB8;
  margin: 4px 0 8px 2px;
}
.sb-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid #26335A;
  font-size: 0.76rem;
  color: #E8ECF6;
  margin-bottom: 6px;
}
.sb-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}
.sb-version {
  font-size: 0.68rem;
  color: #6B7897;
  padding-left: 4px;
}
</style>
""", unsafe_allow_html=True)

_LOGO_SVG = '<svg width="30" height="22" viewBox="0 0 28 20" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="1" y="1" width="26" height="18" rx="3" stroke="#22C55E" stroke-width="1.6"/><line x1="14" y1="1" x2="14" y2="19" stroke="#22C55E" stroke-width="1.2"/><circle cx="14" cy="10" r="4" stroke="#22C55E" stroke-width="1.2"/><circle cx="14" cy="10" r="1.2" fill="#22C55E"/></svg>'

STATUS_LABELS = {
    "uploading": "Mengunggah",
    "queued": "Dalam Antrean",
    "processing": "Sedang Diproses",
    "completed": "Selesai",
    "failed": "Gagal",
}


def init_state():
    defaults = {
        "nav_page": "Dashboard",
        "workflow_view": None,  # None | "processing" | "results"
        "analysis_id": None,
        "uploaded_filename": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_to_dashboard():
    st.session_state.nav_page = "Dashboard"
    st.session_state.workflow_view = None
    st.session_state.analysis_id = None
    st.session_state.uploaded_filename = None


def render_sidebar():
    is_dash = st.session_state.nav_page == "Dashboard" and st.session_state.workflow_view is None
    is_hist = st.session_state.nav_page == "Riwayat" and st.session_state.workflow_view is None

    dot_color = "#22C55E" if API_BASE_URL else "#FF6B57"
    pill_label = "Backend terhubung" if API_BASE_URL else "Backend belum diatur"

    with st.sidebar:
        st.markdown(
            f'<div class="sb-brand">{_LOGO_SVG}'
            '<div><div class="sb-brand-title">Football Tracking</div>'
            '<div class="sb-brand-sub">Player &amp; Ball Analytics</div></div>'
            '</div>'
            '<div class="sb-section-label">Menu</div>',
            unsafe_allow_html=True,
        )

        if st.button(
            "Dashboard",
            icon=":material/dashboard:",
            type="primary" if is_dash else "secondary",
            use_container_width=True,
        ):
            st.session_state.nav_page = "Dashboard"
            st.session_state.workflow_view = None
            st.rerun()

        if st.button(
            "Riwayat",
            icon=":material/history:",
            type="primary" if is_hist else "secondary",
            use_container_width=True,
        ):
            st.session_state.nav_page = "Riwayat"
            st.session_state.workflow_view = None
            st.rerun()

        st.divider()
        st.markdown(
            f'<div class="sb-pill">'
            f'<span class="sb-dot" style="background:{dot_color};"></span>'
            f'<span>{pill_label}</span>'
            '</div>'
            '<div class="sb-version">v1.0</div>',
            unsafe_allow_html=True,
        )


def show_dashboard():
    st.title("Football Player Tracking")
    st.write(
        "Unggah video pertandingan sepak bola untuk mendeteksi dan melacak "
        "posisi pemain, tim, bola, serta visualisasi radar."
    )

    uploaded_file = st.file_uploader(
        "Unggah video pertandingan (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov"],
        accept_multiple_files=False,
    )

    is_valid = False
    if uploaded_file is not None:
        is_valid, error_msg = validate_video(uploaded_file)
        size_mb = getattr(uploaded_file, "size", 0) / (1024 * 1024)

        if not is_valid:
            st.error(error_msg)
        else:
            st.success(f"Video siap dianalisis: **{uploaded_file.name}** ({size_mb:.1f} MB)")
            try:
                st.video(uploaded_file)
            except Exception:
                st.info("Preview video tidak didukung oleh browser untuk format ini.")

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        start_clicked = st.button(
            "Mulai Analisis",
            type="primary",
            disabled=not is_valid,
            use_container_width=True,
        )

    if start_clicked and uploaded_file is not None:
        with st.spinner("Mengunggah video dan memulai analisis..."):
            try:
                resp = submit_video(uploaded_file)
                analysis_id = resp.get("analysis_id")
                if not analysis_id:
                    st.error("Backend tidak mengembalikan `analysis_id`.")
                    return
                st.session_state.analysis_id = analysis_id
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.workflow_view = "processing"
                st.rerun()
            except BackendError as e:
                st.error(str(e))


def show_processing():
    st.title("Status Pemrosesan")

    analysis_id = st.session_state.analysis_id
    if not analysis_id:
        st.error("ID analisis tidak ditemukan.")
        if st.button("Kembali ke Dashboard"):
            reset_to_dashboard()
            st.rerun()
        return

    if st.session_state.uploaded_filename:
        st.write(f"**File:** {st.session_state.uploaded_filename}")
    st.write(f"**ID Analisis:** `{analysis_id}`")

    col_back, col_refresh, _ = st.columns([1, 1, 3])
    with col_back:
        if st.button("Kembali ke Dashboard"):
            reset_to_dashboard()
            st.rerun()
    with col_refresh:
        refresh_now = st.button("Perbarui Status")

    try:
        status_data = get_status(analysis_id)
    except BackendError as e:
        st.error(str(e))
        return

    raw_status = str(status_data.get("status", "unknown")).lower()
    label = STATUS_LABELS.get(raw_status, raw_status.capitalize())
    progress = status_data.get("progress")
    error_message = status_data.get("error") or status_data.get("message")

    st.subheader(f"Status: {label}")

    if isinstance(progress, (int, float)):
        norm_progress = progress / 100.0 if progress > 1 else float(progress)
        norm_progress = max(0.0, min(1.0, norm_progress))
        st.progress(norm_progress, text=f"{int(norm_progress * 100)}%")

    if raw_status in ("uploading", "queued", "processing"):
        st.info("Analisis sedang berjalan di backend. Halaman akan diperbarui otomatis.")
        if not refresh_now:
            time.sleep(5)
            st.rerun()
    elif raw_status == "completed":
        st.success("Analisis selesai! Silakan buka hasil analisis.")
        if st.button("Lihat Hasil", type="primary"):
            st.session_state.workflow_view = "results"
            st.rerun()
    elif raw_status == "failed":
        st.error(error_message or "Analisis gagal diproses oleh backend.")
    else:
        st.warning(f"Status tidak dikenali dari backend: `{raw_status}`")


def show_results():
    st.title("Hasil Analisis")

    analysis_id = st.session_state.analysis_id
    if not analysis_id:
        st.error("ID analisis tidak ditemukan.")
        if st.button("Kembali ke Dashboard"):
            reset_to_dashboard()
            st.rerun()
        return

    if st.button("Kembali ke Dashboard"):
        reset_to_dashboard()
        st.rerun()

    st.write(f"**ID Analisis:** `{analysis_id}`")

    try:
        result = get_results(analysis_id)
    except BackendError as e:
        st.error(str(e))
        return

    video_url = result.get("video_url")
    radar_url = result.get("radar_url")
    summary = result.get("summary") or {}
    status = result.get("status")

    col_video, col_radar = st.columns(2, gap="medium")

    with col_video:
        st.subheader("Video Pertandingan Teranotasi")
        if video_url:
            st.video(video_url)
            try:
                video_bytes = download_file(video_url)
                st.download_button(
                    "Unduh Video Teranotasi",
                    data=video_bytes,
                    file_name=f"{analysis_id}_annotated.mp4",
                    mime="video/mp4",
                )
            except BackendError as e:
                st.warning(str(e))
        else:
            st.info("Video teranotasi tidak tersedia dari backend.")

    with col_radar:
        st.subheader("Visualisasi Radar")
        if radar_url:
            if str(radar_url).lower().endswith((".png", ".jpg", ".jpeg")):
                st.image(radar_url, use_container_width=True)
                file_ext = "png"
                mime = "image/png"
            else:
                st.video(radar_url)
                file_ext = "mp4"
                mime = "video/mp4"
            try:
                radar_bytes = download_file(radar_url)
                st.download_button(
                    "Unduh Output Radar",
                    data=radar_bytes,
                    file_name=f"{analysis_id}_radar.{file_ext}",
                    mime=mime,
                )
            except BackendError as e:
                st.warning(str(e))
        else:
            st.info("Output radar tidak tersedia dari backend.")

    if video_url and radar_url:
        st.caption(
            "Catatan: Video dan radar ditampilkan berdampingan; sinkronisasi frame otomatis "
            "hanya tersedia jika backend menyertakan metadata waktu."
        )

    st.divider()
    st.subheader("Ringkasan Pertandingan")

    metrics = []
    if "total_players" in summary:
        metrics.append(("Total Pemain Terlacak", summary["total_players"]))
    if "total_teams" in summary:
        metrics.append(("Jumlah Tim", summary["total_teams"]))
    if "ball_detected" in summary:
        metrics.append(("Bola Terdeteksi", "Ya" if summary["ball_detected"] else "Tidak"))
    if "duration" in summary:
        metrics.append(("Durasi (detik)", summary["duration"]))
    if status:
        metrics.append(("Status", STATUS_LABELS.get(str(status).lower(), str(status))))

    if metrics:
        cols = st.columns(len(metrics))
        for col, (label, value) in zip(cols, metrics):
            col.metric(label, value)
    else:
        st.info("Ringkasan statistik belum tersedia dari backend.")


def show_history():
    st.title("Riwayat Analisis")

    try:
        data = get_history()
        analyses = data.get("analyses", [])
    except BackendError as e:
        st.warning(str(e))
        return

    if not analyses:
        st.info("Belum ada riwayat analisis. Unggah video di Dashboard untuk memulai.")
        return

    for item in analyses:
        aid = item.get("analysis_id", "-")
        filename = item.get("filename", "-")
        date = item.get("date", "-")
        raw_status = str(item.get("status", "-")).lower()
        label = STATUS_LABELS.get(raw_status, raw_status.capitalize())

        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([2, 3, 2, 2, 1.5], vertical_alignment="center")
            c1.write(f"`{aid}`")
            c2.write(filename)
            c3.write(date)
            c4.write(label)
            if raw_status == "completed" and aid != "-":
                if c5.button("Lihat Hasil", key=f"hist_{aid}", use_container_width=True):
                    st.session_state.analysis_id = aid
                    st.session_state.uploaded_filename = filename
                    st.session_state.workflow_view = "results"
                    st.rerun()
            elif raw_status in ("uploading", "queued", "processing") and aid != "-":
                if c5.button("Pantau", key=f"proc_{aid}", use_container_width=True):
                    st.session_state.analysis_id = aid
                    st.session_state.uploaded_filename = filename
                    st.session_state.workflow_view = "processing"
                    st.rerun()
            else:
                c5.write("-")


init_state()
render_sidebar()

if st.session_state.workflow_view == "processing":
    show_processing()
elif st.session_state.workflow_view == "results":
    show_results()
elif st.session_state.nav_page == "Riwayat":
    show_history()
else:
    show_dashboard()
