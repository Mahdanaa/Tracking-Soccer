import streamlit as st
from service import (
    AnalysisError,
    download_file,
    get_engine_status,
    get_history,
    get_results,
    get_status,
    run_analysis,
    submit_video,
)
from validators import validate_video

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

    engine_ok, pill_label = get_engine_status()
    dot_color = "#22C55E" if engine_ok else "#FF6B57"

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
            except AnalysisError as e:
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

    try:
        status_data = get_status(analysis_id)
    except AnalysisError as e:
        st.error(str(e))
        return

    raw_status = str(status_data.get("status", "unknown")).lower()
    label = STATUS_LABELS.get(raw_status, raw_status.capitalize())
    progress = status_data.get("progress")
    error_message = status_data.get("error") or status_data.get("message")

    col_back, col_refresh, _ = st.columns([1, 1, 3])
    with col_back:
        if st.button("Kembali ke Dashboard"):
            reset_to_dashboard()
            st.rerun()
    with col_refresh:
        st.button("Perbarui Status", disabled=raw_status in ("queued", "processing"))

    st.subheader(f"Status: {label}")

    if raw_status in ("queued", "processing"):
        phase_text = st.empty()
        progress_bar = st.progress(0.0, text="Memulai analisis...")

        def on_progress(phase: str, done: int, total: int) -> None:
            phase_text.info(f"Tahap: **{phase}**")
            if phase == "Processing frames" and total:
                ratio = max(0.0, min(1.0, done / total))
                progress_bar.progress(ratio, text=f"{done} / {total} frame ({int(ratio * 100)}%)")

        try:
            run_analysis(analysis_id, progress_callback=on_progress)
            st.rerun()
        except AnalysisError as e:
            progress_bar.empty()
            st.error(str(e))
    elif raw_status == "completed":
        if isinstance(progress, (int, float)):
            norm_progress = progress / 100.0 if progress > 1 else float(progress)
            norm_progress = max(0.0, min(1.0, norm_progress))
            st.progress(norm_progress, text=f"{int(norm_progress * 100)}%")
        st.success("Analisis selesai! Silakan buka hasil analisis.")
        if st.button("Lihat Hasil", type="primary"):
            st.session_state.workflow_view = "results"
            st.rerun()
    elif raw_status == "failed":
        st.error(error_message or "Analisis gagal diproses.")
    else:
        st.warning(f"Status tidak dikenali: `{raw_status}`")


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
    except AnalysisError as e:
        st.error(str(e))
        return

    video_path = result.get("video_path")
    summary = result.get("summary") or {}
    status = result.get("status")

    st.subheader("Video Pertandingan Teranotasi & Minimap Radar 2D")
    if video_path:
        st.video(video_path)
        try:
            video_bytes = download_file(video_path)
            st.download_button(
                "Unduh Video Hasil",
                data=video_bytes,
                file_name=f"{analysis_id}_annotated.mp4",
                mime="video/mp4",
            )
        except AnalysisError as e:
            st.warning(str(e))
    else:
        st.info("Video teranotasi tidak tersedia.")

    st.divider()
    st.subheader("Ringkasan Pertandingan")

    metrics = []
    if "frames_processed" in summary:
        metrics.append(("Total Frame", summary["frames_processed"]))
    if "processing_fps" in summary:
        metrics.append(("Kecepatan (FPS)", summary["processing_fps"]))
    if "total_seconds" in summary:
        metrics.append(("Waktu Proses (detik)", summary["total_seconds"]))
    if "duration" in summary:
        metrics.append(("Durasi Video (detik)", summary["duration"]))
    if "device" in summary:
        metrics.append(("Perangkat", summary["device"]))
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
    except AnalysisError as e:
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
            elif raw_status in ("queued", "processing") and aid != "-":
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
