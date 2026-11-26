import requests
from config import API_BASE_URL, API_TIMEOUT


class BackendError(Exception):
    pass


def _get_base_url():
    if not API_BASE_URL:
        raise BackendError("Backend belum dikonfigurasi. Atur `API_BASE_URL` di environment atau `.streamlit/secrets.toml`.")
    return API_BASE_URL


def _request(method, endpoint, files=None, json=None):
    url = f"{_get_base_url()}{endpoint}"
    try:
        resp = requests.request(method, url, files=files, json=json, timeout=API_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise BackendError("Backend tidak dapat dihubungi. Pastikan server backend sedang berjalan.")
    except requests.exceptions.Timeout:
        raise BackendError("Permintaan ke backend timeout.")
    except requests.exceptions.HTTPError as e:
        detail = e.response.text[:200] if e.response is not None else ""
        code = e.response.status_code if e.response is not None else "HTTP"
        raise BackendError(f"Error dari backend ({code}): {detail}")
    except ValueError:
        raise BackendError("Respons dari backend bukan format JSON yang valid.")


def submit_video(file_obj):
    file_obj.seek(0)
    files = {"file": (file_obj.name, file_obj.read(), file_obj.type or "video/mp4")}
    return _request("POST", "/api/analyses", files=files)


def get_status(analysis_id):
    return _request("GET", f"/api/analyses/{analysis_id}/status")


def get_results(analysis_id):
    return _request("GET", f"/api/analyses/{analysis_id}/results")


def get_history():
    return _request("GET", "/api/analyses")


def download_file(url):
    try:
        resp = requests.get(url, timeout=API_TIMEOUT)
        resp.raise_for_status()
        return resp.content
    except requests.exceptions.RequestException as e:
        raise BackendError(f"Gagal mengunduh file: {e}")
