"""
download_weights.py — Download NVlabs SegFormer-B2 ADE20K pre-trained weights.

Run once:
    python segformer_model/download_weights.py

The script tries gdown (Google Drive) automatically.
If that fails, it prints the manual download URL.

NVlabs SegFormer-B2 ADE20K (160k iterations, 512×512):
    Paper:   https://arxiv.org/abs/2105.15203
    Repo:    https://github.com/NVlabs/SegFormer
    Weights: available from NVlabs model zoo (Google Drive)
"""

import os, sys, ssl, warnings
from pathlib import Path

# Patch SSL globally — corporate IT tools install custom root certs that
# break SSL verification even on home networks.
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore")
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# Patch requests.Session so gdown (which uses requests internally) also
# skips certificate verification.
try:
    import requests
    from requests import Session as _Session
    _orig_request = _Session.request
    def _no_ssl_request(self, *a, **kw):
        kw.setdefault("verify", False)
        return _orig_request(self, *a, **kw)
    _Session.request = _no_ssl_request
except ImportError:
    pass

DEST_DIR   = Path(__file__).parent
CKPT_FILE  = DEST_DIR / "segformer.b2.512x512.ade.160k.pth"

# NVlabs SegFormer-B2 ADE20K Google Drive file ID.
# Source: https://github.com/NVlabs/SegFormer (Table in README)
GDRIVE_ID  = "1ILRqSCMB7zBgK3JlSN_hNZoGKWYDi4IK"

# Direct fallback URL (try if gdown fails)
DIRECT_URL = (
    "https://drive.google.com/uc"
    f"?export=download&id={GDRIVE_ID}"
)

MANUAL_URL = (
    "https://github.com/NVlabs/SegFormer#training"
    "\n  OR go to the NVlabs repo README and click the ADE20K SegFormer-B2 link."
    "\n  Save the file as: " + str(CKPT_FILE)
)


def download():
    if CKPT_FILE.exists():
        size_mb = CKPT_FILE.stat().st_size // (1024 * 1024)
        print(f"[weights] Already downloaded: {CKPT_FILE}  ({size_mb} MB)")
        return True

    print(f"[weights] Downloading NVlabs SegFormer-B2 ADE20K weights ...")
    print(f"[weights] Destination: {CKPT_FILE}")

    # Method 1: gdown (Google Drive)
    try:
        import gdown
        print("[weights] Using gdown ...")
        gdown.download(id=GDRIVE_ID, output=str(CKPT_FILE), quiet=False)
        if CKPT_FILE.exists() and CKPT_FILE.stat().st_size > 1_000_000:
            size_mb = CKPT_FILE.stat().st_size // (1024 * 1024)
            print(f"[weights] Downloaded: {CKPT_FILE}  ({size_mb} MB)")
            return True
    except Exception as e:
        print(f"[weights] gdown failed: {e}")

    # Method 2: requests (direct URL)
    try:
        import requests, urllib3
        urllib3.disable_warnings()
        print("[weights] Trying direct requests download ...")
        session = requests.Session()
        session.verify = False
        r = session.get(DIRECT_URL, stream=True, timeout=300)
        r.raise_for_status()
        with open(CKPT_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=2*1024*1024):
                f.write(chunk)
        if CKPT_FILE.exists() and CKPT_FILE.stat().st_size > 1_000_000:
            print(f"[weights] Downloaded via requests: {CKPT_FILE}")
            return True
    except Exception as e:
        print(f"[weights] requests failed: {e}")

    # Manual fallback
    print("\n[weights] Automatic download failed.")
    print("[weights] Please download manually:")
    print(f"  URL:  https://drive.google.com/file/d/{GDRIVE_ID}/view")
    print(f"  Save to: {CKPT_FILE}")
    return False


if __name__ == "__main__":
    success = download()
    sys.exit(0 if success else 1)
