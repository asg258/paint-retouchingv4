"""
download_segformer.py — One-time download of the SegFormer model weights.

Run once to cache the model locally (bypasses enterprise SSL inspection):
    python download_segformer.py

After this succeeds, the normal pipeline will find the weights in the
HuggingFace cache and will NOT need internet access again.
"""

import ssl, os, warnings

# Disable all SSL certificate verification — required on networks that
# use SSL inspection proxies (common in enterprise environments).
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"]                = ""
os.environ["REQUESTS_CA_BUNDLE"]            = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
warnings.filterwarnings("ignore")

# Patch httpx (used by huggingface_hub >= 0.20) to skip SSL verification.
try:
    import httpx
    _orig_init = httpx.Client.__init__
    def _patched_init(self, *args, **kwargs):
        kwargs["verify"] = False
        _orig_init(self, *args, **kwargs)
    httpx.Client.__init__ = _patched_init

    _orig_async_init = httpx.AsyncClient.__init__
    def _patched_async_init(self, *args, **kwargs):
        kwargs["verify"] = False
        _orig_async_init(self, *args, **kwargs)
    httpx.AsyncClient.__init__ = _patched_async_init
    print("[download] httpx SSL verification disabled.")
except ImportError:
    print("[download] httpx not found — skipping httpx patch.")

# Now import and download.
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

MODEL_ID = "nvidia/segformer-b2-finetuned-ade-512-512"

print(f"[download] Downloading {MODEL_ID} ...")
print("[download] This is ~100 MB and only needs to happen once.")

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model     = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)

wall_label = model.config.id2label.get(0, "?")
print(f"[download] Download complete.")
print(f"[download] Class 0 = '{wall_label}'  (must be 'wall')")
print("[download] Model cached. You can now run the main pipeline.")
