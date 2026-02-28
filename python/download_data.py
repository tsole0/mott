# python/download_data.py
import os
from huggingface_hub import hf_hub_download

HF_REPO = "bamille/mott-materials"
OUTPUT  = "data/raw_materials.json"

os.makedirs("data", exist_ok=True)

if os.path.exists(OUTPUT):
    print("Data already present, skipping.")
else:
    print("Downloading raw data from HuggingFace (~1.2GB, grab a coffee)...")
    hf_hub_download(
        repo_id=HF_REPO,
        filename="raw_materials.json",
        repo_type="dataset",
        local_dir="data/",
    )
    print(f"Saved to {OUTPUT}")