# get_subset.py
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

hf_token = os.environ["HF_TOKEN"]
repo_id = "ibrahimhamamci/CT-RATE"
N_VOLUMES = 100  # keep this small for a quick project

api = HfApi()
print("Listing files from HF (may take a moment)...")
all_files = api.list_repo_files(repo_id, repo_type="dataset", revision="main", token=hf_token)

# Filter to just the valid_fixed volumes
valid_volumes = [f for f in all_files if f.startswith("dataset/valid_fixed/") and f.endswith(".nii.gz")]
print(f"Found {len(valid_volumes)} total validation volumes")

subset = valid_volumes[:N_VOLUMES]

for f in subset:
    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f,
        local_dir="./data_volumes",
        token=hf_token,
    )

print(f"Downloaded {len(subset)} volumes to ./data_volumes")