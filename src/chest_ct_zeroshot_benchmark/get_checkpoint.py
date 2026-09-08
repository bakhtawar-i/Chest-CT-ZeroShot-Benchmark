import os
from huggingface_hub import hf_hub_download

hf_token = os.environ["HF_TOKEN"]
repo_id = "ibrahimhamamci/CT-RATE"

hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename="models/CT-CLIP-Related/CT-CLIP_v2.pt",
    local_dir="./checkpoints",
    token=hf_token,
)
print("done")