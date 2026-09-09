import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from transformer_maskgit import CTViT
from ct_clip import CTCLIP
from sklearn.metrics import roc_auc_score, f1_score

from dataset_fixed import CTReportDatasetFixed  # our fixed preprocessing from step 2

# --- paths, adjust if yours differ ---
CHECKPOINT_PATH = "checkpoints/models/CT-CLIP-Related/CT-CLIP_v2.pt"
DATA_FOLDER = "data_volumes/dataset/valid_fixed"
# REPORTS_CSV = "raw_csvs/dataset/radiology_text_reports/validation_reports.csv"
# META_CSV = "raw_csvs/dataset/metadata/validation_metadata.csv"
REPORTS_CSV = "validation_reports.csv"
META_CSV = "validation_metadata.csv"
LABELS_CSV = "valid_labels.csv"  # from the download-helper repo, filtered to your 100-volume subset
RESULTS_DIR = "results/ctclip_zeroshot"

pathologies = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly',
               'Pericardial effusion', 'Coronary artery wall calcification', 'Hiatal hernia',
               'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule', 'Lung opacity',
               'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern',
               'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
               'Interlobular septal thickening']

device = "cuda" if torch.cuda.is_available() else "cpu"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# --- build model, same architecture args as run_zero_shot.py ---
tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
text_encoder.resize_token_embeddings(len(tokenizer))

image_encoder = CTViT(
    dim=512, codebook_size=8192, image_size=480, patch_size=20,
    temporal_patch_size=10, spatial_depth=4, temporal_depth=4,
    dim_head=32, heads=8
)

clip = CTCLIP(
    image_encoder=image_encoder, text_encoder=text_encoder,
    dim_image=294912, dim_text=768, dim_latent=512,
    extra_latent_projection=False, use_mlm=False,
    downsample_image_embeds=False, use_all_token_embeds=False
)
# clip.load(CHECKPOINT_PATH)
state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
missing, unexpected = clip.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

clip.to(device)
clip.eval()

# --- dataset ---
ds = CTReportDatasetFixed(data_folder=DATA_FOLDER, reports_file=REPORTS_CSV, meta_file=META_CSV, labels=LABELS_CSV)
# ds.samples = ds.samples[:2]  # TEMP: sanity check before full 100-volume run
dl = DataLoader(ds, num_workers=4, batch_size=1, shuffle=False)

predicted_all = []
real_all = []
accession_names = []

with torch.no_grad():
    for volume_tensor, text, onehot_labels, acc_name in dl:
        predicted_labels = []
        for pathology in pathologies:
            prompts = [f"{pathology} is present.", f"{pathology} is not present."]
            text_tokens = tokenizer(prompts, return_tensors="pt", padding="max_length",
                                     truncation=True, max_length=512).to(device)
            output = clip(text_tokens, volume_tensor.to(device), device=device)
            probs = torch.nn.functional.softmax(output, dim=0).cpu().numpy()
            predicted_labels.append(probs[0])  # "present" probability

        predicted_all.append(predicted_labels)
        real_all.append(onehot_labels.numpy()[0])
        accession_names.append(acc_name[0])
        print(f"done: {acc_name[0]}")

predicted_all = np.array(predicted_all)
real_all = np.array(real_all)

np.savez(f"{RESULTS_DIR}/predicted.npz", data=predicted_all)
np.savez(f"{RESULTS_DIR}/labels.npz", data=real_all)
with open(f"{RESULTS_DIR}/accessions.txt", "w") as f:
    f.write("\n".join(accession_names))

# --- metrics: per-finding AUC-ROC + F1 (threshold 0.5) ---
rows = []
for i, path_name in enumerate(pathologies):
    y_true = real_all[:, i]
    y_pred = predicted_all[:, i]
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float("nan")  # happens if a finding has 0 positive cases in the subset
    f1 = f1_score(y_true, (y_pred >= 0.5).astype(int), zero_division=0)
    rows.append({"finding": path_name, "auc": auc, "f1": f1, "n_positive": int(y_true.sum())})

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(f"{RESULTS_DIR}/ctclip_metrics.csv", index=False)
print(metrics_df)



#                                finding  auc   f1  n_positive
# 0                     Medical material  NaN  0.0           0
# 1          Arterial wall calcification  NaN  0.0           0
# 2                         Cardiomegaly  NaN  0.0           0
# 3                 Pericardial effusion  NaN  0.0           0
# 4   Coronary artery wall calcification  NaN  0.0           0
# 5                        Hiatal hernia  NaN  0.0           2
# 6                      Lymphadenopathy  NaN  0.0           2
# 7                            Emphysema  NaN  1.0           2
# 8                          Atelectasis  NaN  0.0           0
# 9                          Lung nodule  NaN  1.0           2
# 10                        Lung opacity  NaN  0.0           2
# 11          Pulmonary fibrotic sequela  NaN  0.0           2
# 12                    Pleural effusion  NaN  0.0           0
# 13          Mosaic attenuation pattern  NaN  0.0           0
# 14            Peribronchial thickening  NaN  0.0           0
# 15                       Consolidation  NaN  0.0           2
# 16                      Bronchiectasis  NaN  0.0           0
# 17      Interlobular septal thickening  NaN  0.0           0
