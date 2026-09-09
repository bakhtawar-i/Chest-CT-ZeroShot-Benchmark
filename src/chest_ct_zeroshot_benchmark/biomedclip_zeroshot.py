import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
from sklearn.metrics import roc_auc_score, f1_score

from dataset_fixed import CTReportDatasetFixed
from torch.utils.data import DataLoader

DATA_FOLDER = "data_volumes/dataset/valid_fixed"
REPORTS_CSV = "validation_reports.csv"
META_CSV = "validation_metadata.csv"
LABELS_CSV = "valid_labels.csv"
RESULTS_DIR = "results/biomedclip_zeroshot"
N_SLICES = 16  # evenly spaced axial slices per volume

pathologies = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly',
               'Pericardial effusion', 'Coronary artery wall calcification', 'Hiatal hernia',
               'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule', 'Lung opacity',
               'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern',
               'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
               'Interlobular septal thickening']

device = "cuda" if torch.cuda.is_available() else "cpu"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
model.to(device).eval()

# Same present/absent prompt pairs as CT-CLIP, for a fair comparison — not
# BiomedCLIP's own default template style, since we want identical prompts
# across both models.
def build_prompt_texts():
    all_prompts = []
    for p in pathologies:
        all_prompts.append(f"{p} is present.")
        all_prompts.append(f"{p} is not present.")
    return all_prompts

text_tokens = tokenizer(build_prompt_texts()).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

ds = CTReportDatasetFixed(data_folder=DATA_FOLDER, reports_file=REPORTS_CSV, meta_file=META_CSV, labels=LABELS_CSV)
dl = DataLoader(ds, num_workers=0, batch_size=1, shuffle=False)

def volume_to_slice_batch(volume_tensor):
    # volume_tensor: (1, 1, D, H, W) as produced by CTReportDatasetFixed
    vol = volume_tensor.squeeze(0).squeeze(0)  # (D, H, W), D=240, roughly [-1, 1] range
    D = vol.shape[0]
    idxs = np.linspace(0, D - 1, N_SLICES).astype(int)
    imgs = []
    for i in idxs:
        sl = vol[i].numpy()
        sl_255 = ((sl + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(sl_255).convert("RGB")
        imgs.append(preprocess(pil_img))
    return torch.stack(imgs)  # (N_SLICES, 3, 224, 224)

predicted_all = []
real_all = []
accession_names = []

with torch.no_grad():
    for volume_tensor, text, onehot_labels, acc_name in dl:
        slice_batch = volume_to_slice_batch(volume_tensor).to(device)
        image_features = model.encode_image(slice_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        volume_embedding = image_features.mean(dim=0, keepdim=True)  # mean-pool slices
        volume_embedding = volume_embedding / volume_embedding.norm(dim=-1, keepdim=True)

        logits = (model.logit_scale.exp() * volume_embedding @ text_features.T).squeeze(0)  # (36,)

        predicted_labels = []
        for i in range(len(pathologies)):
            pair_logits = logits[2*i:2*i+2]
            probs = torch.softmax(pair_logits, dim=0).cpu().numpy()
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

rows = []
for i, path_name in enumerate(pathologies):
    y_true = real_all[:, i]
    y_pred = predicted_all[:, i]
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float("nan")
    f1 = f1_score(y_true, (y_pred >= 0.5).astype(int), zero_division=0)
    rows.append({"finding": path_name, "auc": auc, "f1": f1, "n_positive": int(y_true.sum())})

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(f"{RESULTS_DIR}/biomedclip_metrics.csv", index=False)
print(metrics_df)


#                                finding       auc        f1  n_positive
# 0                     Medical material  0.874521  0.000000          12
# 1          Arterial wall calcification  0.648482  0.000000          31
# 2                         Cardiomegaly  0.723140  0.358974          22
# 3                 Pericardial effusion  0.710562  0.000000          18
# 4   Coronary artery wall calcification  0.555556  0.347826          24
# 5                        Hiatal hernia  0.213102  0.000000          16
# 6                      Lymphadenopathy  0.522388  0.000000          32
# 7                            Emphysema  0.498919  0.396040          25
# 8                          Atelectasis  0.607646  0.420000          28
# 9                          Lung nodule  0.276695  0.000000          40
# 10                        Lung opacity  0.555503  0.476923          31
# 11          Pulmonary fibrotic sequela  0.370531  0.148148          30
# 12                    Pleural effusion  0.936607  0.000000          35
# 13          Mosaic attenuation pattern  0.743789  0.153846           7
# 14            Peribronchial thickening  0.684211  0.000000          23
# 15                       Consolidation  0.495482  0.000000          16
# 16                      Bronchiectasis  0.657303  0.000000          10
# 17      Interlobular septal thickening  0.580460  0.000000          12