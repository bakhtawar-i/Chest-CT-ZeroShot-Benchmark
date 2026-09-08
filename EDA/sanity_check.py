# sanity_check.py
import nibabel as nib
import numpy as np

vol_path = "data_volumes/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"  # adjust to an actual downloaded file
img = nib.load(vol_path)
data = img.get_fdata()

print(f"Raw min/max: {data.min()}, {data.max()}")
print(f"Raw mean: {data.mean():.1f}")

# Raw min/max: -1024.0, 3071.0
# Raw mean: -527.6

import nibabel as nib

vol_path = "data_volumes/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
img = nib.load(vol_path)

print("Header scl_slope:", img.header['scl_slope'])
print("Header scl_inter:", img.header['scl_inter'])

# Also compare against raw, unscaled array data
raw = img.dataobj.get_unscaled()
print("Unscaled min/max:", raw.min(), raw.max())