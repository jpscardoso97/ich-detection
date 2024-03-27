# This code loads the CT slices (grayscale images) of the brain-window for each 
# subject in ct_scans folder then saves them to one folder (data\image).
# Their segmentation from the masks folder is saved to another folder (data\label).

import os
from pathlib import Path
import numpy as np
import pandas as pd
from skimage.transform import resize
from imageio.v3 import imwrite
import nibabel as nib

def window_ct(ct_scan, w_level=40, w_width=120):
    w_min = w_level - w_width / 2
    w_max = w_level + w_width / 2
    num_slices=ct_scan.shape[2]
    for s in range(num_slices):
        slice_s = ct_scan[:,:,s]
        slice_s = (slice_s - w_min)*(255/(w_max-w_min))
        slice_s[slice_s < 0] = 0
        slice_s[slice_s > 255] = 255
        ct_scan[:,:,s] = slice_s

    return ct_scan

numSubj = 82
new_size = (512, 512)
window_specs = [40,120] # Brain window
rawDataDir = Path(os.getcwd()) # Set to directory where ct_ich data is located
saveDataDir = Path(os.getcwd()) # Set to where processed data is to be saved

# Reading labels
hemorrhage_diagnosis_df = pd.read_csv(Path(rawDataDir, 'hemorrhage_diagnosis_raw_ct.csv'))
patientNumber = hemorrhage_diagnosis_df["PatientNumber"]

# Paths for saving processed images
train_path = Path(saveDataDir, 'data')
image_path = train_path / 'image'
label_path = train_path / 'label'
train_path.mkdir(exist_ok=True)
image_path.mkdir(exist_ok=True)
label_path.mkdir(exist_ok=True)

counterI = 0
groups = []
for sNo in range(0+49, numSubj+49):
    if sNo > 58 and sNo < 66: # No raw data were available for these subjects
        continue

    print("Processing sNo", sNo)
    # Loading the CT scan
    ct_dir_subj = Path(rawDataDir, 'ct_scans', f"{sNo:0=3d}.nii")
    ct_scan_nifti = nib.load(ct_dir_subj)
    ct_scan = ct_scan_nifti.get_fdata()
    ct_scan = window_ct(ct_scan, window_specs[0], window_specs[1])

    # Loading the masks
    masks_dir_subj = Path(rawDataDir, 'masks', f"{sNo:0=3d}.nii")
    masks_nifti = nib.load(masks_dir_subj)
    masks = masks_nifti.get_fdata().round() # Rounding is needed for get_fdata in masks
    idx = patientNumber == sNo
    sliceNos = hemorrhage_diagnosis_df.loc[idx, "SliceNumber"].values
    NoHemorrhage = hemorrhage_diagnosis_df.loc[idx, "No_Hemorrhage"].values
    if sliceNos.size != ct_scan.shape[2]:
        print('\tWarning: the number of annotated slices does not equal the number of slices in NIFTI file!')

    for sliceI in range(sliceNos.size):
        # Saving the a given CT slice
        x = resize(ct_scan[:,:,sliceI], new_size)
        imwrite(image_path / f'{counterI}.png', x.astype('uint8'))

        # Saving the segmentation for a given slice
        x = resize(masks[:,:,sliceI], new_size)
        imwrite(label_path / f'{counterI}.png', x.astype('uint8'))
        counterI += 1
    groups.extend([sNo]*sliceNos.size)

# Create CSV with grouping data
pd.DataFrame({ "group": groups }).to_csv(Path(saveDataDir, 'groups.csv'), index_label="fileId")