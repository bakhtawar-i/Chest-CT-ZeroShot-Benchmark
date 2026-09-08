from fileinput import filename
import os
from huggingface_hub import hf_hub_download
import pandas as pd 

hf_token = os.environ["HF_TOKEN"]
print(f"Token starts with: {hf_token[:6]}...")  # sanity check, remove after confirming

repo_id = "ibrahimhamamci/CT-RATE"

hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    subfolder="dataset",
    filename="radiology_text_reports/validation_reports.csv",
    # filename="metadata/validation_metadata.csv",
    local_dir="./raw_csvs",
    token=hf_token,
)
print("done")


# -> in bash - head -2 raw_csvs/dataset/metadata/validation_metadata.csv

# VolumeName,Manufacturer,SeriesDescription,ManufacturerModelName,PatientSex,PatientAge,ReconstructionDiameter,DistanceSourceToDetector,DistanceSourceToPatient,GantryDetectorTilt,TableHeight,RotationDirection,ExposureTime,XRayTubeCurrent,Exposure,FilterType,GeneratorPower,FocalSpots,ConvolutionKernel,PatientPosition,RevolutionTime,SingleCollimationWidth,TotalCollimationWidth,TableSpeed,TableFeedPerRotation,SpiralPitchFactor,DataCollectionCenterPatient,ReconstructionTargetCenterPatient,ExposureModulationType,CTDIvol,ImagePositionPatient,ImageOrientationPatient,SliceLocation,SamplesPerPixel,PhotometricInterpretation,Rows,Columns,XYSpacing,RescaleIntercept,RescaleSlope,RescaleType,NumberofSlices,ZSpacing,StudyDate
# valid_1_a_1.nii.gz,Philips,HRCT,iCT 256,M,036Y,350,,,0,71,CW,622,186,116,YA,,,YA,HFS,,,,,,,,,Z DOM,7.830488795,"[-166, 9, 56.2998657]","[1, 0, 0, 0, 1, 0]",56.3,1,MONOCHROME2,1024,1024,"[0.341796875, 0.341796875]",-1024,1,,251,1.5,20210427
