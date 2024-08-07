import os
import SimpleITK as sitk
import re

"""
Some sample have wrong dcm metadata，we use this code to repair this problem
"""

# sorted function
def numerical_sort(value):
    parts = re.findall(r'\d+|\D+', value)
    return [int(part) if part.isdigit() else part for part in parts]


def dicoms_to_nifti(input_folder, output_file):


    # read DICOM
    dicom_reader = sitk.ImageSeriesReader()
    dcmfiles = dicom_reader.GetGDCMSeriesFileNames(input_folder)
    dicom_reader.SetFileNames(dcmfiles)

    # sapcing
    dicom_image = dicom_reader.Execute()

    output_image = sitk.GetImageFromArray(sitk.GetArrayFromImage(dicom_image))
    output_image.SetSpacing(dicom_image.GetSpacing())
    output_image.SetOrigin(dicom_image.GetOrigin())
    output_image.SetDirection(dicom_image.GetDirection())

    sitk.WriteImage(output_image, output_file)

    print(f"Converted DICOMs to 3D NIfTI: {output_file}")


input_path = r"D:\CMR-res\4Dsegment-master\muti-center-dataset\repair_dcm_data"
output_path = r"D:\CMR-res\4Dsegment-master\muti-center-dataset\repair_nii_data"
patient_path = os.listdir(input_path)

for patient_id in range(len(patient_path)):
    patient_layers_path = os.path.join(input_path, patient_path[patient_id])
    patient_layers = os.listdir(patient_layers_path)
    patient_layers = sorted(patient_layers, key=numerical_sort)
    for layers in range(len(patient_layers)):
        input_folder_path = os.path.join(input_path, patient_path[patient_id], patient_layers[layers])
        output_nifti_path = os.path.join(output_path, patient_path[patient_id])
        output_nifti_files = os.path.join(output_path, patient_path[patient_id], f"SE{layers}.nii.gz")
        if not os.path.exists(output_nifti_path):
            os.makedirs(output_nifti_path)
        dicoms_to_nifti(input_folder_path, output_nifti_files)
