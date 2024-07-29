import os
import SimpleITK as sitk
import numpy as np

def get_numeric_part(filename):
    # get file name
    name, ext = os.path.splitext(filename)
    parts = name.split('_')
    # sorting
    return int(parts[0]), int(parts[1])

def dicoms_to_nifti(InputFolder, OutputFile):
    # get dicom files
    DcmFiles = os.listdir(InputFolder)
    if "_" in DcmFiles[0]:
        SortedDcmFiles = sorted(DcmFiles, key=get_numeric_part) # sone sample such as 90_89.dcm not 90.dcm
        DcmFiles = SortedDcmFiles
        print(InputFolder)

    DcmList = []
    for i in range(len(DcmFiles)):
        # read DICOM
        dicom_reader = sitk.ReadImage(os.path.join(InputFolder, DcmFiles[i]))

        # get array
        ImageArray = sitk.GetArrayFromImage(dicom_reader)
        DcmList.append(ImageArray[0,:,:])
    DcmList = np.array(DcmList)
    DcmList = sitk.GetImageFromArray(DcmList)
    DcmList.SetSpacing(dicom_reader.GetSpacing())
    DcmList.SetOrigin(dicom_reader.GetOrigin())
    DcmList.SetDirection(dicom_reader.GetDirection())

    # save NIfTI
    sitk.WriteImage(DcmList, OutputFile)

    print(f"Converted DICOMs to 3D NIfTI: {OutputFile}")


input_path = r"D:\CMR-res\4Dsegment-master\muti-center-dataset\all_mPAP_dcm_data"
output_path = r"D:\CMR-res\4Dsegment-master\muti-center-dataset\all_mPAP_nii_data"
patient_path = os.listdir(input_path)

for patient_id in range(len(patient_path)):
    patient_layers_path = os.path.join(input_path, patient_path[patient_id])
    patient_layers = os.listdir(patient_layers_path)
    for layers in range(len(patient_layers)):
        input_folder_path = os.path.join(input_path, patient_path[patient_id], patient_layers[layers])
        output_nifti_path = os.path.join(output_path, patient_path[patient_id])
        output_nifti_files = os.path.join(output_nifti_path, patient_layers[layers]+".nii.gz")

        if not os.path.exists(output_nifti_path):
            os.makedirs(output_nifti_path)
        dicoms_to_nifti(input_folder_path, output_nifti_files)
