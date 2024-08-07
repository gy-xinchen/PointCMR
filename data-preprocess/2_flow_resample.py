import os
import glob
import numpy as np
import SimpleITK as sitk

"""
    for resample image size, [XX, 384,412] --> [XX, 512,512]
"""

file_dir = r"D:\CMR-res\4Dsegment-master\muti-center-dataset\repair_nii_data"

def resample_image_by_size(ori_image, target_size, mode):
    """

    :param ori_imgae: itk 
    :param target_size: 
    :param mode: "sitk.sitkLinear" OR "sitk.sitkNearestNeighbor"
    :return:
    """

    ori_size = np.array(ori_image.GetSize())
    ori_spacing = np.array(ori_image.GetSpacing())
    target_spacing = ori_spacing * ori_size / np.array(target_size)

    resampler = sitk.ResampleImageFilter()  
    resampler.SetReferenceImage(ori_image)  
    resampler.SetOutputDirection(ori_image.GetDirection())
    resampler.SetOutputOrigin(ori_image.GetOrigin())
    resampler.SetInterpolator(mode)  
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing([float(s) for s in target_spacing])

    itk_img_resampled = resampler.Execute(ori_image)  
    return itk_img_resampled


def process_folder(folder_path):
    for file_name in glob.glob(os.path.join(folder_path, "*.nii*")):
        
        file_split = os.path.split(file_name)[-1]
        image = sitk.ReadImage(file_name)
        img_array = sitk.GetArrayFromImage(image)
        channel, W, H = img_array.shape
        save_img = resample_image_by_size(ori_image=image, target_size=(512, 512, channel), mode=sitk.sitkLinear)


        
        output_dir = os.path.join(file_dir, os.path.relpath(folder_path, file_dir))  
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  
        sitk.WriteImage(save_img, os.path.join(output_dir, file_split))
        print("{} done".format(file_name))

    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            process_folder(subfolder_path)



process_folder(file_dir)
