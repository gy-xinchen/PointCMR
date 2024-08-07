import SimpleITK as sitk
import os
import numpy as np

"""
    for resample image size, [25/30, 512,512] --> [20, 512, 512]
"""


folder_path = r"D:\CMR-res\4Dsegment-master\muti-center-dataset\repair_nii_data"
target_shape = [20, 512, 512]

def interpolate_sitk(image, new_shape):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkBSpline)
    resample.SetOutputSpacing(image.GetSpacing())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetSize(new_shape)
    return resample.Execute(image)

def process_folder(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        if os.path.isdir(item_path):
            # if it is file, for subfile
            process_folder(item_path)
        elif item.endswith('.nii.gz'):
            # if is nii，read data and resample
            img = sitk.ReadImage(item_path)
            # get img shape
            size = img.GetSize()
            data = sitk.GetArrayFromImage(img)
            # resample
            new_data = np.zeros([target_shape[0], size[0], size[1]], dtype=data.dtype)
            for i in range(target_shape[0]):
                t = i / (target_shape[0]-1) * (size[2]-1)
                idx = int(np.floor(t))
                t = t - idx
                if idx == size[2]-1:
                    new_data[i] = data[idx]
                else:
                    new_data[i] = (1-t) * data[idx] + t * data[idx+1]
            interp_img = sitk.GetImageFromArray(new_data)
            interp_img.SetSpacing(img.GetSpacing())
            interp_img.SetDirection(img.GetDirection())
            interp_img.SetOrigin(img.GetOrigin())

            # save path
            output_dir = os.path.join(folder_path, os.path.relpath(item_path, folder_path))  # 构造输出文件夹路径
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            sitk.WriteImage(interp_img, output_dir)
            print("{} done".format(item_path))

process_folder(folder_path)
