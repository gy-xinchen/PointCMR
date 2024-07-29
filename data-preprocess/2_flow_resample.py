import os
import glob
import numpy as np
import SimpleITK as sitk

# 设置文件夹路径和输出路径
file_dir = r"D:\CMR-res\4Dsegment-master\muti-center-dataset\repair_nii_data"

def resample_image_by_size(ori_image, target_size, mode):
    """

    :param ori_imgae: itk 读取的图像
    :param target_size: 列表形式保存的目标尺寸
    :param mode: "sitk.sitkLinear" OR "sitk.sitkNearestNeighbor"
    :return:
    """

    ori_size = np.array(ori_image.GetSize())
    ori_spacing = np.array(ori_image.GetSpacing())
    target_spacing = ori_spacing * ori_size / np.array(target_size)

    resampler = sitk.ResampleImageFilter()  # 初始化滤波器
    resampler.SetReferenceImage(ori_image)  # 传入需要重新采样的目标图像
    resampler.SetOutputDirection(ori_image.GetDirection())
    resampler.SetOutputOrigin(ori_image.GetOrigin())
    resampler.SetInterpolator(mode)  # 设置插值方法
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing([float(s) for s in target_spacing])

    itk_img_resampled = resampler.Execute(ori_image)  # 得到重新采样后的图像
    return itk_img_resampled

# 定义递归函数来遍历所有子文件夹并处理其中的NIfTI文件
def process_folder(folder_path):
    for file_name in glob.glob(os.path.join(folder_path, "*.nii*")):
        # 从文件名中提取文件名并读取图像
        file_split = os.path.split(file_name)[-1]
        image = sitk.ReadImage(file_name)
        img_array = sitk.GetArrayFromImage(image)
        channel, W, H = img_array.shape
        save_img = resample_image_by_size(ori_image=image, target_size=(512, 512, channel), mode=sitk.sitkLinear)


        # 将结果保存为NIfTI格式的文件
        output_dir = os.path.join(file_dir, os.path.relpath(folder_path, file_dir))  # 构造输出文件夹路径
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # 如果输出文件夹不存在，则创建该文件夹
        sitk.WriteImage(save_img, os.path.join(output_dir, file_split))
        print("{} done".format(file_name))

    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            process_folder(subfolder_path)


# 调用递归函数来处理所有子文件夹中的NIfTI文件
process_folder(file_dir)
