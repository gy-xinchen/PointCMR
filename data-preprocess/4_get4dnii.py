import os
import nibabel as nib
import numpy as np

"""
    XX, [20, 512, 512] --> [XX, 20, 512, 512]
"""

def create_4d_nii(folder_path, output_path):
    files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".nii.gz")]
    files.sort(key=lambda x: int(x.split('E')[-1].split('.')[0]))

    # 加载第一个文件以获取头信息和图像尺寸
    first_img = nib.load(os.path.join(folder_path, files[0]))
    data = first_img.get_fdata()
    shape = data.shape
    header = first_img.header

    new_slice_thickness = 8.0
    header["pixdim"][3] = new_slice_thickness

    # 创建一个空的4D数组
    n_volumes = len(files)
    combined_data = np.zeros(shape + (n_volumes,))


    # 将每个3D图像加载到4D数组中
    for i, filename in enumerate(files):
        img = nib.load(os.path.join(folder_path, filename))
        combined_data[..., i] = img.get_fdata()

    combined_data = np.swapaxes(combined_data, 2, 3)
    # 创建新的4D NIfTI图像
    combined_img = nib.Nifti1Image(combined_data, affine=first_img.affine, header=header)

    # 保存为.nii.gz文件
    nib.save(combined_img, output_path)

DirPath = r"D:\CMR-res\4Dsegment-master\muti-center-dataset\repair_nii_data"
OutPath = r"D:\CMR-res\4Dsegment-master\muti-center-dataset\repair_4dnii_data"
Files = os.listdir(DirPath)
for file in range(len(Files)):
    InputFolder = os.path.join(DirPath, Files[file])
    OutputFolder = os.path.join(OutPath, Files[file]+".nii.gz")
    create_4d_nii(InputFolder, OutputFolder)
    print(f"{Files[file]} done!")
