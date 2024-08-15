import csv
import os
import re
import pandas as pd
import numpy as np

dir_path = r"D:\CMR-res\PH点云预测项目代码\4D_segment\4D_motion_point_dataset20240808"
label_path = os.path.join(dir_path, "mPAP.csv")

output_path = r"D:\CMR-res\PH点云预测项目代码\4D_segment\3D_motion_classfication35\3D_motion_systole"
all_items = os.listdir(dir_path)

# 过滤出文件夹
folders = [item for item in all_items if os.path.isdir(os.path.join(dir_path, item))]


# 提取文件夹名称中的数字并排序
def extract_number(folder_name):
    match = re.search(r'\d+', folder_name)
    return int(match.group()) if match else float('inf')


sorted_folders = sorted(folders, key=extract_number)

df = pd.read_csv(label_path)
label_list = df.iloc[:, 1].tolist()

# 处理每个文件夹
for index, folder in enumerate(sorted_folders):
    subject_point_file = os.path.join(dir_path, folder)
    all_point_items = os.listdir(subject_point_file)
    motion_folders = [item for item in all_point_items if item.startswith('rv_es')]
    sub_point_list = []
    for motion_file in motion_folders:
        sub_point_txt = os.path.join(subject_point_file, motion_file)
        sub_point_txt_data = np.loadtxt(sub_point_txt)
        sub_point_list.append(sub_point_txt_data)

    sub_point_xyz = sub_point_list[0][:, 0:3]
    sub_point_wall = sub_point_list[0][:, -1]
    sub_point_curve = sub_point_list[1][:, -1]
    sub_point_sign = sub_point_list[2][:, -1]

    sub_point_all = np.column_stack((sub_point_xyz, sub_point_wall, sub_point_curve, sub_point_sign))

    print(f"{subject_point_file} done!")

    state = 0 if label_list[index] <= 35 else 1
    save_path_name = os.path.join(output_path, f"{folder}_{state}.npz")
    np.savez(save_path_name, point=sub_point_all)
