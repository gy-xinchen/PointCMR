import csv
import os
import re
import pandas as pd
import numpy as np
import concurrent.futures

dir_path = r"D:\CMR-res\4D_segment\all_mPAP_4dnii_data"
label_path = r"D:\CMR-res\4D_segment\all_mPAP_4dnii_data\mPAP.csv"

output_path = r"D:\CMR-res\4Dsegment-master\2_class35_motion_point_npz"
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



def process_folder(index, folder):
    subject_motion_file = os.path.join(dir_path, folder, "motion")
    all_motion_items = os.listdir(subject_motion_file)
    motion_folders = [item for item in all_motion_items if item.endswith('.txt')]
    sub_motion_list = []
    for motion_file in motion_folders:
        sub_motion_txt = os.path.join(subject_motion_file, motion_file)
        sub_motion_txt_data = np.loadtxt(sub_motion_txt)
        sub_motion_list.append(sub_motion_txt_data)
    print(f"{subject_motion_file} done!")
    state = 0 if label_list[index] <= 35 else 1
    save_path_name = os.path.join(output_path, f"{folder}_{state}.npz")
    np.savez(save_path_name, sub_motion_list)

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_folder, i, sorted_folders[i]) for i in range(len(sorted_folders))]
    for future in concurrent.futures.as_completed(futures):
        future.result()
