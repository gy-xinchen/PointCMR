# -*- coding: utf-8 -*-
# @Time    : 2024/11/14 21:01
# @Author  : yuan
# @File    : for_ROC_balance.py
import os
import re

dir_path = r"/root/autodl-tmp/data/4D_motion_classification35_npz"
files = os.listdir(dir_path)
sorted_files = sorted(files, key=lambda x: int(re.search(r'_(\d+)_', x).group(1)))
for file in sorted_files:
    print(file.split('_')[-1].split(".")[0])
