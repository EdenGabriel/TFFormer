
import torch
import os
import numpy as np


egovlp_dir = '/SSD1/yj/ego4d/egovlp_features/egovlp_egonce'
egovlp_out_dir = '/SSD1/yj/ego4d/converted/features/egovlp_features'
file_ext = '.npy'

pt_files=os.listdir(egovlp_dir)

# pt_data=torch.load(os.path.join(egovlp_dir,"ffbe6808-cb5e-447d-958f-f0d507de946f.pt"))
# print(pt_data.size())

for file in pt_files:
    file_name = file.split('.')[0]
    # print(os.path.join(egovlp_dir,file))
    pt_data=torch.load(os.path.join(egovlp_dir,file))
    np.save(os.path.join(egovlp_out_dir,file_name+file_ext),pt_data)

# print(pt_files)