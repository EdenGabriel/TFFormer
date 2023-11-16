
import torch
import os
import numpy as np
from torch.nn import functional as F

egovlp_dir = '/SSD1/yj/ego4d/converted/features/egovlp_features'
omnivore_dir = '/SSD1/yj/ego4d/converted/features/omnivore_features'
egovlp_out_dir = '/SSD1/yj/ego4d/converted/features/egovlp_to_omnivore_slowfast_features'
file_ext = '.npy'

vlp_files=os.listdir(egovlp_dir)
vore_files=os.listdir(omnivore_dir)
assert len(vlp_files) == len(vore_files), "len(vlp_files) != len(vore_files)"
# pt_data=torch.load(os.path.join(egovlp_dir,"ffbe6808-cb5e-447d-958f-f0d507de946f.pt"))
# print(pt_data.size())

for idx,file_name in enumerate(vlp_files):
    if vore_files[idx]!=file_name:
        print(file_name)
    vlp_feat = np.load(os.path.join(egovlp_dir,file_name))
    vore_feat = np.load(os.path.join(omnivore_dir,file_name))

    vlp_feat_tmp = torch.from_numpy(np.ascontiguousarray(vlp_feat.transpose()))

    if vlp_feat.shape[0] != vore_feat.shape[0]:
        print("vlp_feat.shape:",vlp_feat.shape,"vore_feat.shape",vore_feat.shape)
        resize_vlp_feat = F.interpolate(
            vlp_feat_tmp.unsqueeze(0),
            size=vore_feat.shape[0],
            mode='linear',
            align_corners=False
        )
        print(type(resize_vlp_feat),resize_vlp_feat.size())
        vlp_feat_final = resize_vlp_feat.squeeze().numpy().transpose()
        print(vlp_feat_final.shape)

        np.save(os.path.join(egovlp_out_dir,file_name),vlp_feat_final)
    else:
        np.save(os.path.join(egovlp_out_dir,file_name),vlp_feat)

# for file in pt_files:
#     file_name = file.split('.')[0]
#     # print(os.path.join(egovlp_dir,file))
#     pt_data=torch.load(os.path.join(egovlp_dir,file))
#     np.save(os.path.join(egovlp_out_dir,file_name+file_ext),pt_data)

# print(pt_files)