import os.path
import skimage.io
import matplotlib.pyplot as plt
import numpy as np

bandon_path=r'D:\Documents\Dataset\building_footprint\BANDON\train\imgs\bj\t1/'
bonai_path=r'D:\Documents\Dataset\building_footprint\BONAI\train\images/'
row_base_id_bandon=663
column_base_id_bandon=784159
row_base_id_baonai=99336
column_base_id_bonai=215840

row_range=range(639-80-663,1183+1+80-663,8)
column_range=range(783959-80-784159,784351+1+80-784159,8)

bandon_names=[]
bandon_matches=[]
matched_bandon_names=[]
bonai_only_names=[]
for r in row_range:
    bandon_row = row_base_id_bandon + r
    bonai_row = row_base_id_baonai - r
    for c in column_range:
        bandon_column=column_base_id_bandon+c
        bonai_column=column_base_id_bonai-c
        bandon_name="L81_%05d_%d.jpg"%(bandon_row,bandon_column)
        bonai_name="beijing_arg__L18_%d_%d__0_0.png"%(bonai_row,bonai_column)

        if os.path.exists(bandon_path+bandon_name):
            bandon_names.append(bandon_name)
            if os.path.exists(bonai_path+bonai_name):
                # bandon_img=skimage.io.imread(bandon_path+bandon_name)
                # bonai_img=skimage.io.imread(bonai_path+bonai_name)
                # plt.imshow(bandon_img)
                # plt.show()
                # plt.imshow(bonai_img)
                # plt.show()
                bandon_matches.append(1)
                matched_bandon_names.append(bandon_name)
            else:
                bandon_matches.append(0)
        elif os.path.exists(bonai_path+bonai_name):
            bonai_only_names.append(bonai_name)
print(matched_bandon_names)
# bonai ���� ��57��ͼ��2048�ģ��ļ���ÿ��2048�ķ�Ϊ4��1024�ģ�������24������ƥ���ϣ�33��ƥ�䲻��
# bandon ���� ��400+��ͼ��2048��������24������ƥ���ϣ��и߶ȣ���Ŀ���ע����������ƥ�䣨ֻ������ָ��ע��