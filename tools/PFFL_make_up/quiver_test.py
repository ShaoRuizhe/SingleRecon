# encoding:utf-8
import numpy as np
import torch
import matplotlib.pyplot as plt

a=torch.load(r'D:\Documents\Dataset\building_footprint\BONAI\rotated\processeed\train\shanghai_arg__L18_106968_219360__0_0.pt')
show_arrow_interval=20
image_shape=(2000,2000)
x, y = np.meshgrid(np.arange(0, image_shape[0], show_arrow_interval),np.arange(0, image_shape[1], show_arrow_interval))
arrow_len=4
u,v=arrow_len*np.sin(a['gt_crossfield_angle']/255*np.pi),-arrow_len*np.cos(a['gt_crossfield_angle']/255*np.pi)
u[np.bitwise_and(u==0,v==-arrow_len)]=0
v[np.bitwise_and(u==0,v==-arrow_len)]=0
# bbox=[46,110,580,686]
bbox=[400,600,300,500]
step=4
width=int((bbox[1]-bbox[0])/step)
height=int((bbox[3]-bbox[2])/step)
u_bbox=u[bbox[2]:bbox[3]:step,bbox[0]:bbox[1]:step]
v_bbox=v[bbox[2]:bbox[3]:step,bbox[0]:bbox[1]:step]
# u_bbox=u_bbox[::-1]
# v_bbox=v_bbox[::-1]

plt.figure(figsize=(width,height),dpi=100)
ax = plt.gca()
ax.invert_yaxis()
plt.quiver(x[:height,:width],y[:height,:width],u_bbox,v_bbox,pivot='tail',scale=200)# 建筑物边的方向
plt.quiver(x[:height,:width],y[:height,:width],-v_bbox,u_bbox,pivot='tail',scale=200)# 建筑物边的垂直方向
plt.show()
angle=a['gt_crossfield_angle'][bbox[2]:bbox[3]:step,bbox[0]:bbox[1]:step]
pass