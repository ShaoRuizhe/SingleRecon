import mmcv
import numpy as np
import shapely
from matplotlib import pyplot as plt
from shapely.plotting import plot_polygon

from build_detection import BONAI_building_detection
from PFFL_polygonize import PFFL_seg,PFFL_polygonize

def get_target_segs(targets,seg):
    bboxes=np.round(targets[0]['bbox'])[0].astype(int)
    target_segs={'interior_seg':[],'edge_seg':[],'crossfield':[]}
    for bbox in bboxes:
        target_segs['interior_seg'].append(seg['interior_seg'][0,:,bbox[1]:bbox[3],bbox[0]:bbox[2]])
        target_segs['edge_seg'].append(seg['edge_seg'][0,:,bbox[1]:bbox[3],bbox[0]:bbox[2]])
        target_segs['crossfield'].append(seg['crossfield'][0,:,bbox[1]:bbox[3],bbox[0]:bbox[2]])
    return target_segs

def show_polygonized_result(img,polygons_batch,building_targets):
    num_obj=len(polygons_batch)
    bbox_result = building_targets.get('bbox', None)[0]
    segm_result = building_targets.get('segm', None)[0]
    img_temp=img.copy()
    # for i in range(num_obj):
    #     # for i in inds[6:7]:
    #     if segm_result is not None:
    #         mask = segm_result[i] > 0.5
    #         img_temp[:, :, 0][mask] = (img[:, :, 0][mask] * 0.2 + 255 * np.ones((1024, 1024))[mask] * 0.8).astype(
    #             np.uint8)
    plt.figure(figsize=(10, 8), dpi=150)
    plt.imshow(img_temp)
    ax = plt.gca()
    facecolor=np.array([0, 0, 0, 0],dtype=float)
    edgecolor=np.array([0, 1, 0, 1],dtype=float)
    linewidth=1
    markersize=2
    for i in range(num_obj):
        bbox = bbox_result[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], color="blue",
                          fill=False,
                          linewidth=1))
    for i,polygon in enumerate(polygons_batch):
        if polygon is not None:
            polygon = shapely.affinity.translate(polygon, xoff=bbox_result[i][0], yoff=bbox_result[i][1])
            p, _ = plot_polygon(polygon, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
            ax.add_patch(p)

            ax.plot(*polygon.exterior.xy, marker="o", color=edgecolor, markersize=markersize)
    plt.show()

def single_test(img,img_meta):
    seg = PFFL_seg(img)
    # show_result
    building_targets=BONAI_building_detection(img,img_meta,show=True)

    target_segs=get_target_segs(building_targets,seg)
    polygons_batch,probs_batch=PFFL_polygonize(target_segs)
    show_polygonized_result(img,polygons_batch,building_targets[0])

if __name__ == '__main__':
    # image_file=r'D:\Documents\Dataset\building_footprint\BONAI\rotated\test\images/L18_104400_210392__0_0.png'
    image_file=r'D:\Documents\Dataset\building_footprint\BONAI\rotated\test\images/L18_104512_210416__0_1024.png'
    file_client=mmcv.FileClient()
    img_bytes=file_client.get(image_file)
    img=mmcv.imfrombytes(img_bytes)
    # pipline
    img_meta={'img_shape':img.shape,'scale_factor':np.array([1.,1.,1.,1.],dtype=np.float32),'ori_shape':(1024,1024,3)}
    single_test(img,img_meta)