# encoding:utf-8
# annotation.json+images=>.pt（rastered_data)
'''
    本脚本的作用：根据json标注，制作crossfield，并默认是旋转到offset水平向右的图片，据此绘制确定facade范围，制作边缘，顶点，分割的多分类mask，
    将图片，crossfield和mask记录为.pt文件。
    通过先修改373行左右的路径（输入是json_file,img_path等，输出位置为processed_save_path)，然后直接运行此脚本即可
    使用的具体方法为：
        for img in data:
            compute_raster_distances_sizes 将输入的polygon和offset组成的offnadir建筑物结构绘制在尺寸为shape的图上。
                for polygon in polygons:
                    draw_polygon_offnadir 给定bbox shape，绘制单个polygon在bbox内部的seg（包括interior,seg,vertex)，
            init_angle_field_offnadir 将polygon-offset组成的offnadir建筑物的crossfield场绘制在尺寸为shape的mask上。
                for polygon in polygons:
                    draw_linear_ring_offnadir   在参数给的draw上。 绘制单个polygon的angle field
        torch.save
'''

import math
import os.path

import PIL.Image
import cv2
import mmcv
import numpy as np
import shapely
import shapely.affinity
import shapely.geometry
import skimage.morphology
import torch
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
from tqdm import tqdm

horizental_edge_thre=0.1# 纵向/横向小于此阈值，则判定边为横边
def compute_distances(distance_maps):
    distance_maps.sort(axis=2)
    distance_maps = distance_maps[:, :, :2]
    distances = np.sum(distance_maps, axis=2)
    return distances

def draw_circle(draw, center, radius, fill):
    draw.ellipse([center[0] - radius,
                  center[1] - radius,
                  center[0] + radius,
                  center[1] + radius], fill=fill, outline=None)

def draw_polygons(polygons, shape, fill=True, edges=True, vertices=True, line_width=3, antialiasing=False):
    assert type(polygons) == list, "polygons should be a list"
    assert type(polygons[0]) == shapely.geometry.Polygon, "polygon should be a shapely.geometry.Polygon"

    if antialiasing:
        draw_shape = (2 * shape[0], 2 * shape[1])
        polygons = [shapely.affinity.scale(polygon, xfact=2.0, yfact=2.0, origin=(0, 0)) for polygon in polygons]
        line_width *= 2
    else:
        draw_shape = shape
    # Channels
    fill_channel_index = 0  # Always first channel
    edges_channel_index = fill  # If fill == True, take second channel. If not then take first
    vertices_channel_index = fill + edges  # Same principle as above
    channel_count = fill + edges + vertices
    im_draw_list = []
    for channel_index in range(channel_count):
        im = Image.new("L", (draw_shape[1], draw_shape[0]))
        im_px_access = im.load()
        draw = ImageDraw.Draw(im)
        im_draw_list.append((im, draw))

    for polygon in polygons:
        if fill:
            draw = im_draw_list[fill_channel_index][1]
            draw.polygon(polygon.exterior.coords, fill=255)
            for interior in polygon.interiors:
                draw.polygon(interior.coords, fill=0)
        if edges:
            draw = im_draw_list[edges_channel_index][1]
            draw.line(polygon.exterior.coords, fill=255, width=line_width)
            for interior in polygon.interiors:
                draw.line(interior.coords, fill=255, width=line_width)
        if vertices:
            draw = im_draw_list[vertices_channel_index][1]
            for vertex in polygon.exterior.coords:
                draw_circle(draw, vertex, line_width / 2, fill=255)
            for interior in polygon.interiors:
                for vertex in interior.coords:
                    draw_circle(draw, vertex, line_width / 2, fill=255)

    im_list = []
    if antialiasing:
        # resize images:
        for im_draw in im_draw_list:
            resize_shape = (shape[1], shape[0])
            im_list.append(im_draw[0].resize(resize_shape, Image.BILINEAR))
    else:
        for im_draw in im_draw_list:
            im_list.append(im_draw[0])

    # Convert image to numpy array with the right number of channels
    array_list = [np.array(im) for im in im_list]
    array = np.stack(array_list, axis=-1)
    return array
def get_font_back_edge(polygon):
    font_edge = []
    back_edge = []
    if isinstance(polygon,shapely.geometry.Polygon):
        coords=polygon.exterior.coords
    elif isinstance(polygon,list) or isinstance(polygon,np.ndarray):
        coords=polygon
    coords=np.array(coords)
    below_vex = np.argmax(coords[:,1])# 0-x 1-y
    edge_between = coords[below_vex] - coords[(below_vex + 1) % (len(coords-1))]
    if edge_between[0]!=0 and abs(edge_between[1] / edge_between[0]) < 0.1:  # 如果边过于平
        below_vex = (below_vex + 1) % (len(coords-1))
    upper_vex = np.argmin(coords[:,1])
    edge_between=coords[upper_vex]-coords[(upper_vex-1)%(len(coords-1))]
    if edge_between[0]!=0 and abs(edge_between[1]/edge_between[0])<0.1:# 如果边过于平
        upper_vex=(upper_vex-1)%(len(coords-1))

    num_vex = len(coords) - 1  # 最后一个是重复的
    if upper_vex<below_vex:
        upper_vex+=num_vex# 多边形顺时针 默认从below开始，先构建font，再构建back
    for i in range(num_vex+1):
        if below_vex + i <= upper_vex:
            font_edge.append(tuple(coords[int((below_vex + i) % num_vex)]))
        if below_vex + i >= upper_vex:
            back_edge.append(tuple(coords[int((below_vex + i) % num_vex)]))

    return font_edge,back_edge

def draw_polygon_offnadir(polygon, offset,shape, line_width=3, antialiasing=False):
    '''
    给定bbox shape，绘制单个polygon在bbox内部的seg（包括interior,seg,vertex)，
    Args:
        polygon:
        offset:
        shape:
        line_width:
        antialiasing:

    Returns:
            输出为channel=3的array: array(shape,3) 3个channel分别是
                0：建筑物屋顶/侧面分割图 mask中有3个值： [0,128,255]0-background 128-facade 255-roof
                1：建筑物边缘 mask中有4个值：[0,85,170,255] 0-background 85-roof_font_edges 170-roof_back_edges 255-side_edges
                2：角点 mask中有两个值：[0,255] 0-background 255-角点（以roof多边形角点为圆心，半径line_width的圆）
    '''

    assert type(polygon) == shapely.geometry.Polygon, "polygon should be a shapely.geometry.Polygon"

    if antialiasing:
        draw_shape = (2 * shape[0], 2 * shape[1])
        polygon = shapely.affinity.scale(polygon, xfact=2.0, yfact=2.0, origin=(0, 0))
        line_width *= 2
        offset=np.array(offset)*2
    else:
        draw_shape = shape


    fill_im = Image.new("L", (draw_shape[1], draw_shape[0]))
    fill_draw = ImageDraw.Draw(fill_im)
    edge_im = Image.new("L", (draw_shape[1], draw_shape[0]))
    edge_draw = ImageDraw.Draw(edge_im)
    vex_im = Image.new("L", (draw_shape[1], draw_shape[0]))
    vex_draw = ImageDraw.Draw(vex_im)

    font_edge,back_edge=get_font_back_edge(polygon)
    footprint_font_edge = [(vex[0] - offset[0], vex[1]) for vex in font_edge]

    # fill roof:
    fill_draw.polygon(polygon.exterior.coords, fill=255)
    for interior in polygon.interiors:
        fill_draw.polygon(interior.coords, fill=255)
    # fill font face
    font_face=font_edge+footprint_font_edge[::-1]
    fill_draw.polygon(font_face, fill=128)

    # # roof font edges:
    edge_draw.line(font_edge, fill=85, width=line_width)
    for interior in polygon.interiors:
        edge_draw.line(interior.coords, fill=85, width=line_width)
    # roof back edges:
    edge_draw.line(back_edge, fill=170, width=line_width)
    # side edges
    edge_draw.line((font_edge[0],footprint_font_edge[0]),fill=255, width=line_width)
    edge_draw.line((font_edge[-1],footprint_font_edge[-1]),fill=255, width=line_width)
    # footprint font edges:
    # edge_draw.line(footprint_font_edge, fill=255, width=line_width)

    # vertices:
    for vertex in polygon.exterior.coords:
        draw_circle(vex_draw, vertex, line_width / 2, fill=255)


    if antialiasing:
        # resize images:
        resize_shape = (shape[1], shape[0])
        fill_im=fill_im.resize(resize_shape, Image.BILINEAR)
        edge_im=edge_im.resize(resize_shape, Image.BILINEAR)
        vex_im=vex_im.resize(resize_shape, Image.BILINEAR)

    out_array=np.stack([np.array(fill_im),np.array(edge_im),np.array(vex_im)],axis=-1)
    return out_array

def compute_raster_distances_sizes(polygons, offsets,shape, fill=True, edges=True, vertices=True, line_width=3, antialiasing=False):
    """
    将输入的polygon和offset组成的offnadir建筑物结构绘制在尺寸为shape的图上。
    绘制的方法是对每个polygon-offset，计算其外接矩形框，然后调用draw_polygon_offnadir绘制框内的的mask，再将框内mask贴到整体mask上。
    使用框而不直接在mask上操作是为了节约计算时间，在大图上找polygon内/外元素是很耗时的。
    Returns:
         - distances: sum of distance to closest and second-closest annotation for each pixel.
         - size: relative size (normalized by image area) of annotation the pixel belongs to.
         polygons_raster：1024，1024，3的array
            3个channel分别是
                0：建筑物屋顶/侧面分割图 mask中有3个值： [0,128,255]0-background 128-facade 255-roof
                1：建筑物边缘 mask中有4个值：[0,85,170,255] 0-background 85-roof_font_edges 170-roof_back_edges 255-side_edges
                2：角点 mask中有两个值：[0,255] 0-background 255-角点（以roof多边形角点为圆心，半径line_width的圆）
         instance_indicator：array(1024,1024)type-uint8 其中记录每个像素属于的建筑物的序号，background为255
    """
    assert type(polygons) == list, "polygons should be a list"

    # Filter out zero-area polygons
    polygons = [polygon for polygon in polygons if 0 < polygon.area]

    # tic = time.time()

    channel_count = fill + edges + vertices
    polygons_raster = np.zeros((*shape, channel_count), dtype=np.uint8)
    instance_indicator = np.ones((*shape,), dtype=np.uint8)*255
    distance_maps = np.ones((*shape, len(polygons)))  # Init with max value (distances are normed)
    sizes = np.ones(shape)  # Init with max value (sizes are normed)
    image_area = shape[0] * shape[1]
    for i, polygon in enumerate(polygons):
        # 找到bbox 此时只去掉整个在边界外的建筑物，而部分在边界内的问题贴上去的时候再考虑
        minx, miny, maxx, maxy = polygon.bounds
        if offsets[i][0]>0:
            minx = minx-offsets[i][0]
        else:
            maxx = maxx + offsets[i][0]
        if offsets[i][1]>0:
            miny = miny-offsets[i][1]
        else:
            maxy = maxy + offsets[i][1]
        if minx>maxx:
            continue
        if minx > shape[1] or miny > shape[0] or maxx < 0 or maxy < 0:
            continue
        mini = math.floor(miny) - 2*line_width
        minj = math.floor(minx) - 2*line_width
        maxi = math.ceil(maxy) + 2*line_width
        maxj = math.ceil(maxx) + 2*line_width
        bbox_shape = (maxi - mini, maxj - minj)
        # 将polygon转换为bbox内的polygon
        bbox_polygon = shapely.affinity.translate(polygon, xoff=-minj, yoff=-mini)
        # 绘制mask
        bbox_raster = draw_polygon_offnadir(bbox_polygon, offsets[i],bbox_shape, line_width, antialiasing)
        # 将bbox内的绘好的mask贴到大mask上
        if mini<0:
            bbox_raster=bbox_raster[-mini:]
            mini = 0
        if minj<0:
            bbox_raster=bbox_raster[:,-minj:]
            minj = 0
        patch_shape=polygons_raster[mini:maxi, minj:maxj].shape
        bbox_raster=bbox_raster[:patch_shape[0],:patch_shape[1]]
        polygons_raster[mini:maxi, minj:maxj] = np.maximum(polygons_raster[mini:maxi, minj:maxj], bbox_raster)
        assert i<255,'num instances max out'
        instance_indicator[mini:maxi, minj:maxj][np.any(bbox_raster,axis=-1)]=i
        bbox_mask = 0 < np.sum(bbox_raster, axis=2)  # Polygon interior + edge + vertex
        if bbox_mask.max():  # Make sure mask is not empty
            polygon_mask = np.zeros(shape, dtype=bool)
            polygon_mask[mini:maxi, minj:maxj] = bbox_mask
            polygon_dist = cv2.distanceTransform(1 - polygon_mask.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_5,
                                        dstType=cv2.CV_64F)
            polygon_dist /= (polygon_mask.shape[0] + polygon_mask.shape[1])  # Normalize dist
            distance_maps[:, :, i] = polygon_dist

            selem = skimage.morphology.disk(line_width)
            bbox_dilated_mask = skimage.morphology.binary_dilation(bbox_mask, selem=selem)
            sizes[mini:maxi, minj:maxj][bbox_dilated_mask] = polygon.area / image_area

    polygons_raster = np.clip(polygons_raster, 0, 255)
    # skimage.io.imsave("polygons_raster.png", polygons_raster)

    if edges:
        edge_channels = -1 + fill + edges
        # Remove border edges because they correspond to cut buildings:
        polygons_raster[:line_width, :, edge_channels] = 0
        polygons_raster[-line_width:, :, edge_channels] = 0
        polygons_raster[:, :line_width, edge_channels] = 0
        polygons_raster[:, -line_width:, edge_channels] = 0

    distances = compute_distances(distance_maps)
    # skimage.io.imsave("distances.png", distances)

    distances = distances.astype(np.float16)
    sizes = sizes.astype(np.float16)

    # toc = time.time()
    # print(f"Rasterize {len(polygons)} polygons: {toc - tic}s")

    return polygons_raster, distances, sizes,instance_indicator

def init_angle_field(polygons, shape, line_width=1):
    """
    Angle field {\theta_1} the tangent vector's angle for every pixel, specified on the polygon edges.
    Angle between 0 and pi.
    This is not invariant to symmetries.

    :param polygons:
    :param shape:
    :return: (angles: np.array((num_edge_pixels, ), dtype=np.uint8),
              mask: np.array((num_edge_pixels, 2), dtype=np.int))
    """
    assert type(polygons) == list, "polygons should be a list"
    if len(polygons):
        assert type(polygons[0]) == shapely.geometry.Polygon, "polygon should be a shapely.geometry.Polygon"

    im = Image.new("L", (shape[1], shape[0]))
    im_px_access = im.load()
    draw = ImageDraw.Draw(im)

    for polygon in polygons:
        draw_linear_ring_offnadir(draw, polygon.exterior, line_width)
        for interior in polygon.interiors:
            draw_linear_ring(draw, interior, line_width)

    # Convert image to numpy array
    array = np.array(im)
    return array

def draw_linear_ring(draw, linear_ring, line_width):
    # --- edges:
    coords = np.array(linear_ring.xy).transpose()
    edge_vect_array = np.diff(coords, axis=0)
    edge_angle_array = np.angle(edge_vect_array[:, 1] + 1j * edge_vect_array[:, 0])  # ij coord sys
    neg_indices = np.where(edge_angle_array < 0)
    edge_angle_array[neg_indices] += np.pi

    first_uint8_angle = None
    for i in range(coords.shape[0] - 1):
        edge = (coords[i], coords[i + 1])
        angle = edge_angle_array[i]
        uint8_angle = int((255 * angle / np.pi).round())
        if first_uint8_angle is None:
            first_uint8_angle = uint8_angle
        line = [(edge[0][0], edge[0][1]), (edge[1][0], edge[1][1])]
        draw.line(line, fill=uint8_angle, width=line_width)
        draw_circle(draw, line[0], radius=line_width / 2, fill=uint8_angle)

    # Add first vertex back on top (equals to last vertex too):
    draw_circle(draw, line[1], radius=line_width / 2, fill=first_uint8_angle)

def init_angle_field_offnadir(polygons,offsets, shape, line_width=1):
    """
    将polygon-offset组成的offnadir建筑物的crossfield场绘制在尺寸为shape的mask上。
    具体的方法为构建一个draw，然后对每个polygon-offset调用draw_linear_ring_offnadir将polygon-offset绘制其上。

    :param polygons:
    :param shape:
    :return: array(1024,1024)其中每个像素值域为[0，255],是角度经过int((255 * angle / np.pi).round())映射之后的uint8类型数值，background部分的数值为0
    """
    assert type(polygons) == list, "polygons should be a list"
    if len(polygons):
        assert type(polygons[0]) == shapely.geometry.Polygon, "polygon should be a shapely.geometry.Polygon"

    im = Image.new("L", (shape[1], shape[0]))
    im_px_access = im.load()
    draw = ImageDraw.Draw(im)

    for i,polygon in enumerate(polygons):
        minx, miny, maxx, maxy = polygon.bounds
        minx = minx-offsets[i][0]
        draw_linear_ring_offnadir(draw, polygon.exterior,offsets[i] ,line_width)
        for interior in polygon.interiors:
            draw_linear_ring(draw, interior, line_width)

    # Convert image to numpy array
    array = np.array(im)
    return array

def draw_linear_ring_offnadir(draw, linear_ring, offset,line_width):
    # 在参数给的draw上。 绘制单个polygon的angle field
    # 绘制的值域为[0，255],是角度经过int((255 * angle / np.pi).round())映射之后的uint8类型数值
    coords = np.array(linear_ring.xy).transpose()
    font_edge,back_edge=get_font_back_edge(coords)
    
    back_edge_vect_array = np.diff(back_edge, axis=0)
    back_edge_angle_array = np.angle(back_edge_vect_array[:, 1] + 1j * back_edge_vect_array[:, 0])  # ij coord sys
    neg_indices = np.where(back_edge_angle_array < 0)
    back_edge_angle_array[neg_indices] += np.pi

    first_uint8_angle = None
    for i in range(len(back_edge) - 1):
        edge = (back_edge[i], back_edge[i + 1])
        angle = back_edge_angle_array[i]
        uint8_angle = int((255 * angle / np.pi).round())
        if first_uint8_angle is None:
            first_uint8_angle = uint8_angle
        line = [(edge[0][0], edge[0][1]), (edge[1][0], edge[1][1])]
        draw.line(line, fill=uint8_angle, width=line_width)
        draw_circle(draw, line[0], radius=line_width / 2, fill=uint8_angle)
    
    font_edge_vect_array = np.diff(font_edge, axis=0)
    font_edge_angle_array = np.angle(font_edge_vect_array[:, 1] + 1j * font_edge_vect_array[:, 0])  # ij coord sys
    neg_indices = np.where(font_edge_angle_array < 0)
    font_edge_angle_array[neg_indices] += np.pi
    
    for i in range(len(font_edge) - 1):
        edge = (font_edge[i], font_edge[i + 1])
        angle = font_edge_angle_array[i]
        uint8_angle = int((255 * angle / np.pi).round())
        facade_poly=[font_edge[i], font_edge[i + 1],
                      (font_edge[i+1][0]-offset[0],font_edge[i+1][1]),
                      (font_edge[i][0]-offset[0],font_edge[i][1])]
        draw.polygon(facade_poly,fill=uint8_angle)
        line = [(edge[0][0], edge[0][1]), (edge[1][0], edge[1][1])]
        draw.line(line, fill=uint8_angle, width=line_width)
        # draw.line([facade_poly[2],facade_poly[3]], fill=uint8_angle, width=line_width)
        draw_circle(draw, line[0], radius=line_width / 2, fill=uint8_angle)

    # Add first vertex back on top (equals to last vertex too):
    draw_circle(draw, line[1], radius=line_width / 2, fill=first_uint8_angle)

# annotation.json+images=>.pt
# read Bonai rotated
bonai_data_root = r'/datapool/data/BONAI/rotated'

# 经历过三次调整，第一次rotated_8将边缘的宽度增加到8，第二次调整rotated_8_adjust_edge了某些roof具有的近似水平的边缘，在前侧边缘和后侧边缘的划分中，将这些边缘全部划分为后侧（远离facade的一侧），以防止出现狭细的一条facade
# 第三次调整为保存的数据增加了instance_indicator,用于指示seg属于那个目标
# training set
processed_save_path = r'/datapool/data/BONAI/rotated_24_makeup/processed/train'
os.makedirs(processed_save_path,exist_ok=True)
json_file=os.path.join(bonai_data_root,'train/annotation.json')
img_path=os.path.join(bonai_data_root,'train/images')

# # test set
# processed_save_path = r'/datapool/data/BONAI/rotated_24_makeup/processed/test'
# os.makedirs(processed_save_path,exist_ok=True)
# json_file=os.path.join(bonai_data_root,'test/annotation.json')
# img_path=os.path.join(bonai_data_root,'test/images')

# no rotate training set(240228)
# bonai_data_root = r'/datapool/data/BONAI'
# processed_save_path = r'/datapool/data/BONAI/crossfield_bonai/processed/train'
# os.makedirs(processed_save_path,exist_ok=True)
# json_file=os.path.join(bonai_data_root,'train/bonai_cat_train.json')
# img_path=os.path.join(bonai_data_root,'train/images')

# # # no rotate test set
# bonai_data_root = r'/datapool/data/BONAI'
# processed_save_path = r'/datapool/data/BONAI/crossfield_bonai/processed/test'
# os.makedirs(processed_save_path,exist_ok=True)
# json_file=os.path.join(bonai_data_root,'test/bonai_shanghai_xian_test.json')
# img_path=os.path.join(bonai_data_root,'test/images')

coco_data=COCO(json_file)
image_info=coco_data.loadAnns()
# PFFL make up
for i,img in tqdm(coco_data.imgs.items()):
# for i,img_anns in tqdm(coco_data.imgToAnns.items()):
    line_width=8
    # if not 'beijing_arg__L18_99136_216016__1024_1024' in coco_data.imgs[i]['file_name']:
    #     continue
    img_shape=(coco_data.imgs[i]['height'],coco_data.imgs[i]['width'])
    img = PIL.Image.open(os.path.join(img_path,coco_data.imgs[i]['file_name']))
    polygons=[]
    if i in coco_data.imgToAnns:
        for img_ann in coco_data.imgToAnns[i]:
            polygons.append(shapely.geometry.Polygon(np.array(img_ann['segmentation']).reshape(-1,2)))
        offsets=[ann['offset'] for ann in coco_data.imgToAnns[i]]
        out=compute_raster_distances_sizes(polygons=polygons,offsets=offsets,shape=img_shape,line_width=line_width,antialiasing=False)
        out_angle=init_angle_field_offnadir(polygons,offsets,img_shape,line_width)
    else:
        out=tuple([np.zeros((img.height,img.width,3),np.uint8)])
        out_angle=np.zeros((img.height,img.width),np.uint8)
    # save to file
    torch.save({'image':np.array(img),'gt_polygon_image':out[0],'gt_crossfield_angle':out_angle,'instance_indicator':out[-1]},
               os.path.join(processed_save_path,coco_data.imgs[i]['file_name'].split('.')[0]+'.pt'))

    # mmcv.imwrite(out[0])
    # plt.imshow(out[0])
    # plt.show()
    # plt.imshow(out_angle)
    # plt.show()
    pass


