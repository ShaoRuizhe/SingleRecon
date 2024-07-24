# encoding:utf-8
import os

import shapely
from shapely import intersection, Polygon, box
import cv2
import numpy as np
import matplotlib.pyplot as plt
from osgeo import ogr, osr

def save_3dbuilding(img,building_dicts,outdir,height_scale=1):
    """

    Args:
        img: 整张输入图片，需要原始图像，即HWC，0~255
        building_dicts是list of building_dict，其中需要用到的有两项:
                        {'roof_mask':roof多边形 list形式[vex_x,vex_y,...]或shapely.polygon形式
                        'offset':[offset_x,offset_y],
                        }
    输出:
        存储建筑物信息到geojson文件，存储纹理到图片文件
    """
    # todo:
    #   多个建筑物的构建
    #   被挡住怎么办
    #   超出图片画幅怎么办
    #   offset方向处理
    # show_roof_on_img(img,building_dict)

    roof_polys=[]
    offsets=[]
    texture_area_list=[]
    texture_files=[]
    os.makedirs(outdir, exist_ok=True)
    for i,building_dict in enumerate(building_dicts):
        if isinstance(building_dict['roof_mask'],shapely.Polygon):
            roof_poly = np.array(building_dict['roof_mask'].exterior.xy).transpose(1, 0)
        else:
            roof_poly = np.array(building_dict['roof_mask']).reshape(-1, 2)
        roof_poly=poly_in_img(roof_poly,img.shape)
        if len(roof_poly)==0:
            continue
        # roof_bbox = np.array(building_dict['roof_bbox']).astype(int)
        offset=building_dict['offset']
        if abs(offset[0])<1:# todo:会处理offset方向之后解决此问题：建筑物高度过低，offset看不到墙面怎么办
            continue
        roof_polys.append(roof_poly)
        offsets.append(offset)

        texture_areas=save_texture2img(img,roof_poly,offset,outdir+'/%.3d_texture.png'%i,height_scale)
        texture_area_list.append(texture_areas)
        texture_files.append(outdir+'/%.3d_texture.png'%i)
    save_building2geojson(roof_polys,offsets, texture_area_list,
                          texture_files=texture_files,
                          outdir=outdir + '/buildings.geojson',
                          height_scale=height_scale)

    cv2.imwrite(outdir+'/base_img.png',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    save_base_map2geojson(base_image_file=outdir+'/base_img.png',outdir=outdir + '/base_map.geojson')

def poly_in_img(poly,img_shape):
    """

    Args:
        poly: np.array([[vex_x,vex_y],[]...]
        img_shape:

    Returns:
        poly在img_shape范围内的部分，并且将顶点顺序调整为顺时针,形式也是np.array([[vex_x,vex_y],[]...]
    """
    try:
        poly = intersection(Polygon(poly).buffer(0), box(0, 0, img_shape[0], img_shape[1]))
    except:
        print(1)
    # 获取顺时针roof_poly 注意！！由于y坐标是反的，图像上的“顺时针”和shapely定义的“顺时针”是相反的，因此要获取shapely.is_ccw=False的
    if not isinstance(poly,shapely.Polygon):
        return []
    if poly.exterior.is_ccw:
        poly= np.asarray(poly.exterior.xy).transpose(1, 0)
    else:
        poly=np.asarray(poly.exterior.xy).transpose(1, 0)[::-1]
    if len(poly)==0:
        return []
    if (poly[0] == poly[-1]).all():
        poly= poly[:-1]
    return poly

def save_texture2img(img,roof_poly,offset,outdir,height_scale=1):
    roof_texture=get_roof_texture(img,roof_poly)
    wall_texture=get_wall_texture_hori(img,roof_poly,offset,height_scale)
    texture_img=np.zeros((roof_texture.shape[0]+wall_texture.shape[0],
                          max(wall_texture.shape[1],roof_texture.shape[1]),3))
    texture_img[:roof_texture.shape[0],:roof_texture.shape[1]]=roof_texture
    texture_img[roof_texture.shape[0]:roof_texture.shape[0]+wall_texture.shape[0],:wall_texture.shape[1]]=wall_texture
    cv2.imwrite(outdir,texture_img[:,:,::-1])
    roof_texture_area=[0.,0.,roof_texture.shape[1]/max(wall_texture.shape[1],roof_texture.shape[1]),
                       roof_texture.shape[0]/texture_img.shape[0]]
    wall_texture_area=[0.,roof_texture_area[3],
                       wall_texture.shape[1]/max(wall_texture.shape[1],roof_texture.shape[1]),1.]
    return {'roof_texture_area':roof_texture_area,'wall_texture_area':wall_texture_area}

def show_roof_on_img(img,building_dict):
    plt.imshow(img)
    roof_poly = np.array(building_dict['roof_mask']).reshape(-1, 2)
    ax = plt.gca()
    ax.add_patch(plt.Polygon(xy=roof_poly, fill=False))
    plt.show()

def get_roof_texture(img,roof_poly):
    # 截取屋顶：
    # 获取图像范围内的部分
    # roof_poly = roof_poly - np.array([roof_bbox[0], roof_bbox[1]])
    roof_bbox=[min([round(vex[0]) for vex in roof_poly]),
             min([round(vex[1]) for vex in roof_poly]),
             max([round(vex[0]) for vex in roof_poly]),
             max([round(vex[1]) for vex in roof_poly]), ]
    roof_poly=(roof_poly-roof_bbox[:-2]).astype(int)
    roof_bbox_img = img[roof_bbox[1]:roof_bbox[3], roof_bbox[0]: roof_bbox[2]]
    # plt.imshow(roof_bbox_img)
    # plt.show()
    im = np.zeros(np.array([roof_bbox[3]-roof_bbox[1], roof_bbox[2]-roof_bbox[0]]), dtype="uint8")
    cv2.fillPoly(im, [roof_poly], 255)
    masked = cv2.bitwise_and(roof_bbox_img, roof_bbox_img, mask=im)
    # plt.imshow(masked)
    # plt.show()
    return masked

def get_wall_texture(img,roof_poly,offset,height_scale = 1):
    # 截取墙面纹理，按照building_dict['roof_mask']中屋顶多边形的存储顺序围一圈存储
    vex_right = np.argmax(roof_poly[:, 0])  # todo：按照offset的方向来，这个图示因为偏移方向特殊才成功的
    vex_left = np.argmin(roof_poly[:, 0])
    arc_flag = -1
    arcs = [[], []]
    arc_index = [-1] * len(roof_poly)  # 代表第i个顶点和i+1顶点构成的边在哪段弧中
    i = 0
    while arc_flag < 2:
        if arc_flag >= 0:
            arcs[arc_flag].append(roof_poly[i])
            arc_index[i] = arc_flag
            arc_index
        i = (i + 1) % len(roof_poly)
        if i == vex_left or i == vex_right:
            if arc_flag >= 0:
                arcs[arc_flag].append(roof_poly[i])  # 对于左右两个分界点，前一个弧的末尾和后一个弧的起始都是这个点
            arc_flag += 1
    front_arc_idx = np.argmax(
        [np.mean(np.array(arcs[0]), axis=0)[1], np.mean(np.array(arcs[1]), axis=0)[1]])  # y大的为前弧

    # 以下为显示face部分图片的代码，在face部分超出图片范围的时候可能报错（例如roof在最右侧的建筑物还朝左倾斜，或者反过来，总之即是foot在图片外）
    # front_arc = arcs[front_arc_idx]
    # front_poly = front_arc.copy()
    # for vex in front_arc[::-1]:
    #     front_poly.append(
    #         np.array([vex[0] - offset[0], vex[1] - offset[1]]))
    # front_face_box = [min([round(vex[0]) for vex in front_poly]),
    #                   min([round(vex[1]) for vex in front_poly]),
    #                   max([round(vex[0]) for vex in front_poly]),
    #                   max([round(vex[1]) for vex in front_poly]), ]
    # import matplotlib.pyplot as plt
    # face_bbox_img = img[front_face_box[1]:front_face_box[3], front_face_box[0]:front_face_box[2]]
    # front_poly = front_poly - np.array([front_face_box[0], front_face_box[1]])
    # im = np.zeros(np.array([front_face_box[3] - front_face_box[1], front_face_box[2] - front_face_box[0]]),
    #               dtype="uint8")
    # cv2.fillPoly(im, [front_poly.astype(int)], 255)
    # masked = cv2.bitwise_and(face_bbox_img, face_bbox_img, mask=im)
    # plt.imshow(masked)
    # plt.show()

    face_thre = 0.5
    #3857的坐标与高度的关系大概是1：10,即 用于绘制三维模型的高度数值*10=这段高度如果横过来，对应的坐标变化
    height = round(np.linalg.norm(offset) * height_scale*10)
    wall_texture_whole_pic = None
    for i, vex in enumerate(roof_poly):
        roof_edge = vex - roof_poly[(i + 1) % len(roof_poly)]
        wall_texture = np.zeros((height, round(np.linalg.norm(roof_edge)), 3))
        if arc_index[i] == front_arc_idx:
            if abs(roof_edge[0] / (roof_edge[1] + 1e-6)) > face_thre:  # todo:按照offset的方向来
                # 截取墙面图片
                l_wall = round(np.linalg.norm(roof_edge))
                wall_poly = [vex, roof_poly[(i + 1) % len(roof_poly)],
                             roof_poly[(i + 1) % len(roof_poly)] - offset, vex - offset]
                wall_bbox = [min([round(vex[0]) for vex in wall_poly]),
                             min([round(vex[1]) for vex in wall_poly]),
                             max([round(vex[0]) for vex in wall_poly]),
                             max([round(vex[1]) for vex in wall_poly]), ]
                wall_img = img[wall_bbox[1]:wall_bbox[3], wall_bbox[0]:wall_bbox[2]]
                if wall_bbox[2]>img.shape[1] or wall_bbox[3]>img.shape[0]: #若超出图像范围，补0
                    wall_img = np.pad(wall_img,((0,max(0,wall_bbox[3]-img.shape[0])),(0,max(0,wall_bbox[2]-img.shape[1])),(0,0)),'constant')
                wall_poly = wall_poly - np.array([wall_bbox[0], wall_bbox[1]])
                im = np.zeros(
                    np.array([wall_bbox[3] - wall_bbox[1], wall_bbox[2] - wall_bbox[0]]),
                    dtype="uint8")
                cv2.fillPoly(im, [wall_poly.astype(int)], 255)
                wall_img = cv2.bitwise_and(wall_img, wall_img, mask=im)
                # 变换墙面图片
                # 在原图像和目标图像上各选择三个点
                mat_src = np.float32(np.array([[0, 0], -np.array(offset), roof_edge]) +
                                     roof_poly[(i + 1) % len(roof_poly)] - wall_bbox[:2])
                mat_dst = np.float32([[0, 0], [0, height], [l_wall, 0]])
                # 得到变换矩阵
                mat_trans = cv2.getAffineTransform(mat_src, mat_dst)
                wall_texture = cv2.warpAffine(wall_img, mat_trans, (l_wall, height))
                # plt.imshow(wall_texture)
                # plt.show()
        if wall_texture_whole_pic is None:
            wall_texture_whole_pic = wall_texture
        else:
            wall_texture_whole_pic = np.hstack((wall_texture, wall_texture_whole_pic)).astype(
                np.uint)  # roof_poly顺时针，新图放在左边
    # plt.imshow(wall_texture_whole_pic)
    # plt.show()
    return wall_texture_whole_pic

def get_wall_texture_hori(img,roof_poly,offset,height_scale = 1):
    """
    截取墙面纹理，按照building_dict['roof_mask']中屋顶多边形的存储顺序围一圈存储
    Args:
        img:
        roof_poly:
        offset:
        height_scale:

    Returns:

    """
    # 先找到最上点和最下点
    bottom_vexes = roof_poly[roof_poly[:, 1] > roof_poly[:, 1].max() - 1]
    bottom_break_vex = np.argwhere(
        (roof_poly == bottom_vexes[bottom_vexes[:, 0] == bottom_vexes[:, 0].min()]).all(axis=1)).item()
    top_vexes = roof_poly[roof_poly[:, 1] < roof_poly[:, 1].min() + 1]
    top_break_vex = np.argwhere(
        (roof_poly == top_vexes[top_vexes[:, 0] == top_vexes[:, 0].min()]).all(axis=1)).item()
    # 用找到的最上/最下点分割roof多边形，分为左侧和右侧两端弧
    arc_flag = -1
    arcs = [[], []]
    arc_index = [-1] * len(roof_poly)  # 代表第i个顶点和i+1顶点构成的边在哪段弧中
    i = 0
    while arc_flag < 2:
        if arc_flag >= 0:
            arcs[arc_flag].append(roof_poly[i])
            arc_index[i] = arc_flag
            arc_index
        i = (i + 1) % len(roof_poly)
        if i == bottom_break_vex or i == top_break_vex:
            if arc_flag >= 0:
                arcs[arc_flag].append(roof_poly[i])  # 对于上下两个分界点，前一个弧的末尾和后一个弧的起始都是这个点
            arc_flag += 1
    front_arc_idx = np.argmin(
        [np.mean(np.array(arcs[0]), axis=0)[0], np.mean(np.array(arcs[1]), axis=0)[0]])  # x小的为前弧

    # 以下为显示face部分图片的代码，在face部分超出图片范围的时候可能报错（例如roof在最右侧的建筑物还朝左倾斜，或者反过来，总之即是foot在图片外）
    # front_arc = arcs[front_arc_idx]
    # front_poly = front_arc.copy()
    # for vex in front_arc[::-1]:
    #     front_poly.append(
    #         np.array([vex[0] - offset[0], vex[1] - offset[1]]))
    # front_face_box = [min([round(vex[0]) for vex in front_poly]),
    #                   min([round(vex[1]) for vex in front_poly]),
    #                   max([round(vex[0]) for vex in front_poly]),
    #                   max([round(vex[1]) for vex in front_poly]), ]
    # import matplotlib.pyplot as plt
    # face_bbox_img = img[front_face_box[1]:front_face_box[3], front_face_box[0]:front_face_box[2]]
    # front_poly = front_poly - np.array([front_face_box[0], front_face_box[1]])
    # im = np.zeros(np.array([front_face_box[3] - front_face_box[1], front_face_box[2] - front_face_box[0]]),
    #               dtype="uint8")
    # cv2.fillPoly(im, [front_poly.astype(int)], 255)
    # masked = cv2.bitwise_and(face_bbox_img, face_bbox_img, mask=im)
    # plt.imshow(masked)
    # plt.show()

    face_thre = 0.0
    #3857的坐标与高度的关系大概是1：10,即 用于绘制三维模型的高度数值*10=这段高度如果横过来，对应的坐标变化
    height = round(np.linalg.norm(offset) * height_scale*1)
    wall_texture_whole_pic = None
    for i, vex in enumerate(roof_poly):
        roof_edge = vex - roof_poly[(i + 1) % len(roof_poly)]
        wall_texture = np.zeros((height, round(np.linalg.norm(roof_edge)), 3))
        if arc_index[i] == front_arc_idx and vex[1]-roof_poly[(i + 1) % len(roof_poly),1]>1:# 是左边的弧且这段边的纵向跨度大于1像素
            if abs(roof_edge[0] / (roof_edge[1] + 1e-6)) > face_thre:
                # 截取墙面图片
                l_wall = round(np.linalg.norm(roof_edge))
                wall_poly = [vex, roof_poly[(i + 1) % len(roof_poly)],
                             roof_poly[(i + 1) % len(roof_poly)] - offset, vex - offset]
                wall_bbox = [min([round(vex[0]) for vex in wall_poly]),
                             min([round(vex[1]) for vex in wall_poly]),
                             max([round(vex[0]) for vex in wall_poly]),
                             max([round(vex[1]) for vex in wall_poly]), ]
                wall_img = img[max(0, wall_bbox[1]):wall_bbox[3], max(0, wall_bbox[0]):wall_bbox[2]]
                wall_img = np.pad(wall_img, ((max(0, -wall_bbox[1]), max(0, wall_bbox[3] - img.shape[0])),
                                               (max(0, -wall_bbox[0]), max(0, wall_bbox[2] - img.shape[1])), (0, 0)),
                                   'constant')
                wall_poly = wall_poly - np.array([wall_bbox[0], wall_bbox[1]])
                im = np.zeros(
                    np.array([wall_bbox[3] - wall_bbox[1], wall_bbox[2] - wall_bbox[0]]),
                    dtype="uint8")
                cv2.fillPoly(im, [wall_poly.astype(int)], 255)
                wall_img = cv2.bitwise_and(wall_img, wall_img, mask=im)
                # 将图片上的墙面图片通过仿射变换变为mesh墙面纹理
                # 在原图像和目标图像上各选择三个点
                mat_src = np.float32(np.array([[0, 0], -np.array(offset), roof_edge]) +
                                     roof_poly[(i + 1) % len(roof_poly)] - wall_bbox[:2])
                mat_dst = np.float32([[0, 0], [0, height], [l_wall, 0]])
                # 得到变换矩阵
                mat_trans = cv2.getAffineTransform(mat_src, mat_dst)
                wall_texture = cv2.warpAffine(wall_img, mat_trans, (l_wall, height))
                # plt.imshow(wall_texture)
                # plt.show()
        if wall_texture_whole_pic is None:
            wall_texture_whole_pic = wall_texture
        else:
            wall_texture_whole_pic = np.hstack((wall_texture, wall_texture_whole_pic)).astype(
                np.uint)  # roof_poly顺时针，新图放在左边
    # plt.imshow(wall_texture_whole_pic)
    # plt.show()
    return wall_texture_whole_pic

def TransPixCor2GeoCor(PixCor,GeoTrans):
    X = GeoTrans[0] + PixCor[0] * GeoTrans[1] + PixCor[1] * GeoTrans[2];
    Y = GeoTrans[3] + PixCor[0] * GeoTrans[4] + PixCor[1] * GeoTrans[5];
    return [X,Y]

def save_building2geojson(roof_polys,offsets,texture_areas_list,texture_files,outdir,height_scale=1,sr_epsg=3857,geo_trans = (-7.90e+06, 0.15, 0, 5.205e+06, 0, -0.15)):
    # 创建shp文件
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(sr_epsg)
    source_ds = ogr.GetDriverByName('GeoJSON').CreateDataSource(utf8_path=outdir)
    # 创建Driver的later
    source_lyr = source_ds.CreateLayer('poly', srs=sr, geom_type=ogr.wkbPolygon)
    # 为这个layer添加ID属性
    source_lyr.CreateField(ogr.FieldDefn('ID', ogr.OFTInteger))
    source_lyr.CreateField(ogr.FieldDefn('height', ogr.OFTReal))
    source_lyr.CreateField(ogr.FieldDefn('texture_img', ogr.OFTString))
    source_lyr.CreateField(ogr.FieldDefn('roof_texture_area_bottom', ogr.OFTReal))
    source_lyr.CreateField(ogr.FieldDefn('roof_texture_area_right', ogr.OFTReal))
    source_lyr.CreateField(ogr.FieldDefn('wall_texture_area_right', ogr.OFTReal))
    for i in range(len(roof_polys)):
        # 创建feature
        feat = ogr.Feature(source_lyr.GetLayerDefn())
        # 将polygons设置为feature的Geom
        ring = ogr.Geometry(ogr.wkbLinearRing)
        footprint_poly = roof_polys[i] - offsets[i]  # footprint+offset=roof
        for vertex in footprint_poly:
            ring.AddPoint(*TransPixCor2GeoCor(vertex, geo_trans))
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        feat.SetGeometryDirectly(polygon)
        # 设置feature的ID
        feat.SetField('ID', i)
        feat.SetField('height', float(np.linalg.norm(offsets[i]) * height_scale*0.6))# mesh scale
        feat.SetField('texture_img', texture_files[i])
        feat.SetField('roof_texture_area_bottom', texture_areas_list[i]['roof_texture_area'][3])
        feat.SetField('roof_texture_area_right', texture_areas_list[i]['roof_texture_area'][2])
        feat.SetField('wall_texture_area_right', texture_areas_list[i]['wall_texture_area'][2])
        # 将这个feature加入layer中
        source_lyr.CreateFeature(feat)

def save_base_map2geojson(base_image_file,outdir,img_shape=None,sr_epsg=3857,geo_trans = (-7.90e+06, 0.15, 0, 5.205e+06, 0, -0.15)):
    # 将输入遥感影像存储为底图，使之可以同时加载。
    # 其本质是要建立一个与base图片等大小的建筑物，但是没有高度，将图片作为屋顶的texture贴上去

    # 创建geojson文件
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(sr_epsg)
    source_ds = ogr.GetDriverByName('GeoJSON').CreateDataSource(utf8_path=outdir)
    # 创建Driver的later
    source_lyr = source_ds.CreateLayer('poly', srs=sr, geom_type=ogr.wkbPolygon)
    # 为这个layer添加ID属性
    source_lyr.CreateField(ogr.FieldDefn('ID', ogr.OFTInteger))
    source_lyr.CreateField(ogr.FieldDefn('height', ogr.OFTReal))
    source_lyr.CreateField(ogr.FieldDefn('texture_img', ogr.OFTString))
    source_lyr.CreateField(ogr.FieldDefn('roof_texture_area_bottom', ogr.OFTReal))
    source_lyr.CreateField(ogr.FieldDefn('roof_texture_area_right', ogr.OFTReal))
    source_lyr.CreateField(ogr.FieldDefn('wall_texture_area_right', ogr.OFTReal))

    if img_shape is None:
        pass# todo：读取图片文件并获取shape，修改下面写死的“1023”
    feat = ogr.Feature(source_lyr.GetLayerDefn())
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(*TransPixCor2GeoCor([0,0], geo_trans))# 顺序应当是顺时针方向
    ring.AddPoint(*TransPixCor2GeoCor([0,1023], geo_trans))
    ring.AddPoint(*TransPixCor2GeoCor([1023,1023], geo_trans))
    ring.AddPoint(*TransPixCor2GeoCor([1023,0], geo_trans))
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    feat.SetGeometryDirectly(polygon)
    # 设置feature的ID
    feat.SetField('ID', 0)
    feat.SetField('height', 0.0)
    feat.SetField('texture_img', base_image_file)
    feat.SetField('roof_texture_area_bottom', 1.0)
    feat.SetField('roof_texture_area_right', 1.0)
    feat.SetField('wall_texture_area_right', 0.0)
    # 将这个feature加入layer中
    source_lyr.CreateFeature(feat)

